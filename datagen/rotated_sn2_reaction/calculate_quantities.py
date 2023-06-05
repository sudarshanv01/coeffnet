import logging
import yaml
import copy
import argparse

import numpy as np

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

lp = LaunchPad.from_file("/home/vijays/fw_config/my_launchpad_mlts.yaml")


def get_cli():
    args = argparse.ArgumentParser()
    args.add_argument("--dryrun", action="store_true", default=False)
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.rotated_sn2_reaction_calculation
    initial_structure_collection = db.rotated_sn2_reaction_initial_structures
    find_tags = {}

    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    count_structures = 0

    unique_idx = initial_structure_collection.distinct("idx")

    for idx in unique_idx:

        for state in ["reactant", "transition_state", "product"]:

            tags = {
                "state": state,
                "quantities": ["nbo", "coeff_matrix"],
                "idx": idx,
                "inverted_coordinates": True,
                "basis_are_spherical": True,
            }

            # Check if the calculation has already been performed.
            tags_check = copy.deepcopy(tags)
            tags_check = {f"tags.{k}": v for k, v in tags_check.items()}
            if collection.count_documents(tags_check) > 0:
                logger.info(f"Skipping {tags}")
                continue

            document = initial_structure_collection.find_one(
                {"idx": idx},
            )

            tags.update({"euler_angles": document["euler_angles"]})
            molecule = document[state + "_molecule"]
            molecule = Molecule.from_dict(molecule)

            coordinates = np.array(molecule.cart_coords)
            _coordinates = coordinates.copy()
            _coordinates[:, [0, 1, 2]] = _coordinates[:, [2, 0, 1]]

            # Create a new molecule with the inverted coordinates.
            molecule = Molecule(
                species=molecule.species,
                coords=_coordinates,
                charge=molecule.charge,
                spin_multiplicity=molecule.spin_multiplicity,
            )

            count_structures += 1

            firew = SinglePointFW(
                molecule=molecule,
                qchem_input_params=params,
                extra_scf_print=True,
                db_file=">>db_file<<",
            )

            wf = Workflow([firew])
            wf = add_tags(wf, tags)

            if not args.dryrun:
                lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
