import logging
import yaml
import copy
import argparse

import numpy as np

from bson.objectid import ObjectId

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW, ForceFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

lp = LaunchPad.from_file("/global/u1/s/svijay/fw_config/my_launchpad_mlts.yaml")


def get_cli():
    args = argparse.ArgumentParser()
    args.add_argument("--dryrun", action="store_true", default=False)
    args.add_argument("--basis", type=str, default="def2-svp")
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.grambow_green_calculation
    initial_structure_collection = db.grambow_green_initial_structures
    find_tags = {"functional": "b97d3"}

    with open("config/spherical_only_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)
    params["overwrite_inputs"]["rem"]["basis"] = args.basis

    count_structures = 0

    unique_rxn_numbers = initial_structure_collection.distinct("rxn_number")

    for rxn_number in unique_rxn_numbers:

        for state in ["reactant", "transition_state", "product"]:

            tags = {
                "functional": find_tags["functional"],
                "state": state,
                "quantities": ["nbo", "coeff_matrix"],
                "rxn_number": rxn_number,
                "inverted_coordinates": True,
                "basis_are_spherical": True,
            }

            # if collection.count_documents({"tags": tags}) > 0:
            #     logger.info(f"Skipping {tags}")
            #     continue

            document = initial_structure_collection.find_one(
                {"rxn_number": rxn_number},
                {"_id": 0, state: 1},
            )

            molecule = document[state]
            molecule = Molecule.from_dict(molecule)

            coordinates = np.array(molecule.cart_coords)
            _coordinates = coordinates.copy()
            _coordinates[:, [0, 1, 2]] = _coordinates[:, [2, 0, 1]]
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
