import logging
import yaml
import argparse

import numpy as np

from bson.objectid import ObjectId

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

lp = LaunchPad.from_file("/Users/sudarshanvijay/fw_config/my_launchpad_mlts.yaml")


def get_cli():
    args = argparse.ArgumentParser()
    args.add_argument("--dryrun", action="store_true", default=False)
    args.add_argument("--basis", type=str, default="sto-3g")
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.rudorff_lilienfeld_calculation
    data_collection = db.rudorff_lilienfeld_data
    initial_structure_collection = db.rudorff_lilienfeld_initial_structures
    find_tags = {}

    with open("config/spherical_only_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)
    params["overwrite_inputs"]["rem"]["basis"] = args.basis

    count_structures = 0
    documents = data_collection.find(
        find_tags,
        {
            # "perturbed_molecule_0_5_molecule": 1,
            # "perturbed_molecule_-0_5_molecule": 1,
            "transition_state_molecule": 1,
            "rxn_number": 1,
            "reaction_name": 1,
        },
    )
    documents = list(documents)

    for document in documents:

        keys = list(document.keys())
        keys = [key for key in keys if key.endswith("_molecule")]

        for state in keys:
            molecule_dict = document[state]
            molecule = Molecule.from_dict(molecule_dict)

            coordinates = np.array(molecule.cart_coords)
            _coordinates = coordinates.copy()
            _coordinates[:, [0, 1, 2]] = _coordinates[:, [2, 0, 1]]
            molecule = Molecule(
                species=molecule.species,
                coords=_coordinates,
                charge=molecule.charge,
                spin_multiplicity=molecule.spin_multiplicity,
            )

            tags = {
                "state": state,
                "quantities": ["coeff_matrix"],
                "rxn_number": document["rxn_number"],
                "reaction_name": document["reaction_name"],
                "constraints": "only carbon allowed to move",
                "inverted_coordinates": True,
                "basis_are_spherical": True,
            }

            count_structures += 1

            firew = SinglePointFW(
                molecule=molecule,
                qchem_input_params=params,
                extra_scf_print=True,
                db_file=">>db_file<<",
                spec={"_dupefinder": DupeFinderExact()},
            )

            wf = Workflow([firew])
            wf = add_tags(wf, tags)

            if not args.dryrun:
                lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
