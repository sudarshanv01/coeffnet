import logging
import yaml
import copy
import argparse

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

    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)
    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    count_structures = 0

    unique_rxn_numbers = initial_structure_collection.distinct("rxn_number")

    for rxn_number in unique_rxn_numbers:

        for state in ["reactant", "transition_state", "product"]:

            tags = {
                "functional": find_tags["functional"],
                "state": state,
                "quantities": ["nbo", "coeff_matrix"],
                "rxn_number": rxn_number,
            }

            if collection.count_documents({"tags": tags}) > 0:
                logger.info(f"Skipping {tags}")
                continue

            document = initial_structure_collection.find_one(
                {"rxn_number": rxn_number},
                {"_id": 0, state: 1},
            )

            molecule = document[state]
            molecule = Molecule.from_dict(molecule)

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
