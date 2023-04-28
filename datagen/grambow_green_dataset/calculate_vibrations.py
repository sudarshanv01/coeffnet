import logging
import yaml
import copy
import argparse

from bson.objectid import ObjectId

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import FrequencyFW
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

    count_structures = 0
    for document in initial_structure_collection.find(find_tags).limit(500):

        for state in ["reactant", "transition_state", "product"]:
            molecule = document[state]
            molecule = Molecule.from_dict(molecule)

            tags = {
                "functional": find_tags["functional"],
                "state": state,
                "quantities": ["vibrations"],
                "rxn_number": document["rxn_number"],
            }

            if collection.count_documents({"tags": tags}) > 0:
                continue

            count_structures += 1

            firew = FrequencyFW(
                molecule=molecule,
                qchem_input_params=params,
                db_file=">>db_file<<",
            )

            wf = Workflow([firew])
            wf = add_tags(wf, tags)

            if not args.dryrun:
                lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
