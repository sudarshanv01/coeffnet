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
    args.add_argument("--basis", type=str, default="6-31g*")
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.rotated_water_calculations
    initial_structure_collection = db.rotated_water_initial_structures

    params = {
        "dft_rung": 1,
        "overwrite_inputs": {
            "rem": {
                "basis": args.basis,
            }
        },
    }

    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    count_structures = 0

    for document in initial_structure_collection.find({}):

        tags = {
            "angles": document["angles"],
        }

        molecule = Molecule.from_dict(document["molecule"])

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
