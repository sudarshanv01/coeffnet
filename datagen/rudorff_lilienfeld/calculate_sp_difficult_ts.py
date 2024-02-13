import logging
import yaml
import argparse

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW
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

    collection = db.visualize_difficult_transition_state
    find_tags = {}

    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    for idx, document in enumerate(collection.find(find_tags)):

        molecule_dict = document.pop("molecule")
        molecule = Molecule.from_dict(molecule_dict)
        document.pop("_id")
        tags = document
        tags["group"] = "visualize_difficult_transition_state"
        print(tags)

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

        logger.info(f"{idx} structures processed for calculation.")
