import logging
import yaml
import copy
import argparse

from bson.objectid import ObjectId

import numpy as np

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import (
    FrequencyFW,
    FrequencyFlatteningTransitionStateFW,
)
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

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

    collection = db.rudorff_lilienfeld_calculation

    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    cursor = collection.find(
        {"task_label": "frequency calculation"},
        {
            "calcs_reversed.frequencies": 1,
            "tags.reaction_name": 1,
            "tags.rxn_number": 1,
        },
    )

    count_structures = 0
    for doc in cursor:

        frequencies = doc["calcs_reversed"][0]["frequencies"]
        frequencies = np.array(frequencies)
        num_imag_frequencies = np.sum(frequencies < 0)

        if num_imag_frequencies == 1:
            logger.info("Already found transition state for {}".format(doc["tags"]))
            continue

        count_structures += 1

        tags = copy.deepcopy(doc["tags"])

        molecule = doc["input"]["initial_molecule"]
        molecule = Molecule.from_dict(molecule)

        firew = FrequencyFlatteningTransitionStateFW(
            molecule=molecule,
            qchem_input_params=params,
            db_file="/global/home/users/svijay/fw_config/db_mlts.json",
            linked=True,
            freq_before_opt=True,
        )

        wf = Workflow([firew])
        wf = add_tags(wf, tags)

        if not args.dryrun:
            lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
