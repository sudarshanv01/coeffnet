import os

import argparse

import logging

from monty.serialization import dumpfn

from minimal_basis.predata.predata_qchem import TaskdocsToData

from yaml import safe_load

from instance_mongodb import instance_mongodb_sei

__input_folder__ = "input"
__config_folder__ = "config"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_command_line_arguments():
    """Using argparse decide if it is a debug run or not."""

    parser = argparse.ArgumentParser(
        description="Generate a list of dicts that contain data for the Grambow-Green dataset."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a debug version of the script.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Generate a list of dicts that contain data for the Grambow-Green dataset."""

    args = get_command_line_arguments()

    db = instance_mongodb_sei(project="mlts")
    collection = db[f"{args.dataset_name}_calculation"]
    logger.info(f"Using collection: {collection}")

    config = safe_load(
        open(
            os.path.join(__config_folder__, f"{args.dataset_name}_dataparser.yaml"), "r"
        )
    )
    logger.info(f"Using config: {config}")

    tastdocs_to_data = TaskdocsToData(collection=collection, **config)

    train_data, test_data, validation_data = tastdocs_to_data.get_random_split_data(
        train_frac=0.8,
        test_frac=0.1,
        validate_frac=0.1,
        debug=args.debug,
        seed=42,
        reparse=True,
    )

    if args.debug:
        dumpfn(train_data, f"input/debug_{args.dataset_name}_train.json")
        dumpfn(test_data, f"input/debug_{args.dataset_name}_test.json")
        dumpfn(validation_data, f"input/debug_{args.dataset_name}_validation.json")
    else:
        dumpfn(train_data, f"input/{args.dataset_name}_train.json")
        dumpfn(test_data, f"input/{args.dataset_name}_test.json")
        dumpfn(validation_data, f"input/{args.dataset_name}_validation.json")
