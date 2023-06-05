import os

import argparse

import logging

from monty.serialization import dumpfn, loadfn

from minimal_basis.predata.matrices import TaskdocsToData

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
        "--collection_name",
        type=str,
        help="Name of the MongoDB collection from which to parse from.",
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        help="Name of the config file to use.",
    )
    parser.add_argument(
        "--basis_set_type",
        type=str,
        help="Type of basis set to use.",
        default="full",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Generate a list of dicts that contain data in the format required by the model."""

    args = get_command_line_arguments()

    if os.path.exists(__input_folder__) is False:
        os.mkdir(__input_folder__)

    db = instance_mongodb_sei(project="mlts")
    collection = db[f"{args.collection_name}"]
    logger.info(f"Using collection: {collection}")

    config = safe_load(open(os.path.join(f"{args.config_filename}"), "r"))
    basis_set_name = config.pop("basis_set_name")
    basis_set_name = basis_set_name.replace("*", "star")
    config["basis_info_raw"] = loadfn(
        os.path.join(__input_folder__, basis_set_name + ".json")
    )
    config["basis_set_type"] = args.basis_set_type

    tastdocs_to_data = TaskdocsToData(collection=collection, **config)

    train_data, test_data, validation_data = tastdocs_to_data.get_random_split_data(
        train_frac=0.8,
        test_frac=0.1,
        validate_frac=0.1,
        debug=args.debug,
        seed=42,
        reparse=True,
    )
    dataset_name = args.config_filename.split("/")[-1].split(".")[0]
    output_filename = f"input/{dataset_name}_{config['basis_set_type']}_basis"

    if args.debug:
        dumpfn(
            train_data,
            f"{output_filename}_train_debug.json",
        )
        dumpfn(
            test_data,
            f"{output_filename}_test_debug.json",
        )
        dumpfn(
            validation_data,
            f"{output_filename}_validate_debug.json",
        )
    else:
        dumpfn(
            train_data,
            f"{output_filename}_train.json",
        )
        dumpfn(test_data, f"{output_filename}_test.json")
        dumpfn(
            validation_data,
            f"{output_filename}_validate.json",
        )
