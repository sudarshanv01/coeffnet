import os

from pathlib import Path

import json

import argparse

import logging

from monty.serialization import dumpfn, loadfn

import basis_set_exchange as bse

from coeffnet.predata.matrices import TaskdocsToData

from yaml import safe_load

from instance_mongodb import instance_mongodb_sei

__input_folder__ = "input"

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
    parser.add_argument(
        "--basis_set",
        type=str,
        help="Name of the basis set to use.",
        default=None,
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Name of the folder to store the input files.",
        default=__input_folder__,
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

    if args.basis_set is not None:
        config["filter_collection"]["orig.rem.basis"] = args.basis_set

    basis_set_name = args.basis_set.replace("*", "star")
    basis_set_name = basis_set_name.replace("+", "plus")
    basis_set_name = basis_set_name.replace("(", "")
    basis_set_name = basis_set_name.replace(")", "")
    basis_set_name = basis_set_name.replace(",", "")
    basis_set_name = basis_set_name.replace(" ", "_")
    basis_set_name = basis_set_name.lower()
    basis_info_raw = bse.get_basis(
        args.basis_set, fmt="json", elements=list(range(1, 36))
    )
    basis_info_raw = json.loads(basis_info_raw)
    config["basis_info_raw"] = basis_info_raw
    config["basis_set_type"] = args.basis_set_type

    tastdocs_to_data = TaskdocsToData(collection=collection, **config)

    train_data, test_data, validate_data = tastdocs_to_data.get_random_split_data(
        train_frac=0.8,
        test_frac=0.1,
        validate_frac=0.1,
        debug=args.debug,
        seed=42,
        reparse=True,
    )
    logger.info(f"Number of training data: {len(train_data)}")
    logger.info(f"Number of test data: {len(test_data)}")
    logger.info(f"Number of validation data: {len(validate_data)}")

    dataset_name = args.config_filename.split("/")[-1].split(".")[0]
    basis_set_name = args.basis_set.replace("*", "star")
    output_foldername = (
        Path(args.input_folder)
        / dataset_name
        / config["basis_set_type"]
        / basis_set_name
    )
    output_foldername.mkdir(parents=True, exist_ok=True)

    if args.debug:
        dumpfn(
            train_data,
            output_foldername / "train_debug.json",
        )
        dumpfn(
            test_data,
            output_foldername / "test_debug.json",
        )
        dumpfn(
            validate_data,
            output_foldername / "validate_debug.json",
        )
    else:
        dumpfn(
            train_data,
            output_foldername / "train.json",
        )
        dumpfn(test_data, output_foldername / "test.json")
        dumpfn(
            validate_data,
            output_foldername / "validate.json",
        )
