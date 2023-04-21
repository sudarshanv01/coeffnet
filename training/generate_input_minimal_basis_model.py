import argparse

import random

import numpy as np

from instance_mongodb import instance_mongodb_sei

from monty.serialization import loadfn, dumpfn


def split_dataset(all_entries, train_frac, test_frac, validate_frac):
    """Split the dict based on the first key into three different fractions."""

    print(f"Number of entries in the dataset which will be split: {len(all_entries)}")

    random.seed(42)
    random.shuffle(all_entries)

    train_num = int(len(all_entries) * train_frac)
    test_num = int(len(all_entries) * test_frac)
    validate_num = int(len(all_entries) * validate_frac)

    train_list = all_entries[:train_num]
    test_list = all_entries[train_num : train_num + test_num]
    validate_list = all_entries[train_num + test_num :]

    print(f"Number of entries in the training set: {len(train_list)}")
    print(f"Number of entries in the test set: {len(test_list)}")
    print(f"Number of entries in the validation set: {len(validate_list)}")

    return train_list, test_list, validate_list


def get_cli_args():
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of the data to be used for training.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of the data to be used for testing.",
    )
    parser.add_argument(
        "--validate_frac",
        type=float,
        default=0.1,
        help="Fraction of the data to be used for validation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Create train, validation and test sets out of the data."""

    args = get_cli_args()

    print("Loading data from MongoDB...")
    db = instance_mongodb_sei(project="mlts")
    collection = db.sn2_reaction_dataset

    if args.debug:
        cursor = collection.find().limit(100).sort("tags.label", 1).allow_disk_use(True)
    else:
        cursor = collection.find().sort("tags.label", 1).allow_disk_use(True)

    data = list(cursor)
    print(f"Number of entries in the dataset: {len(data)}")
    for _dat in data:
        coeff_matrix = _dat["coeff_matrices"]
        if np.isnan(coeff_matrix).any():
            # Remove _dat from data
            data.remove(_dat)
    print(f"Number of entries in the dataset after pruning nan: {len(data)}")

    train_data, test_data, validate_data = split_dataset(
        data, args.train_frac, args.test_frac, args.validate_frac
    )

    if args.debug:
        dumpfn(train_data, "input_files/debug_train_data_interp_minimal_basis.json")
        dumpfn(test_data, "input_files/debug_test_data_interp_minimal_basis.json")
        dumpfn(
            validate_data, "input_files/debug_validate_data_interp_minimal_basis.json"
        )
    else:
        dumpfn(train_data, "input_files/train_data_interp_minimal_basis.json")
        dumpfn(test_data, "input_files/test_data_interp_minimal_basis.json")
        dumpfn(validate_data, "input_files/validate_data_interp_minimal_basis.json")
