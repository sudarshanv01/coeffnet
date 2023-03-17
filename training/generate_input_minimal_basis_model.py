import argparse

import random

from instance_mongodb import instance_mongodb_sei

from monty.serialization import loadfn, dumpfn


def split_dataset(all_entries, train_frac, test_frac, validate_frac):
    """Split the dict based on the first key into three different fractions."""

    print(f"Number of entries in the dataset: {len(all_entries)}")
    # Find all entries in all_entries that have something in 'tags.failure'
    # If they do, they directly go into the training set
    error_train_list = []
    pruned_entries = []
    for entry in all_entries:
        if len(entry["tags"]["failure"]) > 0:
            error_train_list.append(entry)
        else:
            pruned_entries.append(entry)

    random.seed(42)
    random.shuffle(all_entries)

    # Make sure all the entries now have positive barriers
    for entry in pruned_entries:
        state = entry["state"]
        barriers = (
            entry["final_energy"][state.index("transition_state")]
            - entry["final_energy"][state.index("initial_state")]
        )
        assert barriers > 0

    train_num = int(len(pruned_entries) * train_frac)
    test_num = int(len(pruned_entries) * test_frac)
    validate_num = int(len(pruned_entries) * validate_frac)

    train_list = pruned_entries[:train_num]
    # Add the error entries to the training set
    train_list.extend(error_train_list)
    # Shuffle the training set
    random.shuffle(train_list)
    test_list = pruned_entries[train_num : train_num + test_num]
    validate_list = pruned_entries[train_num + test_num :]

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
    parser.add_argument(
        "--pytest", action="store_true", help="Create dataset for pytest."
    )
    parser.add_argument(
        "--pytest_inputdir",
        type=str,
        default="../tests/inputs",
        help="Directory to save the pytest input files.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Create train, validation and test sets out of the data."""

    args = get_cli_args()

    print("Loading data from MongoDB...")
    db = instance_mongodb_sei(project="mlts")
    collection = db.minimal_basis_interpolated_sn2

    # Get all entries in this collection as a list
    if args.debug:
        cursor = collection.find().limit(100)
    elif args.pytest:
        cursor = collection.find().limit(10)
    else:
        cursor = collection.find()

    data = list(cursor)

    train_data, test_data, validate_data = split_dataset(
        data, args.train_frac, args.test_frac, args.validate_frac
    )

    # Save the data
    if args.debug:
        # Prepend the debug tag
        dumpfn(train_data, "input_files/debug_train_data_interp_minimal_basis.json")
        dumpfn(test_data, "input_files/debug_test_data_interp_minimal_basis.json")
        dumpfn(
            validate_data, "input_files/debug_validate_data_interp_minimal_basis.json"
        )
    elif args.pytest:
        dumpfn(
            train_data, f"{args.pytest_inputdir}/train_data_interp_minimal_basis.json"
        )
        dumpfn(test_data, f"{args.pytest_inputdir}/test_data_interp_minimal_basis.json")
        dumpfn(
            validate_data,
            f"{args.pytest_inputdir}/validate_data_interp_minimal_basis.json",
        )
    else:
        dumpfn(train_data, "input_files/train_data_interp_minimal_basis.json")
        dumpfn(test_data, "input_files/test_data_interp_minimal_basis.json")
        dumpfn(validate_data, "input_files/validate_data_interp_minimal_basis.json")
