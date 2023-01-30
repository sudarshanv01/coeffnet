import random

from instance_mongodb import instance_mongodb_sei

from monty.serialization import loadfn, dumpfn


def split_dataset(all_entries, train_frac, test_frac, validate_frac):
    """Split the dict based on the first key into three different fractions."""

    # Shuffle the list
    random.shuffle(all_entries)

    # Get the number of entries in each list
    train_num = int(len(all_entries) * train_frac)
    test_num = int(len(all_entries) * test_frac)
    validate_num = int(len(all_entries) * validate_frac)

    # Split the list
    train_list = all_entries[:train_num]
    test_list = all_entries[train_num : train_num + test_num]
    validate_list = all_entries[train_num + test_num :]

    return train_list, test_list, validate_list


if __name__ == "__main__":
    """Create train, validation and test sets out of the data."""

    db = instance_mongodb_sei(project="mlts")
    collection = db.minimal_basis_interpolated_sn2

    # Get all entries in this collection as a list
    cursor = collection.find()
    data = list(cursor)

    train_data, test_data, validate_data = split_dataset(data, 0.8, 0.1, 0.1)

    # Save the data
    dumpfn(train_data, "input_files/train_data_interp_minimal_basis.json")
    dumpfn(test_data, "input_files/test_data_interp_minimal_basis.json")
    dumpfn(validate_data, "input_files/validate_data_interp_minimal_basis.json")
