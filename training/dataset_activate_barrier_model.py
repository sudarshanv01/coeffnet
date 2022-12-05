import os
import argparse

from utils import (
    read_inputs_yaml,
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
)

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_classifier import ActivationBarrierDataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reprocess_dataset",
    action="store_true",
)
args = parser.parse_args()

if __name__ == "__main__":

    # --- Inputs
    inputs = read_inputs_yaml(
        os.path.join("input_files", "activation_barrier_model.yaml")
    )
    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]
    dataset_parameters_filename = inputs["dataset_json"]
    batch_size = inputs["batch_size"]

    # Create the training and test datasets
    train_dataset = ActivationBarrierDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        filename_classifier_parameters=dataset_parameters_filename,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = ActivationBarrierDataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        filename_classifier_parameters=dataset_parameters_filename,
    )
    if args.reprocess_dataset:
        validate_dataset.process()

    # Figure out the number of features
    num_node_features = train_dataset.num_node_features
    num_edge_features = train_dataset.num_edge_features
    num_global_features = train_dataset.num_global_features

    global_attr = train_dataset.data.global_attr
    # Reshape the global_attr to be a 2D array
    global_attr = global_attr.reshape(-1, num_global_features)
    interpolated_ts_energies = global_attr[:, -1]
    # Get the target values
    target = train_dataset.data.y

    # Do the same for the validation dataset
    global_attr_val = validate_dataset.data.global_attr
    # Reshape the global_attr to be a 2D array
    global_attr_val = global_attr_val.reshape(-1, num_global_features)
    # Get the interpolated TS energies
    interpolated_ts_energies_val = global_attr_val[:, -1]
    # Get the target values
    target_val = validate_dataset.data.y

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Histogram of the difference between the interpolated and target values
    ax.hist(target - interpolated_ts_energies, bins=100, color="tab:blue")
    ax.hist(target_val - interpolated_ts_energies_val, bins=100, color="tab:red")
    ax.set_xlabel("Target - Interpolated (eV)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Histogram of the difference between the interpolated and target values"
    )
    fig.savefig("output/interpolated_ts_energies.png", dpi=300)
