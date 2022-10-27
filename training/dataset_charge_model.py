import os
import logging
import argparse

import torch

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel

import seaborn
import matplotlib.pyplot as plt

from utils import (
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
)

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "charge_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inspect_data",
    action="store_true",
    help="If set, the data is inspected through a histogram of the outputs.",
)
args = parser.parse_args()


def inspect_data(y):
    """Make a histogram of y."""
    fig, ax = plt.subplots()
    seaborn.histplot(y, ax=ax)
    fig.savefig("output/charge_model_histogram.png", dpi=300)


if __name__ == "__main__":
    """Test a convolutional Neural Network based on the charge model."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    folder_string, PLOT_FOLDER = create_plot_folder()

    inputs = read_inputs_yaml(os.path.join("input_files", "charge_model.yaml"))

    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    train_json_filename = inputs["train_json"]
    BATCH_SIZE = inputs["batch_size"]

    # Create the training and test datasets
    train_dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=GRAPH_GENERATION_METHOD,
    )
    train_dataset.process()
    print(f"Number of training barriers: {len(train_dataset.data.y)}")

    if args.inspect_data:
        inspect_data(train_dataset.data.y)

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for train_batch in train_loader:
        print(train_batch)
        print(train_batch.num_features)
        print(train_batch.num_graphs)
