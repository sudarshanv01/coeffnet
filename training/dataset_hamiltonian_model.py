import os
import logging
import argparse

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import seaborn
import matplotlib.pyplot as plt

import networkx as nx

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
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
args = parser.parse_args()


def inspect_data(y, filename):
    """Make a histogram of y."""
    fig, ax = plt.subplots()
    seaborn.histplot(y, ax=ax)
    fig.savefig(f"output/{filename}.png", dpi=300)


if __name__ == "__main__":
    """Test a convolutional Neural Network based on the charge model."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    folder_string, PLOT_FOLDER = create_plot_folder()

    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))

    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    # Create the training and test datasets
    train_dataset = HamiltonianDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=GRAPH_GENERATION_METHOD,
        basis_file=inputs["basis_file"],
    )
    train_dataset.process()
    print(train_dataset)
    # print(f"Number of training barriers: {len(train_dataset.data.y)}")

    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    for batch in train_loader:
        print(batch)

    if args.inspect_data:
        inspect_data(train_dataset.data.y, "barrier_data")
        inspect_data(train_dataset.data.global_attr, "reaction_energy_data")
