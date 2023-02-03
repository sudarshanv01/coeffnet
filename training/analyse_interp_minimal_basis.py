import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse

import numpy as np

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import seaborn
import matplotlib.pyplot as plt

from utils import (
    get_train_data_path,
    get_validation_data_path,
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
)

if __name__ == "__main__":

    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))

    train_json_filename = inputs["train_json"]
    validation_json_filename = inputs["validate_json"]

    train_dataset = HamiltonianDataset(
        root=get_validation_data_path(),
        filename=validation_json_filename,
        basis_file=inputs["basis_file"],
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for data in train_loader:
        print(data)
        # print(data.x)
        # print(data.edge_index)
        # print(data.edge_attr)
        # print(data.y)
        break
