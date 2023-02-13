import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse

import numpy as np

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from monty.serialization import loadfn, dumpfn

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import seaborn
import matplotlib.pyplot as plt

import plotly.express as px

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

    # Get the schema of the validation_json_dataset
    validation_json = loadfn(validation_json_filename)
    schema = validation_json[0].keys()

    train_dataset = HamiltonianDataset(
        root=get_validation_data_path(),
        filename=validation_json_filename,
        basis_file=inputs["basis_file"],
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for data in train_loader:
        print(data)
        minimal_basis_H = data.x
        minimal_basis_H = minimal_basis_H.reshape((-1, 2, 9, 9))
        edge_attr = data.edge_attr
        edge_attr = edge_attr.reshape((-1, 2, 9, 9))

        # Plot both `minimal_basis_H` and `edge_attr` together
        # as subplots with the animation for the first dimension
        for i in range(2):
            fig = px.imshow(
                minimal_basis_H[:, i, :, :],
                labels=dict(x="Basis", y="Basis", color="Value"),
                animation_frame=0,
            )
            fig.show()
        # for i in range(2):
        #     fig = px.imshow(
        #         edge_attr[:, i, :, :],
        #         labels=dict(x="Basis", y="Basis", color="Value"),
        #         animation_frame=0,
        #     )
        #     fig.show()

        break
