import os
import logging
import argparse

import numpy as np

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import seaborn
import matplotlib.pyplot as plt

from utils import (
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
)

if __name__ == "__main__":

    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))
