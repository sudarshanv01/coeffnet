import os

import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.dataset_reaction import ReactionDataset
from minimal_basis.transforms.absolute import Absolute

from utils import (
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

from ase import units as ase_units
from ase.data import atomic_numbers, atomic_names

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# import wandb
# run = wandb.init()


inputs = read_inputs_yaml(os.path.join("config", "grambow_green_model.yaml"))

train_json_filename = inputs["debug_train_json"]
validate_json_filename = inputs["debug_validate_json"]
kwargs_dataset = inputs["dataset_options"]
kwargs_dataset["use_minimal_basis_node_features"] = inputs[
    "use_minimal_basis_node_features"
]

train_dataset = ReactionDataset(
    root=get_train_data_path(),
    filename=train_json_filename,
    basis_filename=inputs["basis_file"],
    **kwargs_dataset
)

validation_dataset = ReactionDataset(
    root=get_validation_data_path(),
    filename=validate_json_filename,
    basis_filename=inputs["basis_file"],
    **kwargs_dataset
)
