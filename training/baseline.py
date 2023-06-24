from typing import List, Tuple, Dict, Union, Optional

import argparse

import logging

from pathlib import Path

from pprint import pprint

import numpy as np
import numpy.typing as npt

from ase import units as ase_units

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

import torch

import torch_geometric.transforms as T

from utils import (
    read_inputs_yaml,
)

import matplotlib.pyplot as plt
import seaborn as sns
from plot_params import get_plot_params

get_plot_params()

plt.rcParams["figure.dpi"] = 300

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from model_functions import construct_model_name
import wandb

from figure_utils import (
    get_sanitized_basis_set_name,
    get_dataloader_info,
)

wandb_api = wandb.Api()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def linear_model(train_loader):
    """Fit a line between the activation energy and the thermodynamic free energy."""

    x = []
    y = []

    for data in train_loader:
        y.append(data.total_energy_transition_state - data.total_energy)
        x.append(data.total_energy_final_state - data.total_energy)

    x = torch.cat(x)
    y = torch.cat(y)

    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    fit = np.polyfit(x, y, 1)
    return np.poly1d(fit)


def get_relative_energy_performance():
    """Get the relative energy performance of the model."""
    df = pd.DataFrame()
    for loader_name, loader in dataloaders.items():

        for data in loader:

            expected = data.total_energy_transition_state - data.total_energy
            expected = expected.cpu().detach().numpy()

            deltaE = data.total_energy_final_state - data.total_energy
            deltaE = deltaE.cpu().detach().numpy()
            output = model(deltaE)

            output = output * ase_units.Ha
            expected = expected * ase_units.Ha

            data_to_store = {
                "output": output,
                "expected": expected,
                "loader": loader_name,
            }

            df = pd.concat([df, pd.DataFrame(data_to_store)], ignore_index=True)
    return df


def get_cli_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot the results of the Rudorff-Lilienfeld dataset"
    )

    parser.add_argument(
        "--basis_set",
        type=str,
        default="def2-svp",
        help="The basis set to use for the dataset",
    )
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="Whether to use the debug dataset",
    )
    parser.add_argument(
        "--model_config", type=str, default="config/rudorff_lilienfeld_model.yaml"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rudorff_lilienfeld_sn2_dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":

    __input_folder__ = Path("input")
    __config_folder__ = Path("config")
    __output_folder__ = Path("output")
    __output_folder__.mkdir(exist_ok=True)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    args = get_cli_args()
    logger.info(f"Using args: {args}")

    dataset_name = args.dataset_name
    basis_set = args.basis_set
    basis_set_name = get_sanitized_basis_set_name(basis_set)
    debug_dataset = args.debug_dataset
    model_config = args.model_config
    inputs = read_inputs_yaml(model_config)
    basis_set_type = "full"

    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=False,
    )
    logger.info(f"Using model name: {model_name}")

    input_foldername = __input_folder__ / dataset_name / basis_set_type / basis_set_name

    dataloaders, max_basis_functions = get_dataloader_info(
        input_foldername=input_foldername,
        model_name=model_name,
        debug=debug_dataset,
        device=DEVICE,
        **inputs["dataset_options"][f"{basis_set_type}_basis"],
    )

    model = linear_model(dataloaders["train"])

    df = get_relative_energy_performance()

    # get the L1 loss of the model for each loader
    l1_train = np.abs(
        df[df["loader"] == "train"]["output"] - df[df["loader"] == "train"]["expected"]
    ).mean()
    l1_test = np.abs(
        df[df["loader"] == "test"]["output"] - df[df["loader"] == "test"]["expected"]
    ).mean()
    l1_val = np.abs(
        df[df["loader"] == "validation"]["output"]
        - df[df["loader"] == "validation"]["expected"]
    ).mean()

    print(f"Train L1: {l1_train}")
    print(f"Test L1: {l1_test}")
    print(f"Val L1: {l1_val}")

    # Plot the results in the form of a parity plot
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), constrained_layout=True)

    sns.scatterplot(
        data=df,
        x="expected",
        y="output",
        hue="loader",
        ax=ax,
        s=10,
        linewidth=0,
        alpha=0.5,
    )

    ax.set_xlabel("Expected barrier (eV)")
    ax.set_ylabel("Predicted barrier (eV)")

    # Draw the parity line
    x = np.linspace(df["expected"].min(), df["expected"].max(), 100)
    ax.plot(x, x, color="black", linestyle="--", linewidth=1)

    # make the aspect ratio equal
    # ax.set_aspect("equal", "box")

    fig.savefig(__output_folder__ / f"baseline.png", dpi=300)
    fig.savefig(__output_folder__ / f"baseline.pdf")
