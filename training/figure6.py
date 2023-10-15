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
    get_model_data,
    get_best_model,
)

wandb_api = wandb.Api()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# There is something wrong with these entries in the dataset
IGNORE_LIST = [
    "E_D_E_E_A_A",
]


def get_relative_energy_performance():
    """Get the relative energy performance of the model."""
    df = pd.DataFrame()
    for loader_name, loader in dataloaders.items():

        for data in loader:

            output = relative_energy_model(data)

            output = output.cpu().detach().numpy()
            expected = data.total_energy_transition_state - data.total_energy
            expected = expected.cpu().detach().numpy()

            output = output * ase_units.Ha
            expected = expected * ase_units.Ha

            error = np.abs(output - expected)

            if data.identifier[0] in IGNORE_LIST:
                continue

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
        "--debug",
        action="store_true",
        help="Whether to use the debug dataset",
    )
    parser.add_argument(
        "--model_config",
        default="config/rudorff_lilienfeld_model.yaml",
        type=str,
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
    debug_dataset = args.debug
    model_config = args.model_config
    inputs = read_inputs_yaml(model_config)
    basis_set_types = ["minimal", "full"]

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(2.8, 1.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    figs, axs = plt.subplots(
        1,
        2,
        figsize=(3.5, 1.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=False,
    )
    logger.info(f"Using model name: {model_name}")

    for idx_type, basis_set_type in enumerate(basis_set_types):

        input_foldername = (
            __input_folder__ / dataset_name / basis_set_type / basis_set_name
        )

        dataloaders, max_basis_functions = get_dataloader_info(
            input_foldername=input_foldername,
            model_name=model_name,
            debug=debug_dataset,
            device=DEVICE,
            **inputs["dataset_options"][f"{basis_set_type}_basis"],
        )

        df, all_runs = get_model_data(
            dataset_name=dataset_name,
            basis_set_type=basis_set_type,
            basis_set=basis_set_name,
            debug=False,
        )

        relative_energy_model = get_best_model(
            prediction_mode="relative_energy",
            basis_set=basis_set_name,
            basis_set_type=basis_set_type,
            df=df,
            all_runs=all_runs,
            device=DEVICE,
        )

        df_relative_energy = get_relative_energy_performance()
        train_loader_mask = df_relative_energy["loader"].isin(["train", "validation"])
        if idx_type == 0:
            legend = False
        else:
            legend = True
        sns.scatterplot(
            data=df_relative_energy[train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            ax=axs[0, idx_type],
            palette="pastel",
            alpha=0.2,
            legend=legend,
        )
        sns.scatterplot(
            data=df_relative_energy[~train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            ax=ax[0, idx_type],
            palette="pastel",
            alpha=0.5,
            legend=False,
        )

    # Place legend outside of plot for axs
    handles, labels = axs[0, 1].get_legend_handles_labels()
    handles = [
        h
        for h, l in zip(handles, labels)
        if "loader" not in l and "basis_function_type" not in l
    ]
    labels = [l for l in labels if "loader" not in l and "basis_function_type" not in l]
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    axs[0, 1].legend(
        unique_handles, unique_labels, loc="upper left", bbox_to_anchor=(1.05, 1)
    )

    ax[0, 0].set_ylabel(r"Predicted $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[0, 0].set_xlabel(r"DFT $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[0, 1].set_xlabel(r"DFT $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[0, 1].set_ylabel("")

    # Draw parity lines for all plots
    for i in range(2):
        ax[0, i].plot(
            ax[0, i].get_xlim(),
            ax[0, i].get_xlim(),
            ls="--",
            c=".3",
            alpha=0.5,
            zorder=0,
        )
        ax[0, i].set_aspect("equal", "box")
        character = chr(ord("a") + i)
        ax[0, i].set_title(f"{character}) {basis_set_types[i].capitalize()} basis")

        axs[0, i].plot(
            axs[0, i].get_xlim(),
            axs[0, i].get_xlim(),
            ls="--",
            c=".3",
            alpha=0.5,
            zorder=0,
        )
        axs[0, i].set_aspect("equal", "box")
        axs[0, i].set_title(f"{character}) {basis_set_types[i].capitalize()} basis")

    fig.savefig(
        __output_folder__ / f"figure6_{dataset_name}_{basis_set_name}.png", dpi=300
    )
    fig.savefig(
        __output_folder__ / f"figure6_{dataset_name}_{basis_set_name}.pdf",
        dpi=300,
        transparent=True,
    )

    figs.savefig(
        __output_folder__ / f"figure6s_{dataset_name}_{basis_set_name}.png", dpi=300
    )
    figs.savefig(
        __output_folder__ / f"figure6s_{dataset_name}_{basis_set_name}.pdf",
        dpi=300,
        transparent=True,
    )
