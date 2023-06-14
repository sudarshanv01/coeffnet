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
)

wandb_api = wandb.Api()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_best_model(
    prediction_mode: str,
    basis_set: str,
    basis_set_type: str,
    df: pd.DataFrame,
    all_runs,
    device: torch.device,
) -> torch.nn.Module:
    """Get the best model for the given prediction mode."""
    df_options = df[
        (df["basis_set"] == basis_set)
        & (df["basis_set_type"] == basis_set_type)
        & (df["prediction_mode"] == prediction_mode)
    ]
    df_options = df_options[~df_options["val_loss"].isna()]
    df_options["val_loss"] = df_options["val_loss"].astype(float)
    while True:
        best_model_row = df_options.sort_values(by="val_loss").iloc[0]
        best_run = [
            run for run in all_runs if run.name == best_model_row["wandb_model_name"]
        ][0]
        print(f"Best model: {best_run.name}")
        best_artifacts = best_run.logged_artifacts()
        best_model = [
            artifact for artifact in best_artifacts if artifact.type == "model"
        ][0]
        best_model.download()
        try:
            best_model = torch.load(best_model.file())
        except RuntimeError:
            print("Failed to load model, skipping")
            df_options = df_options[
                df_options["val_loss"] != best_model_row["val_loss"]
            ]
            continue
        break
    best_model.eval()
    logger.info(f"Loaded model: {best_model_row['wandb_model_name']}")
    best_model.to(device)
    return best_model


def get_relative_energy_performance():
    """Get the relative energy performance of the model."""
    df = pd.DataFrame()
    for loader_name, loader in dataloaders.items():

        for data in loader:

            output = relative_energy_model(data)
            output = output.mean(dim=1)

            output = output.cpu().detach().numpy()
            expected = data.total_energy_transition_state - data.total_energy
            expected = expected.cpu().detach().numpy()

            output = output * ase_units.Ha
            expected = expected * ase_units.Ha

            data_to_store = {
                "output": output,
                "expected": expected,
                "loader": loader_name,
            }

            df = pd.concat([df, pd.DataFrame(data_to_store)], ignore_index=True)
    return df


def get_coeff_matrix_performance():
    """Get the coefficient matrix performance of the model."""
    df = pd.DataFrame()
    for loader_name, loader in dataloaders.items():

        for data in loader:

            output = coeff_matrix_model(data)
            output = output.cpu().detach().numpy()
            output = np.abs(output)

            expected = data.x_transition_state
            expected = expected.cpu().detach().numpy()
            expected = np.abs(expected)

            # Split both `output` and `expected` into s, p, d, f, g
            output_s = output[:, :max_s]
            expected_s = expected[:, :max_s]

            if max_p > 0:
                output_p = output[:, max_s : max_s + max_p]
                expected_p = expected[:, max_s : max_s + max_p]
            else:
                output_p = None
                expected_p = None

            if max_d > 0:
                output_d = output[:, max_s + max_p : max_s + max_p + max_d]
                expected_d = expected[:, max_s + max_p : max_s + max_p + max_d]
            else:
                output_d = None
                expected_d = None

            if max_f > 0:
                output_f = output[
                    :, max_s + max_p + max_d : max_s + max_p + max_d + max_f
                ]
                expected_f = expected[
                    :, max_s + max_p + max_d : max_s + max_p + max_d + max_f
                ]
            else:
                output_f = None
                expected_f = None

            if max_g > 0:
                output_g = output[
                    :,
                    max_s
                    + max_p
                    + max_d
                    + max_f : max_s
                    + max_p
                    + max_d
                    + max_f
                    + max_g,
                ]
                expected_g = expected[
                    :,
                    max_s
                    + max_p
                    + max_d
                    + max_f : max_s
                    + max_p
                    + max_d
                    + max_f
                    + max_g,
                ]
            else:
                output_g = None
                expected_g = None

            assert max_s + max_p + max_d + max_f + max_g == output.shape[1]

            for output, expected, basis_function_type in zip(
                [output_s, output_p, output_d, output_f, output_g],
                [expected_s, expected_p, expected_d, expected_f, expected_g],
                ["s", "p", "d", "f", "g"],
            ):
                if output is None:
                    continue
                data_to_store = {
                    "output": output.flatten(),
                    "expected": expected.flatten(),
                    "loader": loader_name,
                    "basis_function_type": basis_function_type,
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
        "--debug_model",
        action="store_true",
        help="Whether to use the debug model",
    )
    parser.add_argument(
        "--model_config",
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
    debug_dataset = args.debug_dataset
    debug_model = args.debug_model
    model_config = args.model_config
    inputs = read_inputs_yaml(model_config)
    basis_set_types = ["full", "minimal"]

    fig, ax = plt.subplots(2, 2, figsize=(4, 2.5), constrained_layout=True)

    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=debug_model,
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

        max_s = max_basis_functions["max_s"]
        max_p = max_basis_functions["max_p"]
        max_d = max_basis_functions["max_d"]
        max_f = max_basis_functions["max_f"]
        max_g = max_basis_functions["max_g"]

        df, all_runs = get_model_data(
            dataset_name=dataset_name,
            basis_set_type=basis_set_type,
            basis_set=basis_set_name,
            debug=debug_model,
        )

        relative_energy_model = get_best_model(
            prediction_mode="relative_energy",
            basis_set=basis_set_name,
            basis_set_type=basis_set_type,
            df=df,
            all_runs=all_runs,
            device=DEVICE,
        )
        logger.info(f"Using relative energy model: {relative_energy_model}")

        df_relative_energy = get_relative_energy_performance()
        train_loader_mask = df_relative_energy["loader"] == "train"
        if idx_type == 0:
            legend = True
        else:
            legend = False
        sns.scatterplot(
            data=df_relative_energy[train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            ax=ax[0, idx_type],
            palette="pastel",
            alpha=0.4,
            legend=legend,
        )
        sns.scatterplot(
            data=df_relative_energy[~train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            ax=ax[0, idx_type],
            palette="colorblind",
            legend=legend,
        )

        coeff_matrix_model = get_best_model(
            prediction_mode="coeff_matrix",
            basis_set=basis_set_name,
            basis_set_type=basis_set_type,
            df=df,
            all_runs=all_runs,
            device=DEVICE,
        )
        logger.info(f"Using coefficient matrix model: {coeff_matrix_model}")

        df_coeff_matrix = get_coeff_matrix_performance()
        train_loader_mask = df_coeff_matrix["loader"] == "train"

        sns.scatterplot(
            data=df_coeff_matrix[train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            style="basis_function_type",
            ax=ax[1, idx_type],
            palette="pastel",
            alpha=0.4,
            legend=legend,
        )
        sns.scatterplot(
            data=df_coeff_matrix[~train_loader_mask],
            x="expected",
            y="output",
            hue="loader",
            style="basis_function_type",
            ax=ax[1, idx_type],
            palette="colorblind",
            legend=legend,
        )

    handles, labels = ax[0, 1].get_legend_handles_labels()
    ax[1, 1].legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=1,
        fontsize=4,
    )
    # Remove the legend from the other plots
    ax[0, 0].get_legend().remove()
    ax[0, 1].get_legend().remove()
    ax[1, 0].get_legend().remove()
    ax[0, 0].set_ylabel(r"Predicted $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[0, 0].set_xlabel(r"DFT $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[1, 0].set_ylabel(r"Predicted $\left | \mathbf{C}_{\mathrm{TS}} \right |$")
    ax[1, 0].set_xlabel(r"DFT $ \left | \mathbf{C}_{\mathrm{TS}} \right |$")
    ax[0, 1].set_xlabel(r"DFT $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ (eV)")
    ax[0, 0].set_title("Full basis set")
    ax[0, 1].set_title("Minimal basis set")
    ax[1, 1].set_xlabel(r"DFT $\left | \mathbf{C}_{\mathrm{TS}} \right |$")
    ax[0, 1].set_ylabel("")
    ax[1, 1].set_ylabel("")

    # Set the axis limits for the C matrix plots to be between -0.1 and 1.1
    ax[1, 0].set_ylim([-0.1, 1.1])
    ax[1, 1].set_ylim([-0.1, 1.1])
    ax[1, 0].set_xlim([-0.1, 1.1])
    ax[1, 1].set_xlim([-0.1, 1.1])

    # Set the axis limits for the energy plots to be between the highest and lowest
    # value on the x-axis of 0,0
    ax[0, 0].set_ylim(ax[0, 0].get_xlim())
    ax[0, 1].set_ylim(ax[0, 0].get_xlim())
    ax[0, 0].set_xlim(ax[0, 0].get_xlim())
    ax[0, 1].set_xlim(ax[0, 0].get_xlim())

    # Draw parity lines for all plots
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(
                ax[i, j].get_xlim(),
                ax[i, j].get_xlim(),
                ls="--",
                c=".3",
                alpha=0.5,
                zorder=0,
            )

    fig.savefig(__output_folder__ / f"figure5_{basis_set_name}.png")
