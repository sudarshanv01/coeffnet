from typing import Dict

import argparse

from pathlib import Path

import numpy as np
import numpy.typing as npt

import pandas as pd

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from coeffnet.predata.interpolator import GenerateParametersInterpolator
from coeffnet.dataset.reaction import ReactionDataset

from utils import (
    read_inputs_yaml,
)

from figure_utils import (
    get_sanitized_basis_set_name,
    get_dataloader_info,
)

import matplotlib.pyplot as plt
import seaborn as sns
from plot_params import get_plot_params

get_plot_params()
plt.rcParams["figure.dpi"] = 300

from model_functions import construct_model_name


def get_structure_prediction_performance(dataloaders):
    """Get the MAE of the structure prediction."""
    all_mae_norms = []
    for loader_name, loader in dataloaders.items():
        if loader_name != "train":
            continue
        for idx, data in enumerate(loader):

            interpolated_ts_coords = (
                data.pos_interpolated_transition_state.detach().numpy()
            )
            real_ts_coords = data.pos_transition_state.detach().numpy()
            difference_ts_coords = interpolated_ts_coords - real_ts_coords
            norm_difference_ts_coords = np.linalg.norm(difference_ts_coords, axis=1)

            mae = np.mean(norm_difference_ts_coords)
            all_mae_norms.append(mae)
    return all_mae_norms


def get_linear_model_performance(dataloaders, p: float = 0.5):
    p_prime = 1 - p
    all_mae_norms = []
    for loader_name, loader in dataloaders.items():
        if loader_name != "train":
            continue
        for idx, data in enumerate(loader):
            interpolated_ts_coords = p_prime * data.pos + p * data.pos_final_state
            real_ts_coords = data.pos_transition_state.detach().numpy()
            difference_ts_coords = interpolated_ts_coords - real_ts_coords
            norm_difference_ts_coords = np.linalg.norm(difference_ts_coords, axis=1)

            mae = np.mean(norm_difference_ts_coords)
            all_mae_norms.append(mae)
    return all_mae_norms


def get_node_features_performance(dataloaders):
    """Check the performance of the features against the real features."""
    residual = []
    for loader_name, loader in dataloaders.items():
        if loader_name != "train":
            continue
        for idx, data in enumerate(loader):
            p = data.p[0]
            p_prime = 1 - p

            x_interpolated = p_prime * data.x + p * data.x_final_state
            x_real = data.x_transition_state.detach().numpy()

            difference_x = x_interpolated - x_real

            residual.extend(difference_x.flatten().tolist())
    return residual


def get_linear_node_features_performance(dataloaders, p: float = 0.5):
    residual = []
    for loader_name, loader in dataloaders.items():
        if loader_name != "train":
            continue
        for idx, data in enumerate(loader):

            x_interpolated = p * data.x + (1 - p) * data.x_final_state
            x_real = data.x_transition_state.detach().numpy()

            difference_x = x_interpolated - x_real

            residual.extend(difference_x.flatten().tolist())

    return residual


def get_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot figure 4 of the manuscript.")
    parser.add_argument(
        "--basis_set",
        type=str,
        default="def2-svp",
        help="Basis set to use.",
    )
    parser.add_argument(
        "--basis_set_type",
        type=str,
        default="full",
        help="Basis set type to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Plot figure 4 of the manuscript.
    Panel a) Shows the truncated kernel that is used for the structure prediction.
    Panel b) Shows a histogram of the MAE of the structure prediction for the
             both SN2 and E2 reactions.
    """
    __input_folder__ = Path("input")
    __config_folder__ = Path("config")
    __output_folder__ = Path("output")
    __output_folder__.mkdir(exist_ok=True)
    args = get_cli_args()
    mu = 0.5
    sigma = 0.25
    alpha = 1
    num_samples = 10

    dataset_names = {
        r"S$_\mathrm{N}$2": "rudorff_lilienfeld_sn2_dataset",
    }
    basis_set_type = args.basis_set_type
    basis_set = args.basis_set
    debug_dataset = args.debug

    basis_set_name = get_sanitized_basis_set_name(basis_set)
    model_config = __config_folder__ / "rudorff_lilienfeld_model.yaml"
    inputs = read_inputs_yaml(model_config)

    fig, ax = plt.subplots(1, 3, figsize=(6, 2.0), constrained_layout=True)

    example_deltaG = [-3, 0, 3]  # eV

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, deltaG in enumerate(example_deltaG):
        generate_parameters_interpolator = GenerateParametersInterpolator(deltaG=deltaG)
        distribution = (
            generate_parameters_interpolator.truncated_skew_normal_distribution(
                lower=0,
                upper=1,
                x=np.linspace(-0.2, 1.2, 1000),
                mu=mu,
                sigma=sigma,
                alpha=alpha * deltaG,
            )
        )
        sampled_points = generate_parameters_interpolator.get_sampled_distribution(
            mu=mu,
            sigma=sigma,
            alpha=alpha * deltaG,
            num_samples=num_samples,
        )
        value_sampled_points = (
            generate_parameters_interpolator.truncated_skew_normal_distribution(
                lower=0,
                upper=1,
                x=sampled_points,
                mu=mu,
                sigma=sigma,
                alpha=alpha * deltaG,
            )
        )

        ax[2].plot(
            np.linspace(-0.2, 1.2, 1000),
            distribution,
            label=f"$\Delta G$ = {deltaG} eV",
            color=colors[idx],
        )
        ax[2].scatter(sampled_points, value_sampled_points, s=5, color=colors[idx])

    ax[2].set_xlabel("Interpolated coordinate", fontsize=9)
    ax[2].set_ylabel("Probability density", fontsize=9)
    ax[2].legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )

    for reaction_name, dataset_name in dataset_names.items():

        model_name = construct_model_name(
            dataset_name=dataset_name, debug=debug_dataset
        )
        input_foldername = (
            __input_folder__ / dataset_name / basis_set_type / basis_set_name
        )

        dataloaders, max_basis_functions = get_dataloader_info(
            input_foldername=input_foldername,
            model_name=model_name,
            debug=debug_dataset,
            **inputs["dataset_options"][f"{basis_set_type}_basis"],
        )

        mae_norms = get_structure_prediction_performance(dataloaders)
        p_0p1_mae_norms = get_linear_model_performance(dataloaders, p=0.1)
        p_0p5_mae_norms = get_linear_model_performance(dataloaders, p=0.5)
        p_0p9_mae_norms = get_linear_model_performance(dataloaders, p=0.9)

        sns.histplot(
            x=mae_norms,
            bins=20,
            label=r"$p\propto \Delta G$",
            element="step",
            alpha=0.2,
            ax=ax[0],
        )
        sns.histplot(
            x=p_0p1_mae_norms,
            bins=20,
            label=r"$p=0.1$",
            element="step",
            alpha=0.2,
            ax=ax[0],
        )
        sns.histplot(
            x=p_0p5_mae_norms,
            bins=20,
            label=r"$p=0.5$",
            element="step",
            alpha=0.2,
            ax=ax[0],
        )
        sns.histplot(
            x=p_0p9_mae_norms,
            bins=20,
            label=r"$p=0.9$",
            element="step",
            alpha=0.2,
            ax=ax[0],
        )

        residual_x = get_node_features_performance(dataloaders)
        residual_p_0p1_x = get_linear_node_features_performance(dataloaders, p=0.1)
        residual_p_0p5_x = get_linear_node_features_performance(dataloaders, p=0.5)
        residual_p_0p9_x = get_linear_node_features_performance(dataloaders, p=0.9)
        residual_x = np.abs(residual_x)
        residual_p_0p1_x = np.abs(residual_p_0p1_x)
        residual_p_0p5_x = np.abs(residual_p_0p5_x)
        residual_p_0p9_x = np.abs(residual_p_0p9_x)
        sns.histplot(
            x=residual_x,
            bins=50,
            label=r"$p\propto \Delta G$",
            element="step",
            alpha=0.2,
            ax=ax[1],
        )
        sns.histplot(
            x=residual_p_0p1_x,
            bins=20,
            label=r"$p=0.1$",
            element="step",
            alpha=0.2,
            ax=ax[1],
        )
        sns.histplot(
            x=residual_p_0p5_x,
            bins=50,
            label=r"$p=0.5$",
            element="step",
            alpha=0.2,
            ax=ax[1],
        )
        sns.histplot(
            x=residual_p_0p9_x,
            bins=20,
            label=r"$p=0.9$",
            element="step",
            alpha=0.2,
            ax=ax[1],
        )

    ax[0].set_xlabel("MAE [Å]", fontsize=9)
    ax[0].set_ylabel("Counts", fontsize=9)
    ax[2].set_title(r"c) $h\left( x \right)$")
    ax[0].set_title("a) Train-set MAE of $\mathbf{R}_{\mathrm{transition-state}}$")
    ax[1].set_title("b) Train-set MAE of $\mathbf{C}_{\mathrm{transition-state}}$")
    ax[1].set_xlabel(
        r"$\left|\mathbf{C}_{\mathrm{interpolated}} - \mathbf{C}_{\mathrm{DFT}}\right|$",
        fontsize=9,
    )
    ax[1].set_ylabel("Counts", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].legend(loc="best")

    fig.savefig(__output_folder__ / "figure4.png", dpi=300)
    fig.savefig(__output_folder__ / "figure4.pdf", dpi=300, transparent=True)
