from typing import Dict

import argparse

from pathlib import Path

import numpy as np
import numpy.typing as npt

import pandas as pd

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.predata.interpolator import GenerateParametersInterpolator
from minimal_basis.dataset.reaction import ReactionDataset

from utils import (
    read_inputs_yaml,
)

from figure_utils import (
    get_sanitized_basis_set_name,
    get_dataloader_info,
)

import matplotlib.pyplot as plt
from plot_params import get_plot_params

get_plot_params()
plt.rcParams["figure.dpi"] = 300

from model_functions import construct_model_name


def get_structure_prediction_performance(dataloaders):
    """Get the MAE of the structure prediction."""
    all_mae_norms = []
    for loader in dataloaders.values():
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


def get_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot figure 4 of the manuscript.")
    parser.add_argument(
        "--basis_set",
        type=str,
        default="6-31g*",
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

    dataset_names = {
        r"S$_\mathrm{N}$2": "rudorff_lilienfeld_sn2_dataset",
        "E2": "rudorff_lilienfeld_e2_dataset",
    }
    basis_set_type = args.basis_set_type
    basis_set = args.basis_set
    debug_dataset = args.debug

    basis_set_name = get_sanitized_basis_set_name(basis_set)
    model_config = __config_folder__ / "rudorff_lilienfeld_model.yaml"
    inputs = read_inputs_yaml(model_config)

    fig, ax = plt.subplots(1, 2, figsize=(5.1, 2.4), constrained_layout=True)

    example_deltaG = [-3, 0, 3]  # eV

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, deltaG in enumerate(example_deltaG):
        generate_parameters_interpolator = GenerateParametersInterpolator(deltaG=deltaG)
        distribution = (
            generate_parameters_interpolator.truncated_skew_normal_distribution(
                lower=0,
                upper=1,
                x=np.linspace(-0.2, 1.2, 1000),
                mu=0.5,
                sigma=0.25,
                alpha=1.0 * deltaG,
            )
        )
        sampled_points = generate_parameters_interpolator.get_sampled_distribution(
            mu=0.5,
            sigma=0.25,
            alpha=1.0 * deltaG,
            num_samples=10,
        )
        value_sampled_points = (
            generate_parameters_interpolator.truncated_skew_normal_distribution(
                lower=0,
                upper=1,
                x=sampled_points,
                mu=0.5,
                sigma=0.25,
                alpha=1.0 * deltaG,
            )
        )

        ax[0].plot(
            np.linspace(-0.2, 1.2, 1000),
            distribution,
            label=f"$\Delta G$ = {deltaG} eV",
            color=colors[idx],
        )
        ax[0].scatter(sampled_points, value_sampled_points, s=10, color=colors[idx])

    ax[0].set_xlabel("Interpolated coordinate")
    ax[0].set_ylabel("Probability density")
    ax[0].legend(fontsize=5.5, loc="upper left")

    for reaction_name, dataset_name in dataset_names.items():

        model_name = construct_model_name(
            dataset_name=dataset_name, debug=debug_dataset
        )
        input_foldername = (
            __input_folder__ / dataset_name / basis_set_type / basis_set_name
        )

        dataloaders = get_dataloader_info(
            input_foldername=input_foldername,
            model_name=model_name,
            basis_set_type=basis_set_type,
            basis_set_name=basis_set_name,
            debug=debug_dataset,
            **inputs["dataset_options"][f"{basis_set_type}_basis"],
        )

        mae_norms = get_structure_prediction_performance(dataloaders)

        ax[1].hist(mae_norms, bins=20, label=reaction_name, alpha=0.5)

    ax[1].set_xlabel("MAE of structure prediction [Å]")
    ax[1].set_ylabel("Counts")
    ax[1].legend()

    # Label panels a) and b) on the top left
    ax[0].text(
        -0.1,
        1.1,
        "a)",
        transform=ax[0].transAxes,
        va="top",
    )
    ax[1].text(
        -0.1,
        1.1,
        "b)",
        transform=ax[1].transAxes,
        va="top",
    )

    fig.savefig(__output_folder__ / "figure4.png")
