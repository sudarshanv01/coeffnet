import argparse

from pathlib import Path

import torch

import logging

import numpy as np

from ase import data as ase_data
from ase import units as ase_units

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from PIL import Image

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt

import seaborn as sns

from figure_utils import (
    get_sanitized_basis_set_name,
)

from monty.serialization import loadfn, dumpfn

from model_functions import construct_model_name

from utils import read_inputs_yaml

from plot_params import get_plot_params

get_plot_params()


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
        default="config/rudorff_lilienfeld_model.yaml",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rudorff_lilienfeld_sn2_dataset",
    )
    parser.add_argument(
        "--grid_points_per_axis",
        type=int,
        default=200,
        help="The number of grid points per axis",
    )
    parser.add_argument(
        "--buffer_grid",
        type=float,
        default=1.5,
        help="The number of grid points to buffer the grid by",
    )
    parser.add_argument(
        "--basis_set_type",
        type=str,
        default="full",
        help="The type of basis set to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Plot figure8 of the manuscript. This figure will show the molecular orbitals of any
    molecule visualized in the full and minimal basis set."""

    # Set the tick font size to be 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

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
    basis_set_type = args.basis_set_type
    basis_set_name = get_sanitized_basis_set_name(basis_set)
    debug_dataset = args.debug_dataset
    debug_model = args.debug_model
    model_config = args.model_config
    inputs = read_inputs_yaml(model_config)

    fig, _ax = plt.subplots(3, 2, figsize=(5.5, 6), constrained_layout=True)

    input_foldername = __input_folder__ / dataset_name / basis_set_type / basis_set_name

    idx = 0

    mo_data = loadfn(
        __output_folder__ / f"figure8_{dataset_name}_{basis_set_name}.json"
    )

    for idx, data in enumerate(mo_data):
        ax = _ax[idx, 1]

        # Read in pdf file to plot directly into _ax[idx,0]
        char = chr(idx + ord("a"))
        pdf_file = __output_folder__ / f"figure8_part_{char}.png"
        pdf = plt.imread(pdf_file.as_posix())
        _ax[idx, 0].imshow(pdf)
        # Switch off axis
        _ax[idx, 0].axis("off")

        identifier = data["identifier"]
        species_tag = data["species_tag"]
        grid_output = data["grid_output"]
        grid_expected = data["grid_expected"]
        mo_output = data["mo_output"]
        mo_expected = data["mo_expected"]
        mo_output = np.asarray(mo_output)
        mo_expected = np.asarray(mo_expected)

        R1, R2, R3, R4, _X, _Y = data["species_tag"]
        print(R1, R2, R3, R4, _X, _Y)

        ax.plot(
            grid_output,
            mo_output,
            color="tab:red",
            ls="--",
        )
        ax.plot(
            grid_expected,
            mo_expected,
            color="tab:red",
        )

        ax2 = ax.twinx()
        ax2.plot(
            grid_output,
            mo_output**2,
            color="tab:blue",
            ls="--",
        )
        ax2.plot(
            grid_expected,
            mo_expected**2,
            color="tab:blue",
        )

        ax.set_ylabel(r"Highest Occupied $\psi \left(r\right)$", fontsize=12)
        ax2.set_ylabel(r"$\left | \psi \left(r\right) \right |^2$", fontsize=12)
        ax2.yaxis.label.set_color("tab:blue")
        ax2.tick_params(axis="y", colors="tab:blue")
        ax.yaxis.label.set_color("tab:red")
        ax.tick_params(axis="y", colors="tab:red")
        ax.tick_params(axis="y", which="minor", color="tab:red")
        ax2.tick_params(axis="y", which="minor", color="tab:blue")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlabel(r"$r$ ($\AA$)", fontsize=12)

        label = chr(ord("a") + idx)
        ax.set_title(f"{label}) {_X}$-$C$-${_Y}", fontsize=12)
        idx += 1

    ax.plot([], [], "-", color="k", label="Model")
    ax.plot([], [], "--", color="k", label="DFT")
    ax.legend(loc="upper left", fontsize=9)

    fig.savefig(
        __output_folder__ / f"figure8_{dataset_name}_{basis_set_name}.png",
        dpi=300,
    )

    fig.savefig(
        __output_folder__ / f"figure8_{dataset_name}_{basis_set_name}.pdf",
        dpi=300,
    )
