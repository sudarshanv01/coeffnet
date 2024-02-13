from pathlib import Path

import argparse

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import seaborn as sns

from figure_utils import get_sanitized_basis_set_name

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
    """Plot figure 6 of the manuscript."""

    __output_folder__ = Path("output")

    args = get_cli_args()

    dataset_name = args.dataset_name
    basis_set = args.basis_set
    basis_set_name = get_sanitized_basis_set_name(basis_set)
    basis_set_types = ["minimal", "full"]

    fig, ax = plt.subplots(2, 2, figsize=(4, 3.0), constrained_layout=True)

    for idx, basis_set_type in enumerate(basis_set_types):
        __output_file__ = (
            __output_folder__
            / f"{dataset_name}_{basis_set_name}_{basis_set_type}_mo.csv"
        )
        if args.debug:
            __output_file__ = __output_file__.with_name(
                __output_file__.stem + "_debug" + __output_file__.suffix
            )
        df_mo = pd.read_csv(__output_file__)
        df_mo = df_mo[df_mo["loader"] == "test"]

        __output_file__ = (
            __output_folder__
            / f"{dataset_name}_{basis_set_name}_{basis_set_type}_coeff_matrix.csv"
        )
        if args.debug:
            __output_file__ = __output_file__.with_name(
                __output_file__.stem + "_debug" + __output_file__.suffix
            )

        df_coeff = pd.read_csv(__output_file__)

        df_coeff["diff"] = df_coeff["expected"] - df_coeff["output"]
        df_coeff = df_coeff[df_coeff["loader"] == "test"]

        # Make a histogram of the coefficient differences
        sns.histplot(
            x="diff",
            data=df_coeff,
            ax=ax[idx, 0],
            hue="basis_function_type",
            bins=50,
            palette=sns.color_palette("hls", 3),
            element="step",
        )
        # Remove header from the legend
        legend = ax[idx, 0].get_legend()
        legend.set_title(None)

        # Show the histplot on a darker background
        # the color is based on a log scale
        sns.histplot(
            x="expected",
            y="output",
            data=df_mo,
            ax=ax[idx, 1],
            cbar=True,
            bins=50,
            norm=LogNorm(),
            vmin=None,
            vmax=None,
        )

        ax[idx, 1].plot(
            [df_mo["expected"].min(), df_mo["expected"].max()],
            [df_mo["expected"].min(), df_mo["expected"].max()],
            color="gray",
            linestyle="--",
            alpha=0.2,
        )

    ax[0, 0].set_xlabel(r"$\mathbf{C}^{\mathrm{DFT}} - \mathbf{C}^{\mathrm{model}}$")
    ax[0, 0].set_ylabel("Count")
    ax[1, 0].set_xlabel(r"$\mathbf{C}^{\mathrm{DFT}} - \mathbf{C}^{\mathrm{model}}$")
    ax[1, 0].set_ylabel("Count")
    ax[0, 1].set_xlabel(r"DFT $\psi \left(\mathbf{r}\right)$")
    ax[0, 1].set_ylabel(r"Model $\psi \left(\mathbf{r}\right)$")
    ax[1, 1].set_xlabel(r"DFT $\psi \left(\mathbf{r}\right)$")
    ax[1, 1].set_ylabel(r"Model $\psi \left(\mathbf{r}\right)$")

    ax[0, 0].set_title("Minimal basis set")
    ax[0, 1].set_title("Minimal basis set")
    ax[1, 0].set_title("Full basis set")
    ax[1, 1].set_title("Full basis set")

    ax[0, 0].set_yscale("log")
    ax[1, 0].set_yscale("log")

    # Label the plots with characters a, b, c and d on the top left corner
    for idx, label in enumerate(["a", "b", "c", "d"]):
        ax[idx // 2, idx % 2].text(
            0.05,
            0.95,
            f"{label})",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax[idx // 2, idx % 2].transAxes,
            fontsize=10,
            fontweight="bold",
        )

    for idx in range(2):
        ax[idx, 1].set_aspect("equal", "box")
        ax[idx, 1].set_ylim(-0.5, 0.5)
        ax[idx, 1].set_xlim(-0.5, 0.5)
    for idx in range(2):
        ax[idx, 0].set_aspect("auto", "box")
        # ax[idx, 0].set_xlim(-0.3, 0.3)

    fig.savefig(
        __output_folder__
        / f"figure7_{dataset_name}_{basis_set_name}_mo_coeff_diff.png",
        dpi=300,
    )
    fig.savefig(
        __output_folder__
        / f"figure7_{dataset_name}_{basis_set_name}_mo_coeff_diff.pdf",
        dpi=300,
    )
