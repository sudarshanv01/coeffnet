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

from analysis_utils import (
    get_instance_grid,
    add_grid_to_fig,
)

wandb_api = wandb.Api()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_molecular_orbital_on_grid(
    data, node_output, grid_points_per_axis=5, buffer_grid=1.25
):
    """From the coefficient matrices, get the molecular orbital info."""

    data = data.cpu()

    predicted_ortho_coeff_matrix_to_grid_quantities = get_instance_grid(
        data,
        node_output,
        grid_points_per_axis=grid_points_per_axis,
        buffer_grid=buffer_grid,
        uses_cartesian_orbitals=False,
        invert_coordinates=True,
        basis_name=args.basis_set,
    )

    molecular_orbital = (
        predicted_ortho_coeff_matrix_to_grid_quantities.get_molecular_orbital()
    )

    return molecular_orbital


def get_coeff_matrix_performance():
    """Get the coefficient matrix performance of the model."""
    df = pd.DataFrame()
    df_mo = pd.DataFrame()
    for loader_name, loader in dataloaders.items():

        for data in loader:

            output = coeff_matrix_model(data)
            output = output.cpu().detach().numpy()

            expected = data.x_transition_state
            expected = expected.cpu().detach().numpy()

            loss_positive = np.max(np.abs(output - expected))
            loss_negative = np.max(np.abs(output + expected))
            if loss_positive > loss_negative:
                output = -output

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

            output_mo = get_molecular_orbital_on_grid(
                data,
                output,
                grid_points_per_axis=args.grid_points_per_axis,
                buffer_grid=args.buffer_grid,
            )
            expected_mo = get_molecular_orbital_on_grid(
                data,
                expected,
                grid_points_per_axis=args.grid_points_per_axis,
                buffer_grid=args.buffer_grid,
            )

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

            data_to_store_mo = {
                "output": output_mo.flatten(),
                "expected": expected_mo.flatten(),
                "loader": loader_name,
            }
            df_mo = pd.concat(
                [df_mo, pd.DataFrame(data_to_store_mo)], ignore_index=True
            )

    return df, df_mo


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
    parser.add_argument(
        "--grid_points_per_axis",
        type=int,
        default=4,
        help="The number of grid points per axis",
    )
    parser.add_argument(
        "--buffer_grid",
        type=float,
        default=1.2,
        help="The number of grid points to buffer the grid by",
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

    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=False,
    )
    logger.info(f"Using model name: {model_name}")

    hue_colors = sns.color_palette("colorblind", n_colors=5)

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
            debug=False,
        )

        if idx_type == 0:
            legend = False
        else:
            legend = True

        coeff_matrix_model = get_best_model(
            prediction_mode="coeff_matrix",
            basis_set=basis_set_name,
            basis_set_type=basis_set_type,
            df=df,
            all_runs=all_runs,
            device=DEVICE,
        )

        df_coeff_matrix, df_mo = get_coeff_matrix_performance()
        train_loader_mask = df_coeff_matrix["loader"] == "train"

        # Write both dataframes to csv
        __output_file__ = (
            __output_folder__
            / f"{dataset_name}_{basis_set_name}_{basis_set_type}_coeff_matrix.csv"
        )
        if args.debug:
            __output_file__ = __output_file__.with_name(
                __output_file__.stem + "_debug" + __output_file__.suffix
            )
        df_coeff_matrix.to_csv(__output_file__, index=False)

        __output_file__ = (
            __output_folder__
            / f"{dataset_name}_{basis_set_name}_{basis_set_type}_mo.csv"
        )
        if args.debug:
            __output_file__ = __output_file__.with_name(
                __output_file__.stem + "_debug" + __output_file__.suffix
            )
        df_mo.to_csv(__output_file__, index=False)
