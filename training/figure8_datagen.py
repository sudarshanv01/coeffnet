import argparse

from pathlib import Path

import torch

import logging

import numpy as np

import pandas as pd

from ase import data as ase_data
from ase import units as ase_units

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt

from coeffnet.postprocessing.transformations import (
    OrthoCoeffMatrixToGridQuantities,
    NodeFeaturesToOrthoCoeffMatrix,
    DatapointStoredVectorToOrthogonlizationMatrix,
)

from figure_utils import (
    get_sanitized_basis_set_name,
    get_dataloader_info,
    get_model_data,
    get_best_model,
)

from monty.serialization import loadfn, dumpfn

from model_functions import construct_model_name

from utils import read_inputs_yaml

from plot_params import get_plot_params

get_plot_params()


class OrthoCoeffMatrixTo1DGrid(OrthoCoeffMatrixToGridQuantities):
    def __init__(
        self,
        ortho_coeff_matrix,
        orthogonalization_matrix,
        positions,
        species,
        basis_name,
        indices_to_keep,
        charge=0,
        uses_carterian_orbitals=False,
        buffer_grid=5.0,
        grid_points=100,
        X=None,
        Y=None,
    ):

        self.X = X
        self.Y = Y

        super().__init__(
            ortho_coeff_matrix,
            orthogonalization_matrix,
            positions,
            species,
            basis_name,
            indices_to_keep,
            charge=charge,
            uses_carterian_orbitals=uses_carterian_orbitals,
            buffer_grid=buffer_grid,
            grid_points=grid_points,
        )

    def generate_grid(self):
        """Create a 2D grid in the form of a plane between two the two halogen atoms."""
        species = self.species.reshape(-1)
        species = np.array([int(i) for i in species])
        position_X = self.positions[species == self.X]
        position_C = self.positions[0]
        position_Y = self.positions[species == self.Y]

        if self.X == self.Y:
            position_X = self.positions[species == self.X][0]
            position_Y = self.positions[species == self.Y][1]

        grid1 = np.linspace(position_X, position_C, int(self.grid_points / 2))
        grid2 = np.linspace(position_C, position_Y, int(self.grid_points / 2))
        self.grid = np.concatenate((grid1, grid2), axis=0)
        self.grid = self.grid.reshape(-1, 3)


def get_lin_mo(
    data,
    node_features,
    basis_name: str = "def2-svp",
    charge: int = -1,
    grid_points_per_axis: int = 10,
    buffer_grid: int = 5,
    uses_cartesian_orbitals: bool = True,
    invert_coordinates: bool = False,
    X=None,
    Y=None,
):

    datapoint_to_orthogonalization_matrix = (
        DatapointStoredVectorToOrthogonlizationMatrix(
            data.orthogonalization_matrix_transition_state
        )
    )
    datapoint_to_orthogonalization_matrix()

    orthogonalization_matrix_transition_state = (
        datapoint_to_orthogonalization_matrix.get_orthogonalization_matrix()
    )
    nodefeatures_to_orthocoeffmatrix = NodeFeaturesToOrthoCoeffMatrix(
        node_features=node_features,
        mask=data.basis_mask,
    )
    nodefeatures_to_orthocoeffmatrix()

    positions = data.pos_transition_state.clone()

    if invert_coordinates:
        positions[:, [0, 1, 2]] = positions[:, [2, 0, 1]]

    ortho_coeff_matrix = nodefeatures_to_orthocoeffmatrix.get_ortho_coeff_matrix()
    ortho_coeff_matrix_to_grid_quantities = OrthoCoeffMatrixTo1DGrid(
        ortho_coeff_matrix=ortho_coeff_matrix,
        orthogonalization_matrix=orthogonalization_matrix_transition_state,
        positions=positions,
        species=data.species,
        basis_name=basis_name,
        indices_to_keep=data.indices_to_keep,
        charge=charge,
        uses_carterian_orbitals=uses_cartesian_orbitals,
        buffer_grid=buffer_grid,
        grid_points=grid_points_per_axis,
        X=X,
        Y=Y,
    )
    ortho_coeff_matrix_to_grid_quantities()

    mo = ortho_coeff_matrix_to_grid_quantities.get_molecular_orbital()
    grid = ortho_coeff_matrix_to_grid_quantities.grid

    # Convert the grid, which is a 2D array, to a 1D array of just the euclidean distance
    # Take the relative distance between the first and last point
    grid = grid - grid[0]
    grid = np.linalg.norm(grid, axis=1)
    # Reference the grid to the middle of the grid
    grid = grid - grid[int(len(grid) / 2)]

    return grid, mo


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


def identify_species(identifier):
    """Based on the identifier, return the species name."""
    R = {"A": "H", "B": r"NO$_2$", "C": "CN", "D": r"CH$_3$", "E": r"NH$_2$"}
    X = {"A": "F", "B": "Cl", "C": "Br"}
    Y = {"A": "H", "B": "F", "C": "Cl", "D": "Br"}

    _R1, _R2, _R3, _R4, _X, _Y = identifier.split("_")
    _all_identifiers = []
    for _R in [_R1, _R2, _R3, _R4]:
        _all_identifiers.append(R[_R])
    _all_identifiers.append(X[_X])
    _all_identifiers.append(Y[_Y])

    return _all_identifiers


if __name__ == "__main__":
    """Plot figure7 of the manuscript. This figure will show the molecular orbitals of any
    molecule visualized in the full and minimal basis set."""

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

    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=debug_model,
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

    df, all_runs = get_model_data(
        dataset_name=dataset_name,
        basis_set_type=basis_set_type,
        basis_set=basis_set_name,
        debug=debug_model,
    )

    coeff_matrix_model = get_best_model(
        prediction_mode="coeff_matrix",
        basis_set=basis_set_name,
        basis_set_type=basis_set_type,
        df=df,
        all_runs=all_runs,
        device=DEVICE,
    )

    df_results = pd.DataFrame()
    mo_data = []

    dataloaders.pop("train")
    dataloaders.pop("validation")

    for loader, dataloader in dataloaders.items():

        for idx, data in enumerate(dataloader):

            identifier = data["identifier"][0]
            species_tag = identify_species(identifier)

            _X = species_tag[-2]
            _Y = species_tag[-1]

            if _X == "H" or _Y == "H":
                continue

            X = ase_data.atomic_numbers[_X]
            Y = ase_data.atomic_numbers[_Y]

            output = coeff_matrix_model(data)
            output = output.cpu().detach().numpy()

            expected = data.x_transition_state
            expected = expected.cpu().detach().numpy()

            loss_positive = np.max(np.abs(output - expected))
            loss_negative = np.max(np.abs(output + expected))
            if loss_positive > loss_negative:
                output = -output

            data = data.cpu()

            grid_output, mo_output = get_lin_mo(
                data=data,
                node_features=output,
                basis_name=basis_set_name,
                charge=-1,
                grid_points_per_axis=args.grid_points_per_axis,
                buffer_grid=args.buffer_grid,
                invert_coordinates=True,
                X=X,
                Y=Y,
            )

            grid_expected, mo_expected = get_lin_mo(
                data=data,
                node_features=expected,
                basis_name=basis_set_name,
                charge=-1,
                grid_points_per_axis=args.grid_points_per_axis,
                buffer_grid=args.buffer_grid,
                invert_coordinates=True,
                X=X,
                Y=Y,
            )

            barrier = data.total_energy_transition_state - data.total_energy
            barrier *= ase_units.Hartree
            integral_output = np.trapz(mo_output**2, grid_output)
            data_to_concat = {
                "identifier": identifier,
                "loader": loader,
                "integral": integral_output,
                "barrier": barrier,
            }

            df_results = pd.concat(
                [df_results, pd.DataFrame(data_to_concat, index=[0])]
            )

            if loader == "test" and idx < 3:
                mo_data_to_concat = {
                    "identifier": identifier,
                    "species_tag": species_tag,
                    "grid_output": grid_output.tolist(),
                    "mo_output": mo_output.tolist(),
                    "grid_expected": grid_expected.tolist(),
                    "mo_expected": mo_expected.tolist(),
                }

                mo_data.append(mo_data_to_concat)

    df_results.to_csv(
        __output_folder__ / f"figure8_{dataset_name}_{basis_set_name}.csv"
    )
    dumpfn(mo_data, __output_folder__ / f"figure8_{dataset_name}_{basis_set_name}.json")
