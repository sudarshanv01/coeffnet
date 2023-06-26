import argparse

from pathlib import Path

import torch

import logging

import numpy as np

from ase import data as ase_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        position_X = self.positions[species == X]
        position_C = self.positions[0]
        position_Y = self.positions[species == Y]

        # Generate a grid between the two halogen atoms
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

    return ortho_coeff_matrix_to_grid_quantities.get_molecular_orbital()


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

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0), constrained_layout=True)

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

    for data in dataloaders["validation"]:

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

        mo_output = get_lin_mo(
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

        mo_expected = get_lin_mo(
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

        # Plot the MOs
        ax.plot(
            mo_output,
            color="tab:red",
            ls="--",
        )
        ax.plot(
            mo_expected,
            color="tab:red",
        )

        ax2 = ax.twinx()
        ax2.plot(
            mo_output**2,
            color="tab:blue",
            ls="--",
        )
        ax2.plot(
            mo_expected**2,
            color="tab:blue",
        )

        break

    ax.set_ylabel(r"Highest Occupied $\psi \left(\mathbf{r}\right)$", fontsize=8)
    ax.set_xticks([])
    ax.set_xticks([0, int(len(mo_output) / 2), len(mo_output) - 1])
    ax.set_xticklabels([_X, "C", _Y])
    ax2.set_ylabel(r"$\left | \psi \left(\mathbf{r}\right) \right |^2$", fontsize=8)
    ax2.yaxis.label.set_color("tab:blue")
    ax2.tick_params(axis="y", colors="tab:blue")
    ax.yaxis.label.set_color("tab:red")
    ax.tick_params(axis="y", colors="tab:red")
    # change color of minor ticks
    ax.tick_params(axis="y", which="minor", color="tab:red")
    ax2.tick_params(axis="y", which="minor", color="tab:blue")

    # Remove minor ticks on x-axis
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # Label the MOs
    ax.plot([], [], "-", color="k", label="Model")
    ax.plot([], [], "--", color="k", label="DFT")
    ax.legend(loc="best", frameon=False)

    fig.savefig(
        __output_folder__ / f"figure7_{dataset_name}_{basis_set_name}.png",
        dpi=300,
    )

    fig.savefig(
        __output_folder__ / f"figure7_{dataset_name}_{basis_set_name}.pdf",
        dpi=300,
    )
