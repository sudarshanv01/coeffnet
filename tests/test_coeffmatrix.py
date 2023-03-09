import pytest

from pathlib import Path

import numpy as np

from monty.serialization import loadfn, dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.graphs import MoleculeGraph

import torch

from e3nn import o3

from minimal_basis.data.data_reaction import (
    CoefficientMatrix,
    ModifiedCoefficientMatrix,
)


@pytest.fixture()
def rotated_waters_dataset(tmp_path):
    """Read in the rotated waters dataset and return it."""
    basedir = Path(__file__).parent
    json_dataset = loadfn(basedir / "inputs" / "rotated_waters_dataset.json")
    return json_dataset


@pytest.fixture()
def basis_set(tmp_path):
    """Read in the basis set and return it."""
    basedir = Path(__file__).parent
    basis_set = loadfn(basedir / "inputs" / "sto-3g.json")
    return basis_set


def test_get_coefficient_matrix(rotated_waters_dataset, basis_set, tmp_path):
    """Test return coefficient matrix."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]

        coeff_matrix = CoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
        )
        assert np.allclose(coeff_matrix.get_coefficient_matrix(), alpha_coeff_matrix)


def test_get_coefficient_matrix_single_eigenvalue(
    rotated_waters_dataset, basis_set, tmp_path
):
    """Take the coefficient matrix and return the single eigenvalue."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)
        idx_eigenvalue = 0
        single_eigenval_alpha_coeff_matrix = alpha_coeff_matrix[:, idx_eigenvalue]
        single_eigenval_alpha_coeff_matrix = single_eigenval_alpha_coeff_matrix[
            :, np.newaxis
        ]

        coeff_matrix = CoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
            store_idx_only=idx_eigenvalue,
        )
        assert np.allclose(
            coeff_matrix.get_coefficient_matrix(), single_eigenval_alpha_coeff_matrix
        )


def test_get_coefficient_matrix_for_atom(rotated_waters_dataset, basis_set, tmp_path):
    """Take the coefficient matrix and return the coefficient matrix for a single atom."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)

        coeff_matrix = CoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
        )

        idx_atom = 0
        atom_coeff_matrix = coeff_matrix.get_coefficient_matrix_for_atom(idx_atom)
        atom_irrep = coeff_matrix.get_irreps_for_atom(idx_atom)

        assert atom_coeff_matrix.shape == (atom_irrep.dim, alpha_coeff_matrix.shape[1])


def test_get_coefficient_matrix_for_atom_single_eigenvalue(
    rotated_waters_dataset, basis_set, tmp_path
):
    """Take the coefficient matrix and return the coefficient matrix for a single atom."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)
        idx_eigenvalue = 0
        single_eigenval_alpha_coeff_matrix = alpha_coeff_matrix[:, idx_eigenvalue]
        single_eigenval_alpha_coeff_matrix = single_eigenval_alpha_coeff_matrix[
            :, np.newaxis
        ]

        coeff_matrix = CoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
            store_idx_only=idx_eigenvalue,
        )

        idx_atom = 0
        atom_coeff_matrix = coeff_matrix.get_coefficient_matrix_for_atom(idx_atom)
        atom_irrep = coeff_matrix.get_irreps_for_atom(idx_atom)

        assert atom_coeff_matrix.shape == (atom_irrep.dim, 1)


def test_get_padded_coefficient_matrix(rotated_waters_dataset, basis_set, tmp_path):
    """Take the coefficient matrix and return the padded coefficient matrix."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)

        coeff_matrix = CoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
        )

        max_dim = 0
        for atom_idx, atom in enumerate(molecule_graph.molecule):
            irrep = coeff_matrix.get_irreps_for_atom(atom_idx)
            if irrep.dim > max_dim:
                max_dim = irrep.dim

        for atom_idx, atom in enumerate(molecule_graph.molecule):
            atom_irrep = coeff_matrix.get_irreps_for_atom(atom_idx)
            padded_coeff_matrix = coeff_matrix.get_padded_coefficient_matrix_for_atom(
                atom_idx=atom_idx
            )
            assert padded_coeff_matrix.shape == (max_dim, alpha_coeff_matrix.shape[1])
            assert np.allclose(
                padded_coeff_matrix[atom_irrep.dim :, :],
                np.zeros_like(padded_coeff_matrix[atom_irrep.dim :, :]),
            )


def test_get_minimal_basis_representation(rotated_waters_dataset, basis_set, tmp_path):
    """Test the minimal basis representation function of ModifiedCoefficientMatrix."""
    json_dataset = rotated_waters_dataset
    for data in json_dataset:
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)

        coeff_matrix = ModifiedCoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
        )

        minimal_basis_representation = coeff_matrix.get_minimal_basis_representation()
        assert minimal_basis_representation.shape == (
            len(molecule_graph.molecule),
            4,
            alpha_coeff_matrix.shape[1],
        )


def rotate_three_dimensions(alpha, beta, gamma):
    """Rotate the molecule by arbitrary angles alpha
    beta and gamma."""
    cos = np.cos
    sin = np.sin

    r_matrix = [
        [
            cos(alpha) * cos(beta),
            cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
            cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
        ],
        [
            sin(alpha) * cos(beta),
            sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
            sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
        ],
        [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)],
    ]

    r_matrix = np.array(r_matrix)

    return r_matrix


def test_equivariance_minimal_basis_representation(
    rotated_waters_dataset, basis_set, tmp_path
):
    """Test the equivariance of the minimal basis representation."""
    json_dataset = rotated_waters_dataset
    atom_idx = 0

    for idx, data in enumerate(json_dataset):
        molecule = data["structures"][0]
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        alpha_coeff_matrix = data["coeff_matrices"][0][0]
        alpha_coeff_matrix = np.array(alpha_coeff_matrix)

        alpha, beta, gamma = data["angles"]

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)

        coeff_matrix = ModifiedCoefficientMatrix(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_set,
            coefficient_matrix=alpha_coeff_matrix,
            store_idx_only=0,
            set_to_absolute=True,
        )
        minimal_basis_representation = (
            coeff_matrix.get_minimal_basis_representation_atom(atom_idx)
        )

        if idx == 0:
            orig_minimal_basis_representation = minimal_basis_representation
            rotation_matrix_0 = rotation_matrix

        rotation_matrix = rotation_matrix @ rotation_matrix_0.T
        irreps_output = o3.Irreps("1x0e+1x1o")
        D_matrix = irreps_output.D_from_matrix(rotation_matrix)
        D_matrix = D_matrix.detach().numpy()

        rotated_coeff_matrix = orig_minimal_basis_representation.T @ D_matrix.T
        rotated_coeff_matrix = rotated_coeff_matrix.T
        rotated_coeff_matrix = np.abs(rotated_coeff_matrix)

        assert np.allclose(
            minimal_basis_representation, rotated_coeff_matrix, atol=1e-3
        )
