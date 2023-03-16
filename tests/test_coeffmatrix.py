import pytest

from pathlib import Path

import numpy as np

from monty.serialization import loadfn, dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.graphs import MoleculeGraph

import torch

from e3nn import o3

from conftest import create_CoeffMatrix, rotate_three_dimensions


def test_get_coefficient_matrix_rotated_waters(create_CoeffMatrix):
    """Test return coefficient matrix."""
    for data, coeff_matrix in create_CoeffMatrix():
        true_coeff_matrix = data["coeff_matrices"][0][0]
        assert np.allclose(coeff_matrix.get_coefficient_matrix(), true_coeff_matrix)


def test_get_coefficient_matrix_rotated_sn2(create_CoeffMatrix):
    """Test return coefficient matrix."""
    for data, coeff_matrix in create_CoeffMatrix(dataset_name="rotated_sn2"):
        true_coeff_matrix = data["coeff_matrices"][0][0]
        assert np.allclose(coeff_matrix.get_coefficient_matrix(), true_coeff_matrix)


def test_get_padded_coefficient_matrix(create_CoeffMatrix):
    """Take the coefficient matrix and return the padded coefficient matrix."""
    coeff_data = create_CoeffMatrix
    idx_eigenstate = 0

    for data, coeff_matrix in coeff_data(
        dataset_name="rotated_waters", store_idx_only=idx_eigenstate
    ):
        molecule_graph = coeff_matrix.molecule_graph

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
            assert padded_coeff_matrix.shape == (max_dim, 1)
            assert np.allclose(
                padded_coeff_matrix[atom_irrep.dim :, :],
                np.zeros_like(padded_coeff_matrix[atom_irrep.dim :, :]),
            )


def test_get_minimal_basis_representation(create_CoeffMatrix):
    """Test the minimal basis representation function of ModifiedCoefficientMatrix."""
    coeff_data = create_CoeffMatrix

    for data, coeff_matrix in coeff_data(
        dataset_name="rotated_waters", type_coeff_matrix="modified"
    ):
        true_coeff_matrix = data["coeff_matrices"][0][0]
        true_coeff_matrix = np.array(true_coeff_matrix)
        molecule_graph = coeff_matrix.molecule_graph

        minimal_basis_representation = coeff_matrix.get_minimal_basis_representation()
        assert minimal_basis_representation.shape == (
            len(molecule_graph.molecule),
            4,
            true_coeff_matrix.shape[1],
        )


def test_equivariance_minimal_basis_representation_rotated_waters(create_CoeffMatrix):
    """Test the equivariance of the minimal basis representation."""

    expected_irreps_output = o3.Irreps("1x0e+1x1o")

    for idx, (data, coeff_matrix) in enumerate(
        create_CoeffMatrix(
            dataset_name="rotated_waters",
            type_coeff_matrix="modified",
            set_to_absolute=True,
            store_idx_only=0,
        )
    ):

        alpha, beta, gamma = data["angles"]

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)

        minimal_basis_representation = (
            coeff_matrix.get_minimal_basis_representation_atom(atom_idx=0)
        )

        if idx == 0:
            orig_minimal_basis_representation = minimal_basis_representation
            rotation_matrix_0 = rotation_matrix

        rotation_matrix = rotation_matrix @ rotation_matrix_0.T
        D_matrix = expected_irreps_output.D_from_matrix(rotation_matrix)
        D_matrix = D_matrix.detach().numpy()

        rotated_coeff_matrix = orig_minimal_basis_representation.T @ D_matrix.T
        rotated_coeff_matrix = rotated_coeff_matrix.T
        rotated_coeff_matrix = np.abs(rotated_coeff_matrix)

        assert np.allclose(
            minimal_basis_representation, rotated_coeff_matrix, atol=1e-3
        )


def test_equivariance_minimal_basis_representation_rotated_sn2(create_CoeffMatrix):
    """Test the equivariance of the minimal basis representation."""

    expected_irreps_output = o3.Irreps("1x0e+1x1o")

    for idx, (data, coeff_matrix) in enumerate(
        create_CoeffMatrix(
            dataset_name="rotated_sn2",
            type_coeff_matrix="modified",
            set_to_absolute=True,
            store_idx_only=0,
        )
    ):

        alpha, beta, gamma = data["angles"]

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)

        minimal_basis_representation = (
            coeff_matrix.get_minimal_basis_representation_atom(atom_idx=0)
        )

        if idx == 0:
            orig_minimal_basis_representation = minimal_basis_representation
            rotation_matrix_0 = rotation_matrix

        rotation_matrix = rotation_matrix @ rotation_matrix_0.T
        D_matrix = expected_irreps_output.D_from_matrix(rotation_matrix)
        D_matrix = D_matrix.detach().numpy()

        rotated_coeff_matrix = orig_minimal_basis_representation.T @ D_matrix.T
        rotated_coeff_matrix = rotated_coeff_matrix.T
        rotated_coeff_matrix = np.abs(rotated_coeff_matrix)

        assert np.allclose(
            minimal_basis_representation, rotated_coeff_matrix, atol=1e-3
        )
