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

from conftest import create_CoeffMatrix


@pytest.mark.dataset("rotated_waters")
def test_get_coefficient_matrix_rotated_waters(create_CoeffMatrix):
    """Test return coefficient matrix."""

    coeff_data = create_CoeffMatrix

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]
        assert np.allclose(coeff_matrix.get_coefficient_matrix(), alpha_coeff_matrix)


@pytest.mark.dataset("rotated_sn2")
def test_get_coefficient_matrix_rotated_sn2(create_CoeffMatrix):
    """Test return coefficient matrix."""

    coeff_data = create_CoeffMatrix

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]
        assert np.allclose(coeff_matrix.get_coefficient_matrix(), alpha_coeff_matrix)


@pytest.mark.dataset("rotated_waters")
@pytest.mark.eigenvalue_number("single")
def test_get_coefficient_matrix_single_eigenvalue_rotated_waters(create_CoeffMatrix):
    """Take the coefficient matrix and return the single eigenvalue."""
    coeff_data = create_CoeffMatrix

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]

        assert np.allclose(
            coeff_matrix.get_coefficient_matrix(),
            alpha_coeff_matrix,
        )


@pytest.mark.dataset("rotated_sn2")
@pytest.mark.eigenvalue_number("single")
def test_get_coefficient_matrix_single_eigenvalue_rotated_sn2(create_CoeffMatrix):
    """Take the coefficient matrix and return the single eigenvalue."""
    coeff_data = create_CoeffMatrix

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]

        assert np.allclose(
            coeff_matrix.get_coefficient_matrix(),
            alpha_coeff_matrix,
        )


@pytest.mark.dataset("rotated_waters")
def test_get_coefficient_matrix_for_atom_rotated_waters(create_CoeffMatrix):
    """Take the coefficient matrix and return the coefficient matrix for a single atom."""
    coeff_data = create_CoeffMatrix
    idx_atom = 0

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]

        atom_coeff_matrix = coeff_matrix.get_coefficient_matrix_for_atom(idx_atom)
        atom_irrep = coeff_matrix.get_irreps_for_atom(idx_atom)

        assert atom_coeff_matrix.shape == (atom_irrep.dim, alpha_coeff_matrix.shape[1])


@pytest.mark.dataset("rotated_sn2")
def test_get_coefficient_matrix_for_atom_rotated_sn2(create_CoeffMatrix):
    """Take the coefficient matrix and return the coefficient matrix for a single atom."""
    coeff_data = create_CoeffMatrix
    idx_atom = 0

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]

        atom_coeff_matrix = coeff_matrix.get_coefficient_matrix_for_atom(idx_atom)
        atom_irrep = coeff_matrix.get_irreps_for_atom(idx_atom)

        assert atom_coeff_matrix.shape == (atom_irrep.dim, alpha_coeff_matrix.shape[1])


@pytest.mark.dataset("rotated_waters")
def test_get_padded_coefficient_matrix(create_CoeffMatrix):
    """Take the coefficient matrix and return the padded coefficient matrix."""
    coeff_data = create_CoeffMatrix
    idx_atom = 0

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]
        molecule_graph = _coeff_data["molecule_graph"]
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


@pytest.mark.dataset("rotated_waters")
@pytest.mark.type_coeff_matrix("modified")
def test_get_minimal_basis_representation(create_CoeffMatrix):
    """Test the minimal basis representation function of ModifiedCoefficientMatrix."""
    coeff_data = create_CoeffMatrix

    for _coeff_data in coeff_data:
        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha_coeff_matrix = _coeff_data["alpha_coeff_matrix"]
        molecule_graph = _coeff_data["molecule_graph"]

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


@pytest.mark.dataset("rotated_waters")
@pytest.mark.type_coeff_matrix("modified")
@pytest.mark.set_absolute(True)
@pytest.mark.eigenvalue_number("single")
def test_equivariance_minimal_basis_representation_rotated_waters(create_CoeffMatrix):
    """Test the equivariance of the minimal basis representation."""
    coeff_data = create_CoeffMatrix
    atom_idx = 0
    expected_irreps_output = o3.Irreps("1x0e+1x1o")

    for idx, _coeff_data in enumerate(coeff_data):

        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha, beta, gamma = _coeff_data["angles"]

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)

        minimal_basis_representation = (
            coeff_matrix.get_minimal_basis_representation_atom(atom_idx)
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


@pytest.mark.dataset("rotated_sn2")
@pytest.mark.type_coeff_matrix("modified")
@pytest.mark.set_absolute(True)
@pytest.mark.eigenvalue_number("single")
def test_equivariance_minimal_basis_representation_rotated_sn2(create_CoeffMatrix):
    """Test the equivariance of the minimal basis representation."""
    coeff_data = create_CoeffMatrix
    atom_idx = 0
    expected_irreps_output = o3.Irreps("1x0e+1x1o")

    for idx, _coeff_data in enumerate(coeff_data):

        coeff_matrix = _coeff_data["coeff_matrix"]
        alpha, beta, gamma = _coeff_data["angles"]

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)

        minimal_basis_representation = (
            coeff_matrix.get_minimal_basis_representation_atom(atom_idx)
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
