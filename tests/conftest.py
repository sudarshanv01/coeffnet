import os

from typing import List, Dict, Tuple, Union, Any

from pathlib import Path

import random

from collections import defaultdict

from itertools import product

import pytest

import numpy as np

from monty.serialization import loadfn, dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import torch

from torch_geometric.loader import DataLoader

from minimal_basis.data.data_reaction import (
    CoefficientMatrix,
    ModifiedCoefficientMatrix,
)

from minimal_basis.dataset.dataset_reaction import ReactionDataset

from e3nn import o3


@pytest.fixture
def rotated_waters_dataset(tmp_path):
    """Read in the rotated waters dataset and return it."""
    basedir = Path(__file__).parent
    json_dataset = loadfn(basedir / "inputs" / "rotated_waters_dataset.json")
    return json_dataset


@pytest.fixture
def rotated_sn2_dataset(tmp_path):
    """Read in the rotated waters dataset and return it."""
    basedir = Path(__file__).parent
    json_dataset = loadfn(basedir / "inputs" / "rotated_sn2_dataset.json")
    return json_dataset


@pytest.fixture
def basis_set(tmp_path):
    """Read in the basis set and return it."""
    basedir = Path(__file__).parent
    basis_set = loadfn(basedir / "inputs" / "sto-3g.json")
    return basis_set


@pytest.fixture
def create_ReactionDataset(request, tmp_path):
    """Create the ReactionDataset object."""
    basedir = Path(__file__).parent

    def _create_ReactionDataset(**kwargs):

        rotated_sn2_dataset_filename = basedir / "inputs" / "rotated_sn2_dataset.json"
        basis_set_filename = basedir / "inputs" / "sto-3g.json"

        dataset_reaction = ReactionDataset(
            filename=rotated_sn2_dataset_filename,
            basis_filename=basis_set_filename,
            root=tmp_path,
            **kwargs,
        )

        loader_reaction = DataLoader(dataset_reaction, batch_size=1, shuffle=True)

        return loader_reaction

    return _create_ReactionDataset


@pytest.fixture
def create_SampleQchemOutput(request, tmp_path):
    """Create the SampleQchemOutput object."""
    basedir = Path(__file__).parent

    rotated_sn2_dataset_filename = basedir / "inputs" / "sample_qchem_matrices.json"

    return loadfn(rotated_sn2_dataset_filename)


@pytest.fixture
def create_CoeffMatrix(rotated_sn2_dataset, rotated_waters_dataset, basis_set):
    """Create the CoefficientMatrix object."""

    def _create_CoeffMatrix(
        dataset_name: str = None, type_coeff_matrix: str = None, **kwargs
    ):
        """TestFactory to create the CoefficientMatrix object.

        Args:
            dataset_name (list): Name of the dataset to test on.
            eigenvalue_number (int): Number of eigenvalues to store.
            type_coeff_matrix (str): Type of coefficient matrix to create.
            set_absolute (bool): Whether to set the coefficients to absolute values.
        """

        if dataset_name == "rotated_sn2":
            json_dataset = rotated_sn2_dataset
        elif dataset_name == "rotated_waters":
            json_dataset = rotated_waters_dataset
        elif dataset_name is None:
            json_dataset = rotated_waters_dataset
        else:
            raise ValueError("Dataset not found.")

        if type_coeff_matrix == "regular":
            ClassCoeffMatrix = CoefficientMatrix
        elif type_coeff_matrix == "modified":
            ClassCoeffMatrix = ModifiedCoefficientMatrix
        elif type_coeff_matrix is None:
            ClassCoeffMatrix = CoefficientMatrix
        else:
            raise ValueError("Type of coefficient matrix not found.")

        for data in json_dataset:
            molecule = data["structures"][0]
            molecule_graph = MoleculeGraph.with_local_env_strategy(
                molecule, OpenBabelNN()
            )

            alpha_coeff_matrix = data["coeff_matrices"][0][0]
            alpha_coeff_matrix = np.array(alpha_coeff_matrix)

            coeff_matrix = ClassCoeffMatrix(
                molecule_graph=molecule_graph,
                basis_info_raw=basis_set,
                coefficient_matrix=alpha_coeff_matrix,
                **kwargs,
            )

            yield data, coeff_matrix

    yield _create_CoeffMatrix


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


def get_basis_file_info(basis_set: str = "sto-3g"):
    """Get the absolute path of the requested basis file."""
    relative_filepath = os.path.join("inputs", f"{basis_set}.json")
    confpath = os.path.dirname(os.path.abspath(__file__))
    absolute_filepath = os.path.join(confpath, relative_filepath)
    return absolute_filepath
