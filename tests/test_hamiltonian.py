import numpy as np

import pytest

import torch

from e3nn import o3

from conftest import sn2_reaction_input, get_basis_file_info

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

from minimal_basis.model.model_hamiltonian import (
    generate_equi_rep_from_matrix,
    EquivariantConv,
    SimpleHamiltonianModel,
)

from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_hamiltonian_dataset(sn2_reaction_input, tmp_path):
    """Test the HamiltonianDataset class."""

    # Create the dataset
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=sn2_reaction_input,
        basis_file=get_basis_file_info(),
    )

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Loop over the data
    for data in loader:
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_global_features = data.num_global_features

        assert data.x.shape == (num_nodes, 162)
        assert data.edge_attr.shape == (num_edges, 162)
        assert data.edge_index.shape == (2, num_edges)
        assert data.global_attr.shape == (num_global_features,)
        assert data.y.shape == (1,)

        assert isinstance(data.x, torch.Tensor)
        assert isinstance(data.edge_attr, torch.Tensor)
        assert isinstance(data.edge_index, torch.Tensor)
        assert isinstance(data.global_attr, torch.Tensor)
        assert isinstance(data.y, torch.Tensor)


def test_generate_equi_rep_from_matrix():
    """Test the generate_equi_rep_from_matrix function."""

    matrix = torch.randn(4, 2, 9, 9)
    matrix = (matrix + matrix.transpose(2, 3)) / 2
    equivariant_rep = generate_equi_rep_from_matrix(matrix)
    assert equivariant_rep.shape == (4, 2, 45)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_EquivariantConv(sn2_reaction_input, tmp_path):
    """Test the EquivariantConv class."""

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=sn2_reaction_input,
        basis_file=get_basis_file_info(),
    )

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Loop over the data
    for data in loader:

        conv = EquivariantConv(
            irreps_in="1x0e+1x2e+1x4e+1x6e+1x8e",
            irreps_out="5x0e+4x1e+12x2e+10x3e+16x4e",
            hidden_layers=64,
        )
        output = conv(data.x, data.edge_attr, data.edge_index)

        assert output.shape == (data.num_edges, 2, 45)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_SimpleHamiltonianModel(sn2_reaction_input, tmp_path):
    """Test the SimpleHamiltonianModel class."""

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=sn2_reaction_input,
        basis_file=get_basis_file_info(),
    )

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Loop over the data
    for data in loader:

        model = SimpleHamiltonianModel(
            irreps_in="1x0e+1x2e+1x4e+1x6e+1x8e",
            irreps_intermediate="5x0e+4x1e+12x2e+10x3e+16x4e",
            hidden_layers=64,
        )
        output = model(data)

        # Make sure that the output is a scalar
        assert output.shape == (1,)
