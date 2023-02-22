import numpy as np

import pytest

import torch

from e3nn import o3

from conftest import sn2_reaction_input, rotation_sn2_input, get_basis_file_info

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

from minimal_basis.model.model_hamiltonian import (
    generate_equi_rep_from_matrix,
    EquivariantConv,
    SimpleHamiltonianModel,
)

from e3nn.util.test import assert_equivariant, assert_auto_jitable


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_irreps_from_maxbasis(sn2_reaction_input, tmp_path):
    """Test the irreps_from_maxbasis function."""
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=sn2_reaction_input,
        basis_file=get_basis_file_info(),
    )

    all_basis, all_irreps = dataset.get_irreps_from_maxbasis("3d")
    correct_all_basis = ["1s", "2s", "2p", "3s", "3p", "3d"]

    assert all_irreps == o3.Irreps("1x0e+1x0e+1x1o+1x0e+1x1o+1x2e")
    assert set(all_basis) == set(correct_all_basis)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_hamiltonian_dataset(sn2_reaction_input, tmp_path):
    """Test the HamiltonianDataset class."""

    # Create the dataset
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=sn2_reaction_input,
        basis_file=get_basis_file_info(),
        max_basis="3d",
    )

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get the irreps for the max basis to ensure that the dimensions are correct
    all_basis, all_irreps = dataset.get_irreps_from_maxbasis("3d")

    # Loop over the data
    for data in loader:

        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_global_features = data.num_global_features["initial_state"]

        irreps_node_features = data.irreps_node_features[0]
        irreps_global_attr = data.irreps_global_attr

        assert data.x.shape == (num_nodes, all_irreps.dim)
        assert data.x.shape == (num_nodes, irreps_node_features.dim)

        assert data.edge_index.shape == (2, num_edges)
        assert data.edge_index_interpolated_TS.shape == (2, num_edges)
        assert data.edge_index_final_state.shape == (2, num_edges)

        assert data.global_attr.shape == (num_global_features**2,)

        assert data.y.shape == (1,)
        assert data.pos.shape == (num_nodes, 3)

        assert isinstance(data.x, torch.Tensor)
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
@pytest.mark.filterwarnings("ignore::FutureWarning")
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
            irreps_in=data.irreps_node_features[0],
            irreps_out=data.irreps_node_features[0],
            hidden_layers=64,
            num_basis=10,
            max_radius=4.0,
        )

        output = conv(
            data.x,
            data.x_final_state,
            data.edge_index_interpolated_TS,
            data.pos_interpolated_TS,
        )

        assert output.shape == (data.num_edges, data.irreps_node_features[0].dim)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
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
            num_basis=10,
            max_radius=4.0,
        )
        output = model(data)

        # Make sure that the output is a scalar
        assert output.shape == (1,)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_rotated_conftest(rotation_sn2_input, tmp_path):
    """Test the rotation conftest."""

    rotation_input, rotation_matrix = rotation_sn2_input

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=rotation_input,
        basis_file=get_basis_file_info(),
    )

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    irreps_in = o3.Irreps("3x0e+2x1o+1x2o+3x2e+1x3o+1x4e")
    irreps_out = o3.Irreps("6x0o+25x0e+38x1o+22x1e+34x2o+50x2e")

    # Loop over the data
    for idx, data in enumerate(loader):

        conv = EquivariantConv(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            hidden_layers=64,
            num_basis=10,
            max_radius=4.0,
        )
        output = conv(data.x, data.edge_attr, data.edge_index, data.pos)

        assert output.shape == (data.num_edges, irreps_out.dim)

        if idx == 0:
            # This output will serve as a reference for the other outputs
            output_ref = output

        D_prime_g = irreps_out.D_from_matrix(torch.tensor(rotation_matrix[idx]))
        D_prime_g = torch.tensor(D_prime_g, dtype=torch.float32)

        output_rotated = torch.zeros_like(output)

        for i in range(output.shape[0]):
            output_rotated[i] = D_prime_g @ output_ref[i]

        # Make sure that `output_rotated` matches `output`
        assert output.shape == output_rotated.shape

        print(output)
        print(output_rotated)
        print(torch.max(torch.abs(output - output_rotated)))

        assert torch.allclose(output, output_rotated, atol=1e-3)
