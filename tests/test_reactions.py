import numpy as np

import pytest

import torch

from e3nn import o3

from conftest import create_ReactionDataset

from minimal_basis.model.model_reaction import EquivariantConv
from minimal_basis.model.model_reaction import ReactionModel


def test_io_ReactionDataset(create_ReactionDataset):
    """Check if the dataset can be created and saved."""

    for data in create_ReactionDataset:

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        assert data.x.shape == (
            num_nodes,
            4,
        )  # 1s + 3p
        assert data.x_final_state.shape == (
            num_nodes,
            4,
        )  # 1s + 3p
        assert data.x_transition_state.shape == (
            num_nodes,
            4,
        )  # 1s + 3p

        assert data.pos.shape == (
            num_nodes,
            3,
        )
        assert data.pos_final_state.shape == (
            num_nodes,
            3,
        )
        assert data.pos_transition_state.shape == (
            num_nodes,
            3,
        )
        assert data.pos_interpolated_transition_state.shape == (
            num_nodes,
            3,
        )

        assert data.edge_index.shape == (
            2,
            num_edges,
        )
        assert data.edge_index_final_state.shape == (
            2,
            num_edges,
        )
        assert data.edge_index_transition_state.shape == (
            2,
            num_edges,
        )
        assert data.edge_index_interpolated_transition_state.shape == (
            2,
            num_edges,
        )

        assert data.total_energy.shape == (1,)
        assert data.total_energy_final_state.shape == (1,)
        assert data.total_energy_transition_state.shape == (1,)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_io_EquivariantConv():
    """Check if the EquivariantConv can be created and saved."""

    irreps_sh = o3.Irreps("1x0e+1x1e+1x2e")
    num_basis = 10
    max_radius = 5.0
    hidden_layers = 10
    irreps_in = o3.Irreps("1x0e+1x1e")
    irreps_out = o3.Irreps("1x0e+1x1e")

    equivariant_conv = EquivariantConv(
        irreps_sh,
        num_basis,
        max_radius,
        hidden_layers,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )

    f_1 = torch.randn(10, 4)
    edge_index = torch.randint(0, 10, (2, 20))
    pos = torch.randn(10, 3)

    f_output = equivariant_conv(f_1, edge_index, pos)

    assert f_output.shape == (10, 4)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_equivariance_EquivariantConv(create_ReactionDataset):
    """Check if the EquivariantConv is equivariant."""

    irreps_sh = o3.Irreps("1x0e+1x1e+1x2e")
    num_basis = 10
    max_radius = 5.0
    hidden_layers = 10
    irreps_in = o3.Irreps("1x0e+1x1e")
    irreps_out = o3.Irreps("1x0e+1x1e")

    equivariant_conv = EquivariantConv(
        irreps_sh,
        num_basis,
        max_radius,
        hidden_layers,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )

    for data in create_ReactionDataset:

        f_1 = data.x
        edge_index = data.edge_index
        pos = data.pos

        rot = o3.rand_matrix()
        D_in = irreps_in.D_from_matrix(rot)
        D_out = irreps_out.D_from_matrix(rot)

        f_before = equivariant_conv(f_1 @ D_in.T, edge_index, pos @ rot.T)

        f_output = equivariant_conv(f_1, edge_index, pos)
        f_after = f_output @ D_out.T

        assert torch.allclose(f_before, f_after, atol=1e-5)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_io_ReactionModel(create_ReactionDataset):
    """Check if the ReactionModel can be created and saved."""

    irreps_sh = o3.Irreps("1x0e+1x1e+1x2e")
    num_basis = 10
    max_radius = 5.0
    hidden_layers = 10
    irreps_in = o3.Irreps("1x0e+1x1e")
    irreps_out = o3.Irreps("1x0e+1x1e")

    reaction_model = ReactionModel(
        irreps_sh,
        num_basis,
        max_radius,
        hidden_layers,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )

    for data in create_ReactionDataset:

        f_output = reaction_model(data)

        assert f_output.shape == data.x.shape


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_equivariance_ReactionModel(create_ReactionDataset):
    """Check if the ReactionModel is equivariant."""

    irreps_sh = o3.Irreps("1x0e+1x1e+1x2e")
    num_basis = 10
    max_radius = 5.0
    hidden_layers = 10
    irreps_in = o3.Irreps("1x0e+1x1e")
    irreps_out = o3.Irreps("1x0e+1x1e")

    reaction_model = ReactionModel(
        irreps_sh,
        num_basis,
        max_radius,
        hidden_layers,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )

    for data in create_ReactionDataset:

        rot = o3.rand_matrix()
        D_in = irreps_in.D_from_matrix(rot)
        D_out = irreps_out.D_from_matrix(rot)

        data_rotated = data.clone()
        data_rotated.pos = data.pos @ rot.T
        data_rotated.pos_transition_state = data.pos_transition_state @ rot.T
        data_rotated.pos_final_state = data.pos_final_state @ rot.T
        data_rotated.x = data.x @ D_in.T
        data_rotated.x_transition_state = data.x_transition_state @ D_in.T
        data_rotated.x_final_state = data.x_final_state @ D_in.T

        f_before = reaction_model(data_rotated)

        f_output = reaction_model(data)
        f_after = f_output @ D_out.T

        assert torch.allclose(f_before, f_after, atol=1e-5)
