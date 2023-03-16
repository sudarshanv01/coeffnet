import numpy as np

import pytest

import torch

from e3nn import o3

from conftest import create_ReactionDataset

from minimal_basis.model.model_reaction import ReactionModel


def test_io_ReactionDataset(create_ReactionDataset):
    """Check if the dataset can be created and saved."""

    max_s_functions = 5
    max_p_functions = 5
    max_d_functions = 3
    shape_basis = 1 * max_s_functions + 3 * max_p_functions + 6 * max_d_functions

    for data in create_ReactionDataset(
        max_s_functions=max_s_functions,
        max_p_functions=max_p_functions,
        max_d_functions=max_d_functions,
    ):

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        assert data.x.shape == (num_nodes, shape_basis)
        assert data.x_final_state.shape == (num_nodes, shape_basis)
        assert data.x_transition_state.shape == (num_nodes, shape_basis)

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
        assert data.p.shape == (1,)


def test_io_ReactionDataset_minimal_basis(create_ReactionDataset):
    """Check if the dataset can be created and saved."""

    for data in create_ReactionDataset(
        use_minimal_basis_node_features=True,
    ):

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        assert data.x.shape == (num_nodes, 4)
        assert data.x_final_state.shape == (num_nodes, 4)
        assert data.x_transition_state.shape == (num_nodes, 4)

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
        assert data.p.shape == (1,)
