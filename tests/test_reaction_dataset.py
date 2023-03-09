import numpy as np

import pytest

import torch

from e3nn import o3

from conftest import create_ReactionDataset


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
