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
    shape_basis = 1 * max_s_functions + 3 * max_p_functions

    for data in create_ReactionDataset:

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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_io_ReactionModel(create_ReactionDataset):
    """Test the input and output dimensions of the reaction model."""

    max_s_functions = 5
    max_p_functions = 5

    irreps_in = o3.Irreps(f"{max_s_functions}x0e+{max_p_functions}x1o")
    irreps_node_attr = o3.Irreps("1x0e")
    num_basis = 4
    irreps_edge_attr = o3.Irreps(f"{num_basis}x0e")
    irreps_out = o3.Irreps(f"{max_s_functions}x0e+{max_p_functions}x1o")
    irreps_hidden = o3.Irreps("4x0e+3x1o")
    radial_layers = 4
    radial_neurons = 11
    num_neighbors = 4
    max_radius = 4.0
    typical_number_of_nodes = 15

    reaction_model = ReactionModel(
        irreps_in=irreps_in,
        irreps_hidden=irreps_hidden,
        irreps_out=irreps_out,
        irreps_node_attr=irreps_node_attr,
        irreps_edge_attr=irreps_edge_attr,
        radial_layers=radial_layers,
        max_radius=max_radius,
        num_basis=num_basis,
        radial_neurons=radial_neurons,
        num_neighbors=num_neighbors,
        typical_number_of_nodes=typical_number_of_nodes,
        reduce_output=False,
    )

    for data in create_ReactionDataset:

        output = reaction_model(data)

        assert output.shape == (
            data.num_nodes,
            irreps_out.dim,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_equivariance_ReactionModel(create_ReactionDataset):
    """Test if the ReactionModel is equivariant."""

    max_s_functions = 5
    max_p_functions = 5

    irreps_in = o3.Irreps(f"{max_s_functions}x0e+{max_p_functions}x1o")
    irreps_node_attr = o3.Irreps("1x0e")
    num_basis = 4
    irreps_edge_attr = o3.Irreps(f"{num_basis}x0e")
    irreps_out = o3.Irreps(f"{max_s_functions}x0e+{max_p_functions}x1o")
    irreps_hidden = o3.Irreps("1x0e+1x1o")
    radial_layers = 2
    radial_neurons = 2
    num_neighbors = 4
    max_radius = 2
    typical_number_of_nodes = 15

    reaction_model = ReactionModel(
        irreps_in=irreps_in,
        irreps_hidden=irreps_hidden,
        irreps_out=irreps_out,
        irreps_node_attr=irreps_node_attr,
        irreps_edge_attr=irreps_edge_attr,
        radial_layers=radial_layers,
        max_radius=max_radius,
        num_basis=num_basis,
        radial_neurons=radial_neurons,
        num_neighbors=num_neighbors,
        typical_number_of_nodes=typical_number_of_nodes,
        reduce_output=False,
    )

    for data in create_ReactionDataset:

        output = reaction_model(data)

        rot = o3.rand_matrix()
        D_in = irreps_in.D_from_matrix(rot)
        D_out = irreps_out.D_from_matrix(rot)

        data_rotated = data.clone()
        data_rotated.pos = data.pos @ rot.T
        data_rotated.pos_final_state = data.pos_final_state @ rot.T
        data_rotated.pos_transition_state = data.pos_transition_state @ rot.T
        data_rotated.pos_interpolated_transition_state = (
            data.pos_interpolated_transition_state @ rot.T
        )
        data_rotated.x = data.x @ D_in.T
        data_rotated.x_final_state = data.x_final_state @ D_in.T
        data_rotated.x_transition_state = data.x_transition_state @ D_in.T

        output_rotated = reaction_model(data_rotated)

        print(output @ D_out.T)
        print(output_rotated)

        assert torch.allclose(
            output @ D_out.T,
            output_rotated,
            rtol=1e-2,
        )
