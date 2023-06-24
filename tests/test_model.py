import pytest

import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from e3nn import o3
from e3nn.util.test import assert_equivariant

from coeffnet.dataset.reaction import ReactionDataset
from coeffnet.model.reaction import GateReactionModel
from coeffnet.transforms.rotations import RotationMatrix

from conftest import (
    dataset_options_factory,
    model_options_factory,
    network_factory,
    get_mapping_idx_to_euler_angles,
)


def test_input_model_minimal_basis(network_factory):
    """Test input of model options to the model."""

    model = network_factory(prediction_mode="relative_energy", basis_type="minimal")
    assert model.irreps_in == o3.Irreps("4x0e+3x1o")
    assert model.irreps_hidden == o3.Irreps("100x0e+100x1o")
    assert model.irreps_out == o3.Irreps("4x0e+3x1o")
    assert model.irreps_node_attr == o3.Irreps("5x0e")
    assert model.irreps_edge_attr == o3.Irreps("10x0e")
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.number_of_basis == 10
    assert model.num_neighbors == 4

    model = network_factory(prediction_mode="coeff_matrix", basis_type="minimal")
    assert model.irreps_in == o3.Irreps("4x0e+3x1o")
    assert model.irreps_hidden == o3.Irreps("100x0e+100x1o")
    assert model.irreps_out == o3.Irreps("4x0e+3x1o")
    assert model.irreps_node_attr == o3.Irreps("5x0e")
    assert model.irreps_edge_attr == o3.Irreps("10x0e")
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.number_of_basis == 10
    assert model.num_neighbors == 4


def test_input_model_full_basis(network_factory):
    """Test input of model options to the model."""
    model = network_factory(prediction_mode="relative_energy", basis_type="full")
    assert model.irreps_in == o3.Irreps("4x0e+3x1o+1x2e")
    assert model.irreps_hidden == o3.Irreps("100x0e+100x1o+100x2e")
    assert model.irreps_out == o3.Irreps("4x0e+3x1o+1x2e")
    assert model.irreps_node_attr == o3.Irreps("5x0e")
    assert model.irreps_edge_attr == o3.Irreps("10x0e")
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.number_of_basis == 10
    assert model.num_neighbors == 4

    model = network_factory(prediction_mode="coeff_matrix", basis_type="full")
    assert model.irreps_in == o3.Irreps("4x0e+3x1o+1x2e")
    assert model.irreps_hidden == o3.Irreps("100x0e+100x1o+100x2e")
    assert model.irreps_out == o3.Irreps("4x0e+3x1o+1x2e")
    assert model.irreps_node_attr == o3.Irreps("5x0e")
    assert model.irreps_edge_attr == o3.Irreps("10x0e")
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.number_of_basis == 10
    assert model.num_neighbors == 4


def test_sum_square_one_minimal_basis(dataset_options_factory, model_options_factory):
    """Test that the output of each network pass is normalized to sum of squares one."""
    dataset_options = dataset_options_factory(basis_type="minimal")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_options = model_options_factory(
        prediction_mode="coeff_matrix", basis_type="minimal"
    )
    model_options["normalize_sumsq"] = True
    model = GateReactionModel(**model_options)

    for batch in loader:
        output = model(batch)
        sum_sq_output = torch.sum(output**2)
        assert torch.allclose(sum_sq_output, torch.ones_like(sum_sq_output))


def test_sum_square_one_full_basis(dataset_options_factory, model_options_factory):
    """Test that the output of each network pass is normalized to sum of squares one."""
    dataset_options = dataset_options_factory(basis_type="full")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_options = model_options_factory(
        prediction_mode="coeff_matrix", basis_type="full"
    )
    model_options["normalize_sumsq"] = True
    model = GateReactionModel(**model_options)

    for batch in loader:
        output = model(batch)
        sum_sq_output = torch.sum(output**2)
        assert torch.allclose(sum_sq_output, torch.ones_like(sum_sq_output))


def test_masked_basis_are_masked_minimal_basis(
    dataset_options_factory, model_options_factory
):
    """Test that the output of each network pass is masked correctly as per `basis_mask`."""
    dataset_options = dataset_options_factory(basis_type="minimal")
    model_options = model_options_factory(
        prediction_mode="coeff_matrix", basis_type="minimal"
    )
    model_options["normalize_sumsq"] = True
    model = GateReactionModel(**model_options)

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        output = model(batch)
        masked_output = output[~batch.basis_mask]
        assert torch.allclose(masked_output, torch.zeros_like(masked_output))


def test_masked_basis_are_masked_full_basis(
    dataset_options_factory, model_options_factory
):
    """Test that the output of each network pass is masked correctly as per `basis_mask`."""
    dataset_options = dataset_options_factory(basis_type="full")
    model_options = model_options_factory(
        prediction_mode="coeff_matrix", basis_type="full"
    )
    model_options["normalize_sumsq"] = True
    model = GateReactionModel(**model_options)

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        output = model(batch)
        masked_output = output[~batch.basis_mask]
        assert torch.allclose(masked_output, torch.zeros_like(masked_output))


def test_equivariance_minimal_basis(dataset_options_factory, network_factory):
    """Test that the output of each network pass is equivariant."""

    def wrapped_model(x, x_final_state, pos, pos_final_state):

        data = Data(
            x=x,
            x_final_state=x_final_state,
            pos=pos,
            pos_final_state=pos_final_state,
            batch=batch,
            basis_mask=basis_mask,
            p=p,
            pos_interpolated_transition_state=pos_interpolated_transition_state,
            edge_index=edge_index,
            edge_index_final_state=edge_index_final_state,
            edge_index_interpolated_transition_state=(
                edge_index_interpolated_transition_state
            ),
            node_attr=node_attr,
        )

        return model(data)

    dataset_options = dataset_options_factory(basis_type="minimal")
    model = network_factory(prediction_mode="coeff_matrix", basis_type="minimal")
    irreps_in = model.irreps_in

    irreps_out = model.irreps_out

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:

        batch = data.batch
        basis_mask = data.basis_mask
        p = data.p
        pos_interpolated_transition_state = data.pos_interpolated_transition_state
        edge_index = data.edge_index
        edge_index_final_state = data.edge_index_final_state
        edge_index_interpolated_transition_state = (
            data.edge_index_interpolated_transition_state
        )
        node_attr = data.node_attr

        assert_equivariant(
            func=wrapped_model,
            args_in=[data.x, data.x_final_state, data.pos, data.pos_final_state],
            irreps_in=[irreps_in, irreps_in, "cartesian_points", "cartesian_points"],
            irreps_out=irreps_out,
        )


def test_equivariance_full_basis(dataset_options_factory, network_factory):
    """Test that the output of each network pass is equivariant."""

    def wrapped_model(x, x_final_state, pos, pos_final_state):

        data = Data(
            x=x,
            x_final_state=x_final_state,
            pos=pos,
            pos_final_state=pos_final_state,
            batch=batch,
            basis_mask=basis_mask,
            p=p,
            pos_interpolated_transition_state=pos_interpolated_transition_state,
            edge_index=edge_index,
            edge_index_final_state=edge_index_final_state,
            edge_index_interpolated_transition_state=(
                edge_index_interpolated_transition_state
            ),
            node_attr=node_attr,
        )

        return model(data)

    dataset_options = dataset_options_factory(basis_type="full")
    model = network_factory(prediction_mode="coeff_matrix", basis_type="full")
    irreps_in = model.irreps_in

    irreps_out = model.irreps_out

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:

        batch = data.batch
        basis_mask = data.basis_mask
        p = data.p
        pos_interpolated_transition_state = data.pos_interpolated_transition_state
        edge_index = data.edge_index
        edge_index_final_state = data.edge_index_final_state
        edge_index_interpolated_transition_state = (
            data.edge_index_interpolated_transition_state
        )
        node_attr = data.node_attr

        assert_equivariant(
            func=wrapped_model,
            args_in=[data.x, data.x_final_state, data.pos, data.pos_final_state],
            irreps_in=[irreps_in, irreps_in, "cartesian_points", "cartesian_points"],
            irreps_out=irreps_out,
        )
