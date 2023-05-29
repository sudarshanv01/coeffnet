import pytest

import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from e3nn import o3
from e3nn.util.test import assert_equivariant

from minimal_basis.dataset.reaction import ReactionDataset
from minimal_basis.model.reaction import ReactionModel
from minimal_basis.transforms.rotations import RotationMatrix

from conftest import (
    get_dataset_options,
    model_options_factory,
    network_factory,
    get_mapping_idx_to_euler_angles,
)


def test_input_model(model_options_factory):
    """Test input of model options to the model."""

    model_options = model_options_factory(prediction_mode="relative_energy")
    model = ReactionModel(**model_options)
    assert model.irreps_in == "5x0e+3x1o"
    assert model.irreps_hidden == "64x0e+288x1o"
    assert model.irreps_out == "5x0e+3x1o"
    assert model.irreps_node_attr == "1x0e"
    assert model.irreps_edge_attr == "12x0e"
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.num_basis == 12
    assert model.num_neighbors == 4
    assert model.typical_number_of_nodes == 12
    assert model.reduce_output == True
    assert model.reference_reduced_output_to_initial_state == True
    assert model.make_absolute == False
    assert model.mask_extra_basis == True
    assert model.normalize_sumsq == True

    model_options = model_options_factory(prediction_mode="coeff_matrix")
    model = ReactionModel(**model_options)
    assert model.irreps_in == "5x0e+3x1o"
    assert model.irreps_hidden == "64x0e+288x1o"
    assert model.irreps_out == "5x0e+3x1o"
    assert model.irreps_node_attr == "1x0e"
    assert model.irreps_edge_attr == "12x0e"
    assert model.radial_layers == 2
    assert model.radial_neurons == 64
    assert model.max_radius == 5
    assert model.num_basis == 12
    assert model.num_neighbors == 4
    assert model.typical_number_of_nodes == 12
    assert model.reduce_output == False
    assert model.reference_reduced_output_to_initial_state == False
    assert model.make_absolute == False
    assert model.mask_extra_basis == True
    assert model.normalize_sumsq == True


def test_sum_square_one(get_dataset_options, model_options_factory):
    """Test that the output of each network pass is normalized to sum of squares one."""
    dataset_options = get_dataset_options
    model_options = model_options_factory(prediction_mode="coeff_matrix")
    model_options["normalize_sumsq"] = True
    model = ReactionModel(**model_options)

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        output = model(batch)
        sum_sq_output = torch.sum(output**2)
        assert torch.allclose(sum_sq_output, torch.ones_like(sum_sq_output))


def test_masked_basis_are_masked(get_dataset_options, model_options_factory):
    """Test that the output of each network pass is masked correctly as per `basis_mask`."""
    dataset_options = get_dataset_options
    model_options = model_options_factory(prediction_mode="coeff_matrix")
    model_options["normalize_sumsq"] = True
    model = ReactionModel(**model_options)

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        output = model(batch)
        masked_output = output[~batch.basis_mask]
        assert torch.allclose(masked_output, torch.zeros_like(masked_output))


def test_make_absolute(get_dataset_options, model_options_factory):
    """Test that the output of each network pass is made absolute as per `make_absolute`."""
    dataset_options = get_dataset_options
    model_options = model_options_factory(prediction_mode="coeff_matrix")
    model_options["make_absolute"] = True
    model = ReactionModel(**model_options)

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        output = model(batch)
        assert torch.allclose(output, torch.abs(output))


def test_equivariance(get_dataset_options, network_factory):
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
        )

        return model(data)

    dataset_options = get_dataset_options
    model = network_factory(prediction_mode="coeff_matrix")
    irreps_in = model.irreps_in

    irreps_out = model.irreps_out

    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:

        batch = data.batch
        basis_mask = data.basis_mask
        p = data.p
        pos_interpolated_transition_state = data.pos_interpolated_transition_state

        assert_equivariant(
            func=wrapped_model,
            args_in=[data.x, data.x_final_state, data.pos, data.pos_final_state],
            irreps_in=[irreps_in, irreps_in, "cartesian_points", "cartesian_points"],
            irreps_out=irreps_out,
        )
