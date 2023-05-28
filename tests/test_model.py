import pytest

import numpy as np

import torch

from torch_geometric.loader import DataLoader

from e3nn import o3
from e3nn.util.test import assert_equivariant

from minimal_basis.dataset.reaction import ReactionDataset
from minimal_basis.model.reaction import ReactionModel
from minimal_basis.transforms.rotations import RotationMatrix

from conftest import (
    get_dataset_options,
    model_options_factory,
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


# def test_equivariance(get_dataset_options, model_options_factory, get_mapping_idx_to_euler_angles):
#     """Test that the output of each network pass is equivariant."""
#     dataset_options = get_dataset_options
#     model_options = model_options_factory(prediction_mode="coeff_matrix")
#     model_options['irreps_hidden'] = model_options['irreps_out']
#     model = ReactionModel(**model_options)

#     node_irreps = model.irreps_out
#     node_irreps = o3.Irreps(node_irreps)

#     dataset = ReactionDataset(**dataset_options)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)

#     idx_to_euler_angle = get_mapping_idx_to_euler_angles

#     for _idx, data in enumerate(loader):

#         output = model(data)

#         idx = data.identifier[0]
#         euler_angles = idx_to_euler_angle[idx]
#         rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)()

#         if _idx == 0:
#             rotation_matrix_0 = rotation_matrix
#             output_0 = output
#             continue

#         rotation_matrix = rotation_matrix @ rotation_matrix_0.T

#         D_matrix = node_irreps.D_from_matrix(torch.tensor(rotation_matrix))
#         D_matrix = D_matrix.detach().numpy()

#         for i in range(output.shape[0]):
#             rotated_output = output_0[i, :].detach().numpy() @ D_matrix.T
#             computed_output = output[i, :].detach().numpy()
#             print(rotated_output)
#             print(computed_output)

#             difference = np.allclose(rotated_output, computed_output, atol=1e-2)
#             addition = np.allclose(rotated_output, -computed_output, atol=1e-2)
#             assert difference or addition
