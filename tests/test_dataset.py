import pytest

import numpy as np

import torch

from torch_geometric.loader import DataLoader

from e3nn import o3

from minimal_basis.dataset.reaction import ReactionDataset
from minimal_basis.transforms.rotations import RotationMatrix

from conftest import dataset_options_factory, get_mapping_idx_to_euler_angles


def test_input_dataset(dataset_options_factory):
    """Test input of dataset options to the dataset."""
    dataset_options = dataset_options_factory("minimal")
    dataset = ReactionDataset(**dataset_options)
    assert dataset.max_s_functions == 4
    assert dataset.max_p_functions == 3
    assert dataset.max_d_functions == 0
    assert dataset.idx_eigenvalue == 0
    assert dataset.reactant_tag == "reactant"
    assert dataset.product_tag == "product"
    assert dataset.transition_state_tag == "transition_state"

    dataset_options = dataset_options_factory("full")
    dataset = ReactionDataset(**dataset_options)
    assert dataset.max_s_functions == 4
    assert dataset.max_p_functions == 3
    assert dataset.max_d_functions == 1
    assert dataset.idx_eigenvalue == 0
    assert dataset.reactant_tag == "reactant"
    assert dataset.product_tag == "product"
    assert dataset.transition_state_tag == "transition_state"


def test_download_dataset(dataset_options_factory):
    """Test output of dataset options from the dataset."""
    dataset_options = dataset_options_factory("minimal")
    dataset = ReactionDataset(**dataset_options)
    assert dataset.input_data is not None

    dataset_options = dataset_options_factory("full")
    dataset = ReactionDataset(**dataset_options)
    assert dataset.input_data is not None


def test_output_dataset_minimal_basis(dataset_options_factory):
    """Test the output of the dataset."""
    dataset_options = dataset_options_factory("minimal")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:
        # 12 atoms with 4 s functions and 3 p functions
        num_atoms = 12
        assert data.x.shape == (num_atoms, 13)
        assert data.x_final_state.shape == (num_atoms, 13)
        assert data.x_transition_state.shape == (num_atoms, 13)
        assert data.pos.shape == (num_atoms, 3)
        assert data.pos_final_state.shape == (num_atoms, 3)
        assert data.pos_transition_state.shape == (num_atoms, 3)
        assert data.species.shape == (num_atoms, 1)
        assert data.basis_mask.shape == (num_atoms, 13)


def test_output_dataset_full_basis(dataset_options_factory):
    dataset_options = dataset_options_factory("full")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:
        # 12 atoms with 4 s functions, 3 p functions, and 1 d functions
        num_atoms = 12
        assert data.x.shape == (num_atoms, 18)
        assert data.x_final_state.shape == (num_atoms, 18)
        assert data.x_transition_state.shape == (num_atoms, 18)
        assert data.pos.shape == (num_atoms, 3)
        assert data.pos_final_state.shape == (num_atoms, 3)
        assert data.pos_transition_state.shape == (num_atoms, 3)
        assert data.species.shape == (num_atoms, 1)
        assert data.basis_mask.shape == (num_atoms, 18)


def test_equivariance_dataset_minimal_basis(
    dataset_options_factory, get_mapping_idx_to_euler_angles
):
    """Test the equivariance of each node feature in minimal basis mode."""
    dataset_options = dataset_options_factory("minimal")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    idx_to_euler_angle = get_mapping_idx_to_euler_angles
    node_irreps = o3.Irreps("4x0e+3x1o")
    for _idx, data in enumerate(loader):

        x = data.x
        x_final_state = data.x_final_state
        x_transition_state = data.x_transition_state

        idx = data.identifier[0]
        euler_angles = idx_to_euler_angle[idx]
        rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)()

        if _idx == 0:
            rotation_matrix_0 = rotation_matrix
            x_0 = x
            x_final_state_0 = x_final_state
            x_transition_state_0 = x_transition_state
            continue

        # Reference all rotations to the first rotation
        rotation_matrix = rotation_matrix @ rotation_matrix_0.T

        D_matrix = node_irreps.D_from_matrix(torch.tensor(rotation_matrix))
        D_matrix = D_matrix.detach().numpy()

        for i in range(x.shape[0]):
            rotated_x = x_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x[i, :].detach().numpy()

            # Eigenvectors are only defined up to a sign
            # so we need to check for both the difference and the addition
            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition

        for i in range(x_final_state.shape[0]):
            rotated_x = x_final_state_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x_final_state[i, :].detach().numpy()

            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition

        for i in range(x_transition_state.shape[0]):
            rotated_x = x_transition_state_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x_transition_state[i, :].detach().numpy()

            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition


def test_equivariance_dataset_full_basis(
    dataset_options_factory, get_mapping_idx_to_euler_angles
):
    """Test the equivariance of each node feature in full basis mode."""
    dataset_options = dataset_options_factory("full")
    dataset = ReactionDataset(**dataset_options)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    idx_to_euler_angle = get_mapping_idx_to_euler_angles
    node_irreps = o3.Irreps("4x0e+3x1o+1x2e")
    for _idx, data in enumerate(loader):

        x = data.x
        x_final_state = data.x_final_state
        x_transition_state = data.x_transition_state

        idx = data.identifier[0]
        euler_angles = idx_to_euler_angle[idx]
        rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)()

        if _idx == 0:
            rotation_matrix_0 = rotation_matrix
            x_0 = x
            x_final_state_0 = x_final_state
            x_transition_state_0 = x_transition_state
            continue

        # Reference all rotations to the first rotation
        rotation_matrix = rotation_matrix @ rotation_matrix_0.T

        D_matrix = node_irreps.D_from_matrix(torch.tensor(rotation_matrix))
        D_matrix = D_matrix.detach().numpy()

        for i in range(x.shape[0]):
            rotated_x = x_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x[i, :].detach().numpy()

            # Eigenvectors are only defined up to a sign
            # so we need to check for both the difference and the addition
            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition

        for i in range(x_final_state.shape[0]):
            rotated_x = x_final_state_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x_final_state[i, :].detach().numpy()

            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition

        for i in range(x_transition_state.shape[0]):
            rotated_x = x_transition_state_0[i, :].detach().numpy() @ D_matrix.T
            computed_x = x_transition_state[i, :].detach().numpy()

            difference = np.allclose(rotated_x, computed_x, atol=1e-2)
            addition = np.allclose(rotated_x, -computed_x, atol=1e-2)
            assert difference or addition
