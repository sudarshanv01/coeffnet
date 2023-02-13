import numpy as np

import torch

from e3nn import o3

from conftest import sn2_reaction_input, get_basis_file_info

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

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
