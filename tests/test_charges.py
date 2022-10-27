import os

import torch

from conftest import sn2_reaction_input, get_basis_file_info
from conftest import get_test_data_path

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import EdgeModel


def test_charge_dataset_sn2_graph(sn2_reaction_input):
    """Test the charge dataset."""
    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()
    assert dataset.input_data is not None
    assert dataset.data is not None


def test_charge_datapoint_sn2_graph(sn2_reaction_input):
    """Check the charge datapoint."""
    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    # All the datasets are concatenated with each other.
    datapoint = dataset.data

    # Make sure that datapoint contains all the information.
    assert datapoint.num_nodes is not None
    assert datapoint.edge_index is not None
    assert datapoint.y is not None
    assert datapoint.pos is not None


def test_edge_update_model_sn2_graph(sn2_reaction_input):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for datapoint in loader:

        # Infer the number of nodes and edges attributes
        x = datapoint.x
        x = x.view(-1, 1)
        ek = datapoint.edge_attr
        ek = ek.view(-1, 1)
        u = datapoint.global_attr
        u = u.view(-1, 1)

        num_node_features = x.shape[1]
        num_edge_features = ek.shape[1]
        num_global_features = u.shape[1]

        # Perform an update of the edge features.
        edge_model = EdgeModel(
            hidden_channels=32,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_targets=10,
        )

        row, col = datapoint.edge_index
        vrk = x[row]
        vsk = x[col]

        batch = datapoint.batch
        batch = batch[row]

        output = edge_model(ek, vrk, vsk, u, batch)

        assert output.shape[0] == ek.shape[0]
        assert output.shape[1] == 10
