import os

import torch

from conftest import sn2_reaction_input, get_basis_file_info
from conftest import get_test_data_path

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel


def test_charge_dataset_sn2_graph(sn2_reaction_input):
    """Test the charge dataset."""

    filename = sn2_reaction_input

    # Create the Hamiltonian dataset.
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

    # Create the Hamiltonian dataset.
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


def test_charge_model_sn2_graph(sn2_reaction_input):
    """Check that the model for the SN2 graph works."""
    filename = sn2_reaction_input

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )
    dataset.process()

    # Instantiate the model.
    model = ChargeModel(out_channels=40)

    # Make sure the forward pass works.
    output = model(dataset)

    assert output.shape == torch.Size([dataset.len()]), "Output should be a vector."
