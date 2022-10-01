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
    """Check the charge dataPoint."""
    filename = sn2_reaction_input

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()

    for datapoint in dataset.data:
        x = datapoint.x
        # x should be a dictionary composed of square tensors.
        n_react = 0
        n_prod = 0
        for react_index in x:
            if int(react_index) < 0:
                n_react += len(x[react_index])
            elif int(react_index) > 0:
                n_prod += len(x[react_index])
            assert react_index != 0, "Index should not be 0."

        assert (
            n_react == n_prod
        ), "Number of reactant atoms should be equal to number of product atoms."

        # Make sure that the y-data is a scalar
        assert datapoint.y.shape == torch.Size([]), "y should be a scalar."


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
    model = ChargeModel(out_channels=1)

    # Make sure the forward pass works.
    output = model(dataset)

    assert output.shape == torch.Size([dataset.len()]), "Output should be a vector."
