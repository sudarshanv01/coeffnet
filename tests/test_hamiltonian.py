import os

import torch

from conftest import sn2_reaction_input, get_basis_file_info, get_benchmark_y_data

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset


def test_hamiltonian_dataset_sn2_graph(sn2_reaction_input):
    """Test that the Hamiltonian dataset."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    data_point = HamiltonianDataset(
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    data_point.load_data()
    assert data_point.input_data is not None
    data_point.parse_basis_data()
    assert data_point.basis_info is not None
    dataset = data_point.get_data()
    assert dataset is not None


def test_activate_energy_yvalue(sn2_reaction_input):
    """Check that the y-value is the activation energy."""
    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    data_point = HamiltonianDataset(
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    data_point.load_data()
    data_point.parse_basis_data()
    dataset = data_point.get_data()

    benchmark_y_data = get_benchmark_y_data()

    for datapoint in dataset:
        y = datapoint.y
        # Convert y to a scalar from torch.tensor
        y = y.item()
        assert y in benchmark_y_data


def test_hamiltonian_datapoint_sn2_graph(sn2_reaction_input):
    """Check the Hamiltonian DataPoint."""
    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    data_point = HamiltonianDataset(
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    data_point.load_data()
    data_point.parse_basis_data()
    dataset = data_point.get_data()

    for datapoint in dataset:
        x = datapoint.x
        # x should be a dictionary composed of square tensors.
        n_react = 0
        n_prod = 0
        for react_index in x:
            assert (
                x[react_index].shape[0] == x[react_index].shape[1]
            ), "x should be a square tensor."
            assert x[react_index].shape[-1] == 2, "Spin dimension should be 2"

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
