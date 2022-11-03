import numpy as np

import torch

from e3nn import o3

from conftest import sn2_reaction_input, get_basis_file_info

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

from minimal_basis.model.model_hamiltonian import EquivariantNodeConv

from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_hamiltonian_dataset_sn2_graph(sn2_reaction_input, tmp_path):
    """Test that the Hamiltonian dataset."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()
    assert dataset.input_data is not None
    assert dataset.basis_info is not None


def test_equivariance_equivariantnodeconv(sn2_reaction_input, tmp_path):
    """Test the equivariance of the EquivariantNodeConv layer."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()

    # Analyse a single datapoint
    datapoint = dataset[0]

    irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")

    expected_ndim = irreps_out.dim

    # Create a convolution input
    conv = EquivariantNodeConv(
        irreps_out_per_basis=irreps_out,
        hidden_layers=10,
        num_basis=10,
    )

    # Perform the convolution
    output = conv(
        f_in=datapoint.x,
        edge_index=datapoint.edge_index,
        pos=datapoint.pos,
        max_radius=10,
        num_nodes=datapoint.num_nodes,
    )

    assert output.shape[0] == datapoint.num_nodes
    assert output.shape[1] == expected_ndim * 3
