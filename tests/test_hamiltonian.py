import numpy as np

import torch

from e3nn import o3

from conftest import sn2_reaction_input, get_basis_file_info

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

from minimal_basis.model.model_hamiltonian import (
    EquivariantConv,
    EdgeEquiModel,
    NodeEquiModel,
    EquiGraph2GraphModel,
    EquiHamiltonianModel,
)

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


def test_node_equivariantconv(sn2_reaction_input, tmp_path):
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
    conv = EquivariantConv(
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
        target_dim=datapoint.num_nodes,
    )

    assert output.shape[0] == datapoint.num_nodes
    assert output.shape[1] == expected_ndim * 3


def test_edge_equivariantconv(sn2_reaction_input, tmp_path):
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
    conv = EquivariantConv(
        irreps_out_per_basis=irreps_out,
        hidden_layers=10,
        num_basis=10,
    )

    # Perform the convolution
    output = conv(
        f_in=datapoint.edge_attr,
        edge_index=datapoint.edge_index,
        pos=datapoint.pos,
        max_radius=10,
        num_nodes=datapoint.num_nodes,
        target_dim=datapoint.edge_index.shape[1],
    )

    assert output.shape[0] == datapoint.edge_index.shape[1]
    assert output.shape[1] == expected_ndim * 3


def test_edge_equi_mode(sn2_reaction_input, tmp_path):
    """Test the `EdgeEquiModel` to make sure inputs and output dimensions are correct."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()
    GRAPH_GENERTION_METHOD = "sn2"

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
        basis_file=basis_file,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for datapoint in loader:

        # Create the model
        model = EdgeEquiModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=10,
            num_basis=10,
        )

        output = model(
            edge_attr=datapoint.edge_attr,
            edge_index=datapoint.edge_index,
            pos=datapoint.pos,
            max_radius=10,
            num_nodes=datapoint.num_nodes,
        )

        # Check the output dimensions
        assert output.shape[0] == datapoint.edge_attr.shape[0]
        assert output.shape[1] == irreps_out.dim * 3


def test_node_equi_mode(sn2_reaction_input, tmp_path):
    """Test the `NodeEquiModel` to make sure inputs and output dimensions are correct."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()
    GRAPH_GENERTION_METHOD = "sn2"

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
        basis_file=basis_file,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for datapoint in loader:

        # Create the model
        model = NodeEquiModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=10,
            num_basis=10,
        )

        output = model(
            x=datapoint.x,
            edge_index=datapoint.edge_index,
            pos=datapoint.pos,
            max_radius=10,
            num_nodes=datapoint.num_nodes,
        )

        # Check the output dimensions
        assert output.shape[0] == datapoint.num_nodes
        assert output.shape[1] == irreps_out.dim * 3


def test_equi_graph2graph_model(sn2_reaction_input, tmp_path):
    """Test the `EquiGraphToGraphModel` to make sure inputs and output dimensions are correct."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()
    GRAPH_GENERTION_METHOD = "sn2"

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
        basis_file=basis_file,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for datapoint in loader:

        num_node_features = datapoint.num_node_features
        num_edge_features = datapoint.num_edge_features

        g2g_model = EquiGraph2GraphModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=10,
            num_basis=10,
            num_global_features=1,
            num_targets=10,
            num_updates=4,
            hidden_channels=10,
        )

        u = datapoint.global_attr
        u = u.view(-1, 1)

        x, edge_attr, u = g2g_model(
            x_=datapoint.x,
            edge_index=datapoint.edge_index,
            edge_attr_=datapoint.edge_attr,
            u_=u,
            batch_=datapoint.batch,
            pos=datapoint.pos,
            max_radius=10,
            num_nodes=datapoint.num_nodes,
        )

        # Check that the dimensions are appropriate
        assert x.shape[0] == datapoint.num_nodes
        assert x.shape[1] == irreps_out.dim * 3
        assert edge_attr.shape[0] == datapoint.num_edges
        assert edge_attr.shape[1] == irreps_out.dim * 3
        assert u.shape[0] == datapoint.global_attr.shape[0]
        assert u.shape[1] == 10  # Same as the number of hidden channels


def test_equivariant_hamiltonian_model(sn2_reaction_input, tmp_path):
    """Test the `EquivariantHamiltonianModel` to make sure inputs and output dimensions are correct."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()
    GRAPH_GENERTION_METHOD = "sn2"

    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
        basis_file=basis_file,
    )

    # Initiate the process of creating the dataset.
    dataset.process()

    irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")
    max_radius = 8.0

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for datapoint in loader:

        model = EquiHamiltonianModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=7,
            num_basis=10,
            num_global_features=1,
            num_targets=5,
            num_updates=4,
            hidden_channels=20,
            max_radius=max_radius,
        )

        u = datapoint.global_attr
        u = u.view(-1, 1)

        output = model(datapoint)

        assert output.shape[0] == u.shape[0]
