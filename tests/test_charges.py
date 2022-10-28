from conftest import sn2_reaction_input

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import (
    EdgeModel,
    NodeModel,
    GlobalModel,
    Graph2GraphModel,
    Graph2PropertyModel,
)


def test_charge_dataset_sn2_graph(sn2_reaction_input, tmp_path):
    """Test the charge dataset."""
    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
        filename=filename,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()
    assert dataset.input_data is not None
    assert dataset.data is not None


def test_charge_datapoint_sn2_graph(sn2_reaction_input, tmp_path):
    """Check the charge datapoint."""
    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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


def test_edge_update_model_sn2_graph(sn2_reaction_input, tmp_path):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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


def test_node_update_model_sn2_graph(sn2_reaction_input, tmp_path):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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
        x = x.view(-1, datapoint.num_node_features)
        ek = datapoint.edge_attr
        ek = ek.view(-1, datapoint.num_edge_features)
        u = datapoint.global_attr
        u = u.view(-1, 1)

        num_node_features = x.shape[1]
        num_edge_features = ek.shape[1]
        num_global_features = u.shape[1]

        node_model = NodeModel(
            hidden_channels=32,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_targets=10,
        )

        output = node_model(x, datapoint.edge_index, ek, u, datapoint.batch)

        assert output.shape[0] == x.shape[0]
        assert output.shape[1] == 10


def test_global_update_model_sn2_graph(sn2_reaction_input, tmp_path):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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
        x = x.view(-1, datapoint.num_node_features)
        ek = datapoint.edge_attr
        ek = ek.view(-1, datapoint.num_edge_features)
        u = datapoint.global_attr
        u = u.view(-1, 1)

        num_node_features = x.shape[1]
        num_global_features = u.shape[1]

        global_model = GlobalModel(
            hidden_channels=32,
            num_global_features=num_global_features,
            num_node_features=num_node_features,
            num_targets=10,
        )

        output = global_model(x, datapoint.edge_index, ek, u, datapoint.batch)

        assert output.shape[0] == u.shape[0]
        assert output.shape[1] == 10


def test_graph2graph_model_sn2_graph(sn2_reaction_input, tmp_path):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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
        x = x.view(-1, datapoint.num_node_features)
        ek = datapoint.edge_attr
        ek = ek.view(-1, datapoint.num_edge_features)
        u = datapoint.global_attr
        u = u.view(-1, 1)

        num_node_features = x.shape[1]
        num_edge_features = ek.shape[1]
        num_global_features = u.shape[1]

        graph2graph_model = Graph2GraphModel(
            hidden_channels=32,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_updates=3,
        )

        output = graph2graph_model(x, datapoint.edge_index, ek, u, datapoint.batch)

        x_updated, ek_updated, u_updated = output

        # Check the shapes of the updated features. The shape of the updated
        # node features should be the same as the original node features.
        assert x_updated.shape[0] == x.shape[0]
        assert x_updated.shape[1] == x.shape[1]

        # The shape of the updated edge features should be the same as the
        # original edge features.
        assert ek_updated.shape[0] == ek.shape[0]
        assert ek_updated.shape[1] == ek.shape[1]

        # The shape of the updated global features should be the same as the
        # original global features.
        assert u_updated.shape[0] == u.shape[0]
        assert u_updated.shape[1] == u.shape[1]


def test_graph2property_model_sn2_graph(sn2_reaction_input, tmp_path):
    """Check if the edge update of the model is correct."""

    filename = sn2_reaction_input
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = ChargeDataset(
        root=tmp_path,
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

        graph2property_model = Graph2PropertyModel()

        output = graph2property_model(x, datapoint.edge_index, ek, u, datapoint.batch)

        assert output.shape[0] == u.shape[0]
