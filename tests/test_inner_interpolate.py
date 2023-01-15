from conftest import inner_interpolate_input

from torch_geometric.loader import DataLoader

from minimal_basis.dataset import InnerInterpolateDataset
from minimal_basis.model import MessagePassingInnerInterpolateModel


def test_dataset_process(inner_interpolate_input, tmp_path):
    """Test the charge dataset."""
    filename = inner_interpolate_input
    dataset = InnerInterpolateDataset(
        root=tmp_path,
        filename=filename,
    )
    assert dataset.input_data is not None
    assert dataset.data is not None


def test_model(inner_interpolate_input, tmp_path):
    """Test the InnerInterpolate model to see if it runs."""
    # Load the dataset
    filename = inner_interpolate_input
    dataset = InnerInterpolateDataset(
        root=tmp_path,
        filename=filename,
    )

    # Make a DataLoader object
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get the number of node, edge and global features.
    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    num_global_features = dataset.num_global_features

    # Create the model
    model = MessagePassingInnerInterpolateModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels=64,
        num_updates=3,
    )

    # Run the model
    for datapoint in loader:
        output = model(datapoint)

        assert output.shape[0] == 2
