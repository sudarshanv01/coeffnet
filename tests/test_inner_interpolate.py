from conftest import inner_interpolate_input

from torch_geometric.loader import DataLoader

from minimal_basis.dataset import InnerInterpolateDataset


def test_dataset_process(inner_interpolate_input, tmp_path):
    """Test the charge dataset."""
    filename = inner_interpolate_input
    dataset = InnerInterpolateDataset(
        root=tmp_path,
        filename=filename,
    )
    assert dataset.input_data is not None
    assert dataset.data is not None
