import os

from conftest import sn2_reaction_input, get_basis_file_info

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset


def test_hamiltonian_dataset_sn2_graph(sn2_reaction_input):
    """Test that the Hamiltonian dataset works."""

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
    datapoint = data_point.get_data()

    assert datapoint is not None
    for data in datapoint:
        assert data["x"] is not None
        assert data["y"] is not None
