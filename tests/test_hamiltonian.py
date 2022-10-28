from conftest import sn2_reaction_input, get_basis_file_info

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset


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
