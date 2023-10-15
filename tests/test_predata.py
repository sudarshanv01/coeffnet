import numpy as np
import pytest
from conftest import basis_raw, dataset_config_factory, mock_database

from coeffnet.predata.matrices import TaskdocsToData


def test_TaskdocToData_minimal_basis_input(
    mock_database, dataset_config_factory, basis_raw
):
    """Test the input of the TaskdocToData class."""
    db = mock_database
    dataset_config = dataset_config_factory(basis_type="minimal")

    taskdoc_to_data = TaskdocsToData(
        collection=db.collection,
        basis_info_raw=basis_raw,
        **dataset_config,
    )
    data = taskdoc_to_data.get_data()

    train_frac = 0.8
    test_frac = 0.1
    validate_frac = 0.1

    train_data, test_data, validate_data = taskdoc_to_data.get_random_split_data()

    assert len(train_data) == len(data) * train_frac
    assert len(test_data) == len(data) * test_frac
    assert len(validate_data) == len(data) * validate_frac
    assert len(train_data) + len(test_data) + len(validate_data) == len(data)

    for data in [train_data, test_data, validate_data]:
        for _data in data:
            assert "eigenvalues" in _data
            assert "coeff_matrices" in _data
            assert "orthogonalization_matrices" in _data
            assert "state" in _data
            assert "structures" in _data
            assert "identifiers" in _data
            assert "final_energy" in _data
            assert "indices_to_keep" in _data

            for idx, state in enumerate(_data["state"]):
                coeff_matrix = _data["coeff_matrices"][idx]
                alpha_coeff_matrix = coeff_matrix[0]
                beta_coeff_matrix = coeff_matrix[1]
                assert np.allclose(np.sum(alpha_coeff_matrix**2, axis=0), 1)
                assert np.allclose(np.sum(alpha_coeff_matrix**2, axis=1), 1)
                assert np.allclose(np.sum(beta_coeff_matrix**2, axis=0), 1)
                assert np.allclose(np.sum(beta_coeff_matrix**2, axis=1), 1)

                assert _data["eigenvalues"][idx].shape[1] == coeff_matrix.shape[1]
                assert _data["eigenvalues"][idx].shape[1] == coeff_matrix.shape[2]


def test_TaskdocToData_full_basis_input(
    mock_database, dataset_config_factory, basis_raw
):
    """Test the input of the TaskdocToData class."""
    db = mock_database
    dataset_config = dataset_config_factory(basis_type="full")

    taskdoc_to_data = TaskdocsToData(
        collection=db.collection,
        basis_info_raw=basis_raw,
        **dataset_config,
    )
    data = taskdoc_to_data.get_data()

    train_frac = 0.8
    test_frac = 0.1
    validate_frac = 0.1

    train_data, test_data, validate_data = taskdoc_to_data.get_random_split_data()

    assert len(train_data) == len(data) * train_frac
    assert len(test_data) == len(data) * test_frac
    assert len(validate_data) == len(data) * validate_frac
    assert len(train_data) + len(test_data) + len(validate_data) == len(data)

    for data in [train_data, test_data, validate_data]:
        for _data in data:
            assert "eigenvalues" in _data
            assert "coeff_matrices" in _data
            assert "orthogonalization_matrices" in _data
            assert "state" in _data
            assert "structures" in _data
            assert "identifiers" in _data
            assert "final_energy" in _data
            assert "indices_to_keep" in _data

            for idx, state in enumerate(_data["state"]):
                coeff_matrix = _data["coeff_matrices"][idx]
                alpha_coeff_matrix = coeff_matrix[0]
                beta_coeff_matrix = coeff_matrix[1]
                assert np.allclose(np.sum(alpha_coeff_matrix**2, axis=0), 1)
                assert np.allclose(np.sum(alpha_coeff_matrix**2, axis=1), 1)
                assert np.allclose(np.sum(beta_coeff_matrix**2, axis=0), 1)
                assert np.allclose(np.sum(beta_coeff_matrix**2, axis=1), 1)

                assert _data["eigenvalues"][idx].shape[1] == coeff_matrix.shape[1]
                assert _data["eigenvalues"][idx].shape[1] == coeff_matrix.shape[2]
