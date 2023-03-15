import numpy as np

from conftest import create_SampleQchemOutput

from minimal_basis.predata.predata_qchem import BaseQuantitiesQChem


def test_predata_qchem(create_SampleQchemOutput):
    """Test the predata_qchem class."""

    eigenvalues = create_SampleQchemOutput["eigenvalues"]
    fock_matrix = create_SampleQchemOutput["fock_matrix"]
    coeff_matrix = create_SampleQchemOutput["coeff_matrix"]

    eigenvalues = np.array(eigenvalues)
    eigenvalues = eigenvalues.flatten()
    fock_matrix = np.array(fock_matrix)
    coeff_matrix = np.array(coeff_matrix)

    instance = BaseQuantitiesQChem(
        fock_matrix=fock_matrix,
        eigenvalues=eigenvalues,
        coeff_matrix=coeff_matrix,
    )
    overlap_matrix = instance.get_overlap_matrix()
    orthogonalization_matrix = instance.get_orthogonalization_matrix()
    ortho_coeff_matrix = instance.get_ortho_coeff_matrix()
    eigenvalues_ortho_fock = instance.get_eigenvalues_ortho_fock()

    assert np.allclose(np.diag(overlap_matrix), np.ones(overlap_matrix.shape[0]))
    assert np.allclose(
        np.sum(ortho_coeff_matrix**2, axis=0), np.ones(ortho_coeff_matrix.shape[1])
    )
    assert np.allclose(
        np.sum(ortho_coeff_matrix**2, axis=0), np.ones(ortho_coeff_matrix.shape[0])
    )

    rotated_coeff_matrix = np.dot(orthogonalization_matrix, ortho_coeff_matrix)

    assert np.allclose(
        np.abs(rotated_coeff_matrix), np.abs(coeff_matrix), atol=1e-5, rtol=1e-5
    )
    assert np.allclose(eigenvalues_ortho_fock, eigenvalues)
