import numpy as np
import plotly.express as px
import plotly.io as pio

from instance_mongodb import instance_mongodb_sei

from pymatgen.core.structure import Molecule

import torch

from monty.serialization import loadfn, dumpfn

from e3nn import o3

from utils import rotate_three_dimensions, subdiagonalize_matrix


def add_bounds_to_water_figure(fig):

    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=4.5,
        x1=6.5,
        y1=4.5,
        line=dict(color="black", width=2),
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=5.5,
        x1=6.5,
        y1=5.5,
        line=dict(color="black", width=2),
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=-0.5,
        x1=-0.5,
        y1=6.5,
        line=dict(color="black", width=2),
    )
    fig.add_shape(
        type="line",
        x0=4.5,
        y0=-0.5,
        x1=4.5,
        y1=6.5,
        line=dict(color="black", width=2),
    )

    fig.update_xaxes(
        ticktext=[
            "O1s",
            "O2s",
            "O2p",
            "O2p",
            "O2p",
            "H1s",
            "H1s",
        ],
        tickvals=np.arange(11),
    )
    fig.update_yaxes(
        ticktext=["O1s", "O2s", "O2p", "O2p", "O2p", "H1s", "H1s"],
        tickvals=np.arange(11),
    )

    return fig


def generate_coefficients_water():
    """Generate quantities for water molecule."""

    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_waters_dataset

    hamiltonians = []
    angles = []
    structures = []
    overlap_matrices = []
    coefficients = []
    eigenvalues = []

    for doc in collection.find({}).limit(10):
        hamiltonians.append(doc["fock_matrices"][0])
        overlap_matrices.append(doc["overlap_matrices"][0])
        structure = Molecule.from_dict(doc["structures"][0])
        structures.append(structure)
        coefficients.append(doc["coeff_matrices"][0])
        angles.append(doc["angles"])
        eigenvalues.append(doc["eigenvalues"][0])

    hamiltonians = np.array(hamiltonians)
    overlap_matrices = np.array(overlap_matrices)
    coefficients = np.array(coefficients)
    angles = np.array(angles)
    eigenvalues = np.array(eigenvalues)
    eigenvalues = eigenvalues[..., np.newaxis, :]

    all_eigenvec = []

    for i in range(coefficients.shape[0]):

        specific_coeff = coefficients[i, 0, ...]
        specific_overlap = overlap_matrices[i, 0, ...]
        specific_hamiltonian = hamiltonians[i, 0, ...]
        specific_eigenvalues = eigenvalues[i, 0, ...]

        eigenval_overlap, eigenvec_overlap = np.linalg.eigh(specific_overlap)
        orthogonalisation_matrix = np.dot(
            eigenvec_overlap,
            np.dot(np.diag(1 / np.sqrt(eigenval_overlap)), eigenvec_overlap.T),
        )

        orthogonalised_hamiltonian = np.dot(
            orthogonalisation_matrix.T,
            np.dot(specific_hamiltonian, orthogonalisation_matrix),
        )
        eigenval_hamiltonian, eigenvec_hamiltonian = np.linalg.eigh(
            orthogonalised_hamiltonian
        )
        sum_of_squares = np.sum(eigenvec_hamiltonian**2, axis=-1)

        orig_eigenvec = np.dot(orthogonalisation_matrix, eigenvec_hamiltonian)

        eigenvec_hamiltonian = np.abs(eigenvec_hamiltonian)
        orig_eigenvec = np.abs(orig_eigenvec)

        all_eigenvec.append(eigenvec_hamiltonian)

    all_eigenvec = np.array(all_eigenvec)

    irreps_hamiltonian = o3.Irreps("1x0e+1x0e+1x1o+1x0e+1x0e")

    D_matrices = []
    for idx, angle in enumerate(angles):
        alpha, beta, gamma = angle

        rotation_matrix = rotate_three_dimensions(alpha, beta, gamma)
        rotation_matrix = torch.tensor(rotation_matrix)
        if idx == 0:
            rotation_matrix_0 = rotation_matrix

        # Reference the rotation matrix to the first one
        rotation_matrix = rotation_matrix @ rotation_matrix_0.T

        D_matrix = irreps_hamiltonian.D_from_matrix(rotation_matrix)
        D_matrices.append(D_matrix)

    D_matrices = torch.stack(D_matrices)
    # Convert to numpy array
    D_matrices = D_matrices.detach().numpy()

    # Check the rotation of the coefficient matrix
    coeff_chosen_idx = 4
    all_data = []
    for i in range(len(angles)):
        init_coeff = coefficients[0, 0, :, coeff_chosen_idx]
        calc_coeff = coefficients[i, 0, :, coeff_chosen_idx]
        coeff_rotated = init_coeff @ D_matrices[i].T
        _coefficients_rotated_diff = coeff_rotated - calc_coeff
        _coefficients_rotated_add = coeff_rotated + calc_coeff
        _max_diff = np.max(np.abs(_coefficients_rotated_diff))
        _max_add = np.max(np.abs(_coefficients_rotated_add))
        coefficients_rotated_diff = (
            _coefficients_rotated_diff
            if _max_diff < _max_add
            else _coefficients_rotated_add
        )
        _all_data = np.stack([calc_coeff, coeff_rotated, coefficients_rotated_diff])
        all_data.append(_all_data)

    all_data = np.array(all_data)

    return all_eigenvec, all_data
