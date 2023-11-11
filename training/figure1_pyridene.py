from pathlib import Path
import numpy as np
from ase import units as ase_units
import matplotlib.pyplot as plt

from figure1 import get_data
from instance_mongodb import instance_mongodb_sei
from plot_params import get_plot_params

__output_dir__ = Path("output")
__output_dir__.mkdir(exist_ok=True)
basis_set = "def2-svp"
get_plot_params()

def get_selected_eigenvalues(eigenvalues):
    """Get the selected eigenvalues."""
    selected_eigenval = eigenvalues[eigenvalues < 0]
    selected_eigenval = np.sort(selected_eigenval)
    selected_eigenval = selected_eigenval[-1]
    selected_eigenval_index = np.where(eigenvalues == selected_eigenval)[0][0]
    return list(range(selected_eigenval_index-1, selected_eigenval_index+2))

def get_interesting_indices(coeff_matrix):
    """Given a Nx3 matrix, return the row indices (<N) which are interesting.
    Interesting indices are those any of the values in the row are greater than 0.2."""
    interesting_indices = np.where(np.any(np.abs(coeff_matrix) > 0.125, axis=1))[0]
    return interesting_indices

if __name__ == "__main__":
    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_pyridene_calculations
    data = get_data(basis_set=basis_set, collection=collection)
    #
    eigenvals = data.eigenvalues[0] * ase_units.Hartree
    ortho_coeff_matrix = data.ortho_coeff_matrix[0,...].T
    selected_eigenvals = get_selected_eigenvalues(eigenvals)
    truncated_ortho_coeff_matrix = ortho_coeff_matrix[:,selected_eigenvals] 
    interesting_ind = get_interesting_indices(truncated_ortho_coeff_matrix)
    truncated_ortho_coeff_matrix = truncated_ortho_coeff_matrix[interesting_ind]
    #
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
    plot_kwargs = {"cmap": "coolwarm", "vmin":-1, "vmax":1}
    cax = ax.imshow(truncated_ortho_coeff_matrix, **plot_kwargs)
    basis_function_orbital = data.basis_functions_orbital
    truncated_basis_functions = basis_function_orbital[interesting_ind]
    ax.set_yticks(np.arange(len(truncated_basis_functions)))
    ax.set_yticklabels(truncated_basis_functions)
    ax.set_xticks(np.arange(len(eigenvals[selected_eigenvals])))
    ax.set_xticklabels(
        np.round(eigenvals[selected_eigenvals], 2),
        rotation=90,
    )
    ax.set_xlabel("Eigenvalue [eV]")
    cbar = fig.colorbar(cax, ax=ax)
    fig.savefig(__output_dir__ / "figure1_pyridene.png", dpi=300)