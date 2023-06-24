from pathlib import Path

import json

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
from plot_params import get_plot_params

get_plot_params()

import basis_set_exchange as bse

from minimal_basis.predata.matrices import TaskdocsToData

from instance_mongodb import instance_mongodb_sei


def get_data(basis_set_type: str, basis_set: str):
    """Get the data from the MongoDB database."""
    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_sn2_reaction_calculation

    basis_info_raw = bse.get_basis(basis_set, fmt="json", elements=list(range(1, 19)))
    basis_info_raw = json.loads(basis_info_raw)

    kwargs = {"store_extra_tags": "euler_angles"}
    specs = {
        "inverted_coordinates": True,
        "basis_are_spherical": True,
    }
    tags_specs = {f"tags.{k}": v for k, v in specs.items()}
    tags_specs["orig.rem.basis"] = basis_set
    tags_specs["orig.rem.purecart"] = "1111"

    taskdocs_to_data = TaskdocsToData(
        collection=collection,
        filter_collection=tags_specs,
        identifier="idx",
        state_identifier="state",
        reactant_tag="reactant",
        transition_state_tag="transition_state",
        product_tag="product",
        basis_set_type=basis_set_type,
        basis_info_raw=basis_info_raw,
        p_orbital_specs=specs,
        d_orbital_specs=specs,
        f_orbital_specs=specs,
        g_orbital_specs=specs,
        switch_to_spherical=False,
        **kwargs,
    )
    data = taskdocs_to_data.get_data()

    return data


def get_selected_eigenvalues(eigenvalues):
    """Get the selected eigenvalues."""
    selected_eigenval = eigenvalues[eigenvalues < 0]
    selected_eigenval = np.sort(selected_eigenval)
    selected_eigenval = selected_eigenval[-1]
    selected_eigenval_index = np.where(eigenvalues == selected_eigenval)[0][0]
    return selected_eigenval_index


def get_interesting_indices(coeff_matrix):
    """Given a Nx3 matrix, return the row indices (<N) which are interesting.
    Interesting indices are those any of the values in the row are greater than 0.2."""
    interesting_indices = np.where(np.any(np.abs(coeff_matrix) > 0.125, axis=1))[0]
    return interesting_indices


def main():
    """Main function."""
    data_full = get_data(basis_set_type="full", basis_set="def2-svp")
    data_minimal = get_data(basis_set_type="minimal", basis_set="def2-svp")

    full_labels = data_full[0]["orbital_info"][0]
    full_labels = [f"{label[0]}({label[2]})" for label in full_labels]
    minimal_labels = data_minimal[0]["orbital_info"][0]
    minimal_labels = [f"{label[0]}({label[2]})" for label in minimal_labels]

    full_labels = np.array(full_labels)
    minimal_labels = np.array(minimal_labels)

    full_states = data_full[0]["state"]
    minimal_states = data_minimal[0]["state"]

    reactant_index = np.where(full_states == "reactant")[0][0]
    product_index = np.where(full_states == "product")[0][0]
    transition_state_index = np.where(full_states == "transition_state")[0][0]

    full_coeff_matrix = data_full[0]["coeff_matrices"][:, 0, ...]
    minimal_coeff_matrix = data_minimal[0]["coeff_matrices"][:, 0, ...]

    full_eigenvalues = data_full[0]["eigenvalues"][:, 0, ...]
    minimal_eigenvalues = data_full[0]["eigenvalues"][:, 0, ...]

    full_homo_idx = get_selected_eigenvalues(full_eigenvalues[reactant_index, ...])
    minimal_homo_idx = get_selected_eigenvalues(
        minimal_eigenvalues[reactant_index, ...]
    )

    plot_full_coeff_matrix = full_coeff_matrix[
        [reactant_index, transition_state_index, product_index], :, full_homo_idx
    ]
    plot_minimal_coeff_matrix = minimal_coeff_matrix[
        [reactant_index, transition_state_index, product_index], :, minimal_homo_idx
    ]

    return (
        plot_full_coeff_matrix,
        plot_minimal_coeff_matrix,
        full_labels,
        minimal_labels,
    )


if __name__ == "__main__":
    """Create figure 2 of the manuscript."""

    __output_dir__ = Path("output")
    __output_dir__.mkdir(exist_ok=True)

    (
        plot_full_coeff_matrix,
        plot_minimal_coeff_matrix,
        full_labels,
        minimal_labels,
    ) = main()

    plot_full_coeff_matrix = np.abs(plot_full_coeff_matrix.T)
    plot_minimal_coeff_matrix = np.abs(plot_minimal_coeff_matrix.T)

    full_interesting_idx = get_interesting_indices(plot_full_coeff_matrix)
    minimal_interesting_idx = get_interesting_indices(plot_minimal_coeff_matrix)

    fig, ax = plt.subplots(1, 2, figsize=(2, 3), constrained_layout=True)

    cax0 = ax[0].imshow(
        plot_full_coeff_matrix[full_interesting_idx], cmap="Blues", vmin=0, vmax=1
    )
    cax1 = ax[1].imshow(
        plot_minimal_coeff_matrix[minimal_interesting_idx], cmap="Blues", vmin=0, vmax=1
    )

    ax[0].set_yticks(np.arange(len(full_interesting_idx)))
    ax[0].set_yticklabels(full_labels[full_interesting_idx])
    ax[1].set_yticks(np.arange(len(minimal_interesting_idx)))
    ax[1].set_yticklabels(minimal_labels[minimal_interesting_idx])

    # Set `reactant` `transition_state` and `product` labels
    ax[0].set_xticks(np.arange(3))
    ax[0].set_xticklabels(["reactant", "transition state", "product"], rotation=90)
    ax[1].set_xticks(np.arange(3))
    ax[1].set_xticklabels(["reactant", "transition state", "product"], rotation=90)

    ax[0].set_ylabel("Selected basis functions")
    ax[0].set_title("a) Full basis")
    ax[1].set_title("b) Minimal basis")

    cbar1 = fig.colorbar(cax1, ax=ax[1])

    # Switch off all minor ticks
    ax[0].tick_params(axis="both", which="both", length=0)
    ax[1].tick_params(axis="both", which="both", length=0)

    fig.savefig(__output_dir__ / f"figure2.png", dpi=300)
    fig.savefig(__output_dir__ / f"figure2.pdf", dpi=300)
