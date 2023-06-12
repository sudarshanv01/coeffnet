import os

from pathlib import Path

import json

from ase.data import atomic_numbers
from ase import units as ase_units

from collections import defaultdict

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import basis_set_exchange as bse

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300

import torch

from e3nn import o3

from minimal_basis.predata.matrices import BaseMatrices, ReducedBasisMatrices
from minimal_basis.transforms.rotations import RotationMatrix

from instance_mongodb import instance_mongodb_sei
from plot_params import get_plot_params

get_plot_params()


def get_data(basis_sets: list = None):
    """Get all the data from the MongoDB database."""
    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_water_calculations
    if basis_sets is None:
        basis_sets = collection.distinct("orig.rem.basis")
    data = defaultdict(lambda: defaultdict(list))

    for _basis_set in basis_sets:
        print(f"Processing {_basis_set}...")
        basis_info = bse.get_basis(_basis_set, fmt="json", elements=[1, 8])
        basis_info = json.loads(basis_info)

        electron_shells = {
            k: basis_info["elements"][k]["electron_shells"]
            for k in basis_info["elements"].keys()
        }
        angular_momenta = {
            k: [shell["angular_momentum"] for shell in electron_shells[k]]
            for k in electron_shells.keys()
        }
        angular_momenta = {
            k: [item for sublist in angular_momenta[k] for item in sublist]
            for k in angular_momenta.keys()
        }

        find_tags = {
            "orig.rem.basis": _basis_set,
            "tags.inverted_coordinates": True,
            "tags.basis_are_spherical": True,
            "orig.rem.purecart": "1111",
        }
        for doc in collection.find(find_tags).sort("tags.idx", 1):
            _alpha_coeff_matrix = doc["calcs_reversed"][0]["alpha_coeff_matrix"]
            _alpha_eigenvalues = doc["calcs_reversed"][0]["alpha_eigenvalues"]
            _alpha_fock_matrix = doc["calcs_reversed"][0]["alpha_fock_matrix"]

            _alpha_coeff_matrix = np.array(_alpha_coeff_matrix)
            _alpha_eigenvalues = np.array(_alpha_eigenvalues)
            _alpha_fock_matrix = np.array(_alpha_fock_matrix)

            molecule = Molecule.from_dict(doc["orig"]["molecule"])
            symbols = [site.specie.symbol for site in molecule.sites]
            atom_numbers = [atomic_numbers[symbol] for symbol in symbols]

            irreps = ""
            indices_to_keep = []
            indices_s_orbitals = []
            indices_p_orbitals = []
            indices_d_orbitals = []
            indices_f_orbitals = []
            basis_functions_orbital = []

            idx = 0

            for _idx, atom_number in enumerate(atom_numbers):
                _angular_momenta = angular_momenta[str(atom_number)]
                _angular_momenta = np.array(_angular_momenta)
                _basis_functions = 2 * _angular_momenta + 1
                symbol = symbols[_idx]
                for _basis_function in _basis_functions:
                    if _basis_function == 1:
                        basis_functions_orbital.extend([f"{symbol}s"])
                        irreps += "+1x0e"
                        indices_to_keep.append(idx)
                        indices_s_orbitals.append([idx])
                        idx += 1
                    elif _basis_function == 3:
                        basis_functions_orbital.extend(
                            [f"{symbol}p", f"{symbol}p", f"{symbol}p"]
                        )
                        irreps += "+1x1o"
                        indices_to_keep.extend([idx + 1, idx + 2, idx])
                        indices_p_orbitals.append([idx + 1, idx + 2, idx])
                        idx += 3
                    elif _basis_function == 5:
                        if basis_set_type == "full":
                            basis_functions_orbital.extend(
                                [f"{symbol}d" for _ in range(5)]
                            )
                            irreps += "+1x2e"
                            indices_to_keep.extend(
                                [idx, idx + 1, idx + 2, idx + 3, idx + 4]
                            )
                            indices_d_orbitals.append(
                                [idx, idx + 1, idx + 2, idx + 3, idx + 4]
                            )
                        idx += 5
                    elif _basis_function == 7:
                        if basis_set_type == "full":
                            basis_functions_orbital.extend(
                                [f"{symbol}f" for _ in range(7)]
                            )
                            irreps += "+1x3o"
                            indices_to_keep.extend(
                                [
                                    idx,
                                    idx + 1,
                                    idx + 2,
                                    idx + 3,
                                    idx + 4,
                                    idx + 5,
                                    idx + 6,
                                ]
                            )
                            indices_f_orbitals.append(
                                [
                                    idx,
                                    idx + 1,
                                    idx + 2,
                                    idx + 3,
                                    idx + 4,
                                    idx + 5,
                                    idx + 6,
                                ]
                            )
                        idx += 7
                    elif _basis_function == 9:
                        if basis_set_type == "full":
                            basis_functions_orbital.extend(
                                [f"{symbol}g" for _ in range(9)]
                            )
                            irreps += "+1x4e"
                            indices_to_keep.extend(
                                [
                                    idx,
                                    idx + 1,
                                    idx + 2,
                                    idx + 3,
                                    idx + 4,
                                    idx + 5,
                                    idx + 6,
                                    idx + 7,
                                    idx + 8,
                                ]
                            )
                        idx += 9
                    else:
                        raise NotImplementedError

            irreps = irreps[1:]
            base_quantities_qchem = ReducedBasisMatrices(
                fock_matrix=_alpha_fock_matrix,
                eigenvalues=_alpha_eigenvalues,
                coeff_matrix=_alpha_coeff_matrix,
                indices_to_keep=indices_to_keep,
            )

            alpha_ortho_coeff_matrix = base_quantities_qchem.get_ortho_coeff_matrix()
            alpha_coeff_matrix = base_quantities_qchem.get_coeff_matrix()
            alpha_fock_matrix = base_quantities_qchem.get_fock_matrix()
            alpha_eigenvalues = base_quantities_qchem.eigenval_ortho_fock

            data[_basis_set]["alpha_coeff_matrix"].append(alpha_coeff_matrix)
            data[_basis_set]["alpha_ortho_coeff_matrix"].append(
                alpha_ortho_coeff_matrix
            )
            data[_basis_set]["alpha_eigenvalues"].append(alpha_eigenvalues)
            data[_basis_set]["alpha_fock_matrix"].append(alpha_fock_matrix)
            data[_basis_set]["basis_functions_orbital"].append(basis_functions_orbital)
            data[_basis_set]["irreps"].append(irreps)
            data[_basis_set]["euler_angles"].append(doc["tags"]["euler_angles"])
            data[_basis_set]["idx"].append(doc["tags"]["idx"])
            data[_basis_set]["molecule"].append(molecule)

    for _basis_set in basis_sets:
        for key in data[_basis_set].keys():
            if key == "molecule":
                continue
            data[_basis_set][key] = np.array(data[_basis_set][key])

    return data


if __name__ == "__main__":
    """Create the first figure of the manuscript.

    Panels of the figure are:

    a) Coefficient matrix of water at alpha_0, beta_0, gamma_0.
    b) Rotated coefficient matrix of water at alpha, beta and gamma.
    c) Parity plot of rotated coefficients and D-matrix rotated coefficients.
    """

    __output_dir__ = Path("output")
    __output_dir__.mkdir(exist_ok=True)

    basis_set = "def2-svp"
    quantity = "alpha_ortho_coeff_matrix"
    basis_set_type = "full"

    data = get_data(basis_sets=[basis_set])

    alpha_coeff_matrix = data[basis_set][quantity][0]
    rotated_alpha_coeff_matrix = data[basis_set][quantity][2]

    new_coeff_matrix = []
    calculated_coeff_matrix = []

    for _idx in range(len(data[basis_set]["euler_angles"])):

        angles = data[basis_set]["euler_angles"][_idx]
        torch_angles = torch.tensor(angles, dtype=torch.float64)

        irreps = data[basis_set]["irreps"][0]
        irreps = o3.Irreps(irreps)

        rotation_matrix = RotationMatrix(angle_type="euler", angles=angles)()
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64)

        if _idx == 0:
            rotation_matrix_0 = rotation_matrix

        rotation_matrix = rotation_matrix @ rotation_matrix_0.T

        D_matrix = irreps.D_from_matrix(rotation_matrix)
        D_matrix = D_matrix.detach().numpy()

        _new_coeff_matrix = np.zeros_like(alpha_coeff_matrix)
        for i in range(alpha_coeff_matrix.shape[1]):
            _new_coeff_matrix[:, i] = alpha_coeff_matrix[:, i] @ D_matrix.T

        new_coeff_matrix.append(_new_coeff_matrix)
        calculated_coeff_matrix.append(data[basis_set][quantity][_idx])

    new_coeff_matrix = np.array(new_coeff_matrix)
    calculated_coeff_matrix = np.array(calculated_coeff_matrix)

    fig, ax = plt.subplots(1, 3, figsize=(6, 2.0))

    cax = ax[0].imshow(alpha_coeff_matrix, cmap="cividis", vmin=-1, vmax=1)
    cax1 = ax[1].imshow(rotated_alpha_coeff_matrix, cmap="cividis", vmin=-1, vmax=1)

    cbar = fig.colorbar(cax, ax=ax[0])
    cbar1 = fig.colorbar(cax1, ax=ax[1])

    ax[0].set_yticks(np.arange(len(data[basis_set]["basis_functions_orbital"][0])))
    ax[0].set_yticklabels(data[basis_set]["basis_functions_orbital"][0], fontsize=4)
    ax[1].set_yticks(np.arange(len(data[basis_set]["basis_functions_orbital"][0])))
    ax[1].set_yticklabels(data[basis_set]["basis_functions_orbital"][0], fontsize=4)

    eigenvalues = data[basis_set]["alpha_eigenvalues"][0]
    eigenvalues *= ase_units.Hartree
    ax[0].set_xticks(np.arange(len(eigenvalues)))
    ax[0].set_xticklabels(np.round(eigenvalues, 2), rotation=90, fontsize=4)
    ax[1].set_xticks(np.arange(len(eigenvalues)))
    ax[1].set_xticklabels(np.round(eigenvalues, 2), rotation=90, fontsize=4)
    ax[0].set_title(r"$\mathbf{C}\left(\alpha_0,\beta_0,\gamma_0\right)$")
    ax[1].set_title(r"$\mathbf{C}\left(\alpha,\beta,\gamma\right)$")

    ax[0].set_ylabel("Basis function")
    ax[0].set_xlabel("Eigenvalue of molecular orbital [eV]")
    ax[1].set_xlabel("Eigenvalue of molecular orbital [eV]")

    selected_eigenval = eigenvalues[eigenvalues < 0]
    selected_eigenval = np.sort(selected_eigenval)
    selected_eigenval = selected_eigenval[-1]
    selected_eigenval_index = np.where(eigenvalues == selected_eigenval)[0][0]

    for i in range(calculated_coeff_matrix.shape[0]):
        angles = data[basis_set]["euler_angles"][_idx]
        label = r"$\alpha=%1.1f, \beta=%1.1f, \gamma=%1.1f$" % (
            angles[0],
            angles[1],
            angles[2],
        )
        ax[2].plot(
            np.abs(calculated_coeff_matrix[i, :, selected_eigenval_index]),
            np.abs(new_coeff_matrix[i, :, selected_eigenval_index]),
            "o",
            markerfacecolor="none",
            label=label,
        )

    # Draw the parity line
    ax[2].plot(
        [0, 1],
        [0, 1],
        "--",
        color="black",
    )

    ax[2].set_xlabel(
        r"$\left|\mathbf{C}^{\mathrm{HOMO}} (\alpha, \beta, \gamma) \right|$"
    )
    ax[2].set_ylabel(
        r"$\left|\mathbf{C}^{\mathrm{HOMO}} (\alpha_0, \beta_0, \gamma_0) \cdot \mathbf{D}^T(\alpha, \beta, \gamma)\right|$"
    )

    # Switch off minor ticks for the first two ax
    ax[0].tick_params(axis="both", which="both", length=0)
    ax[1].tick_params(axis="both", which="both", length=0)

    fig.tight_layout()
    fig.savefig(__output_dir__ / f"figure1.png")
