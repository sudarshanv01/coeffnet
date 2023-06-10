import os

from pathlib import Path

import json

from ase.data import atomic_numbers

from collections import defaultdict

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import basis_set_exchange as bse

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300

from minimal_basis.predata.matrices import BaseMatrices
from minimal_basis.transforms.rotations import RotationMatrix

from instance_mongodb import instance_mongodb_sei


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

            for idx, atom_number in enumerate(atom_numbers):
                _angular_momenta = angular_momenta[str(atom_number)]
                _angular_momenta = np.array(_angular_momenta)
                _basis_functions = 2 * _angular_momenta + 1
                symbol = symbols[idx]
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
            base_quantities_qchem = BaseMatrices(
                fock_matrix=_alpha_fock_matrix,
                eigenvalues=_alpha_eigenvalues,
                coeff_matrix=_alpha_coeff_matrix,
            )

            alpha_ortho_coeff_matrix = base_quantities_qchem.get_ortho_coeff_matrix()
            alpha_coeff_matrix = base_quantities_qchem.get_coeff_matrix()
            alpha_fock_matrix = base_quantities_qchem.get_fock_matrix()
            alpha_eigenvalues = base_quantities_qchem.eigenval_ortho_fock

            data[_basis_set]["alpha_coeff_matrix"].append(alpha_coeff_matrix)
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

    basis_set = "sto-3g"
    quantity = "alpha_coeff_matrix"
    basis_set_type = "full"

    data = get_data(basis_sets=[basis_set])

    alpha_coeff_matrix = data[basis_set][quantity][0]
    rotated_alpha_coeff_matrix = data[basis_set][quantity][1]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    cax = ax[0].imshow(alpha_coeff_matrix, cmap="cividis")
    ax[1].imshow(rotated_alpha_coeff_matrix, cmap="cividis")

    # Make a colorbar for the heatmap.
    cbar = fig.colorbar(cax, ax=ax[0])
    cbar.ax.set_ylabel("Coefficient value", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax[0].set_yticks(np.arange(len(data[basis_set]["basis_functions_orbital"][0])))
    ax[0].set_yticklabels(data[basis_set]["basis_functions_orbital"][0])
    ax[1].set_yticks(np.arange(len(data[basis_set]["basis_functions_orbital"][0])))
    ax[1].set_yticklabels(data[basis_set]["basis_functions_orbital"][0])

    # Use the x-label as the alpha eigenvalues
    eigenvalues = data[basis_set]["alpha_eigenvalues"][0]
    ax[0].set_xticks(np.arange(len(eigenvalues)))
    ax[0].set_xticklabels(np.round(eigenvalues, 2))
    ax[1].set_xticks(np.arange(len(eigenvalues)))
    ax[1].set_xticklabels(np.round(eigenvalues, 2))

    fig.savefig(__output_dir__ / f"figure1.png")
