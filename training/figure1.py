import json
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

import basis_set_exchange as bse
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from ase import units as ase_units
from ase.data import atomic_numbers
from e3nn import o3
from instance_mongodb import instance_mongodb_sei
from plot_params import get_plot_params
from pymatgen.core.structure import Molecule

from coeffnet.predata.matrices import ReducedBasisMatrices
from coeffnet.transforms.rotations import RotationMatrix

MAX_EIGENVAL = 10
MIN_EIGENVAL = -10
__output_dir__ = Path("output")
__output_dir__.mkdir(exist_ok=True)
basis_set = "def2-svp"
get_plot_params()

def get_angular_moment_for_basis(basis_set, elements=None):
    if not elements:
        elements=list(range(1,18))
    basis_info = bse.get_basis(basis_set, fmt="json", elements=elements)
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
    return angular_momenta

def generate_taskdoc(collection, basis_set):
    find_tags = {
        "orig.rem.basis": basis_set,
        "tags.inverted_coordinates": True,
        "tags.basis_are_spherical": True,
        "orig.rem.purecart": "1111",
    }
    for doc in collection.find(find_tags).sort("tags.idx", 1):
        yield doc

def get_quantity_from_doc(doc, quantity):
    data = doc["calcs_reversed"][0][quantity]
    return np.array(data)

def get_basis_functions_info(molecule, angular_momenta): 
    atom_symbols = [site.specie.symbol for site in molecule.sites]
    atom_numbers = [atomic_numbers[symbol] for symbol in atom_symbols]
    irreps = ""
    basis_functions_orbital = []
    idx = 0
    indices_to_keep = []
    for _idx, atom_number in enumerate(atom_numbers):
        _angular_momenta = angular_momenta[str(atom_number)]
        _angular_momenta = np.array(_angular_momenta)
        _basis_functions = 2 * _angular_momenta + 1
        symbol = atom_symbols[_idx]
        _ns, _np, _nd, _nf, _ng = list(range(1, 6))
        for _basis_function in _basis_functions:
            if _basis_function == 1:
                basis_functions_orbital.extend([f"{symbol}({_ns}s)"])
                indices_to_keep.append(idx)
                irreps += "+1x0e"
                idx += 1
                _ns += 1
            elif _basis_function == 3:
                basis_functions_orbital.extend([f"{symbol}({_np}p)" for _ in range(3)])
                indices_to_keep.extend([idx + 1, idx + 2, idx])
                irreps += "+1x1o"
                idx += 3
                _np += 1
            elif _basis_function == 5:
                basis_functions_orbital.extend([f"{symbol}({_nd}d)" for _ in range(5)])
                indices_to_keep.extend(list(range(idx, idx+5)))
                irreps += "+1x2e"
                _nd += 1
                idx += 5
            elif _basis_function == 7:
                basis_functions_orbital.extend([f"{symbol}({_nf}f)" for _ in range(7)])
                indices_to_keep.extend(list(range(idx, idx+7)))
                irreps += "+1x3o"
                _nf += 1
                idx += 7
            elif _basis_function == 9:
                basis_functions_orbital.extend([f"{symbol}({_ng}g)" for _ in range(9)])
                indices_to_keep.extend(list(range(idx, idx+9)))
                irreps += "+1x4e"
                _ng += 1
                idx += 9
            else:
                raise NotImplementedError
    irreps = irreps[1:]
    return {
        "indices_to_keep":indices_to_keep, 
        "irreps":irreps,
        "basis_functions_orbital": basis_functions_orbital
    }

@dataclass
class DatasetEigenvalueSpin:
    coeff_matrix: npt.ArrayLike = field(default_factory=list)
    ortho_coeff_matrix: npt.ArrayLike = field(default_factory=list)
    eigenvalues: npt.ArrayLike = field(default_factory=list)
    fock_matrix: npt.ArrayLike = field(default_factory=list)
    molecule: Molecule = field(default_factory=list)
    euler_angle: npt.ArrayLike = field(default_factory=list)
    irreps: str = ""
    basis_functions_orbital: npt.ArrayLike = field(default_factory=list)
    rotated_coeff_matrix: npt.ArrayLike = field(default_factory=list)
    rotated_ortho_coeff_matrix: npt.ArrayLike = field(default_factory=list)

    def convert_to_numpy_arrays(self, rotated=False):
        self.coeff_matrix = np.array(self.coeff_matrix)
        self.ortho_coeff_matrix = np.array(self.ortho_coeff_matrix)
        self.eigenvalues = np.array(self.eigenvalues)
        self.fock_matrix = np.array(self.fock_matrix)
        self.euler_angle = np.array(self.euler_angle)
        self.basis_functions_orbital = np.array(self.basis_functions_orbital)
        if rotated:
            self.rotated_coeff_matrix = np.array(self.rotated_coeff_matrix)
            self.rotated_ortho_coeff_matrix = np.array(self.rotated_ortho_coeff_matrix)

def rotate_coeff_matrix(data):
    irreps = o3.Irreps(data.irreps)
    for idx, angle in enumerate(data.euler_angle):
        rotation_matrix = RotationMatrix(angle_type="euler", angles=angle)()
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64)
        if idx == 0:
            rotation_matrix_0 = rotation_matrix
        rotation_matrix = rotation_matrix @ rotation_matrix_0.T
        D_matrix = irreps.D_from_matrix(rotation_matrix)
        D_matrix = D_matrix.detach().numpy()
        rotated_coeff_matrix = np.zeros_like(data.coeff_matrix[idx])
        rotated_ortho_coeff_matrix = np.zeros_like(data.ortho_coeff_matrix[idx])
        for i in range(data.ortho_coeff_matrix.shape[1]):
            rotated_coeff_matrix[:,i] = data.coeff_matrix[0][:,i] @ D_matrix.T
            rotated_ortho_coeff_matrix[:,i] = data.ortho_coeff_matrix[0][:,i] @ D_matrix.T
        data.rotated_coeff_matrix.append(rotated_coeff_matrix)
        data.rotated_ortho_coeff_matrix.append(rotated_ortho_coeff_matrix)

def get_data(collection, basis_set):
    data = defaultdict(lambda: defaultdict(list))
    angular_momenta = get_angular_moment_for_basis(basis_set)
    data = DatasetEigenvalueSpin()
    data.basis_set = basis_set
    for idx, doc in enumerate(generate_taskdoc(collection, basis_set)): 
        raw_alpha_coeff_matrix = get_quantity_from_doc(doc, "alpha_coeff_matrix")
        raw_alpha_fock_matrix = get_quantity_from_doc(doc, "alpha_fock_matrix")
        raw_alpha_eigenvalues = get_quantity_from_doc(doc, "alpha_eigenvalues")
        molecule = Molecule.from_dict(doc["orig"]["molecule"])
        if idx == 0:
            basis_functions_info = get_basis_functions_info(molecule, angular_momenta)
            data.basis_functions_orbital = basis_functions_info["basis_functions_orbital"]
            data.irreps = basis_functions_info["irreps"]
        base_quantities_qchem = ReducedBasisMatrices(
            fock_matrix=raw_alpha_fock_matrix,
            eigenvalues=raw_alpha_eigenvalues,
            coeff_matrix=raw_alpha_coeff_matrix,
            indices_to_keep=basis_functions_info["indices_to_keep"],
        )
        data.coeff_matrix.append(base_quantities_qchem.get_coeff_matrix())
        data.ortho_coeff_matrix.append(base_quantities_qchem.get_ortho_coeff_matrix())
        data.fock_matrix.append(base_quantities_qchem.get_fock_matrix())
        data.eigenvalues.append(base_quantities_qchem.eigenval_ortho_fock)
        data.molecule.append(molecule)
        data.euler_angle.append(doc["tags"]["euler_angles"])
    data.convert_to_numpy_arrays(rotated=False)
    rotate_coeff_matrix(data)
    data.convert_to_numpy_arrays(rotated=True)
    return data


if __name__ == "__main__":
    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_water_calculations
    data = get_data(basis_set=basis_set, collection=collection)
    eigenvals = data.eigenvalues[0] * ase_units.Hartree
    fig = plt.figure(figsize=(3.25, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 3.5])
    ax = gs.subplots()
    _pind = np.where((eigenvals < MAX_EIGENVAL) & (eigenvals > MIN_EIGENVAL))[0]
    plot_kwargs = {"cmap": "coolwarm", "vmin":-1, "vmax":1}
    cax = ax[0].imshow(data.ortho_coeff_matrix[0,:,_pind].T, **plot_kwargs)
    cax1 = ax[1].imshow(data.ortho_coeff_matrix[1,:,_pind].T, **plot_kwargs)
    cbar = fig.colorbar(cax, ax=ax[0])
    cbar1 = fig.colorbar(cax1, ax=ax[1])
    cbar.ax.tick_params(labelsize=5)
    cbar1.ax.tick_params(labelsize=5)
    ax[0].set_yticks(np.arange(len(data.basis_functions_orbital)))
    ax[0].set_yticklabels(data.basis_functions_orbital, fontsize=3)
    ax[1].set_yticks(np.arange(len(data.basis_functions_orbital)))
    ax[1].set_yticklabels(data.basis_functions_orbital, fontsize=3)
    for _ax in ax[0:2]:
        _ax.set_xticks(np.arange(len(eigenvals[_pind])))
        _ax.set_xticklabels(
            np.round(eigenvals[_pind], 2),
            rotation=90,
            fontsize=4,
        )
    ax[0].set_title(
        r"a) $\mathbf{C}\left(\alpha_0,\beta_0,\gamma_0\right)$", fontsize=6
    )
    ax[1].set_title(r"b) $\mathbf{C}\left(\alpha,\beta,\gamma\right)$", fontsize=6)
    ax[0].set_ylabel("Atomic orbital", fontsize=5)
    ax[1].set_ylabel("Atomic orbital", fontsize=5)
    ax[0].set_xlabel("Eigenvalue [eV]", fontsize=5)
    ax[1].set_xlabel("Eigenvalue [eV]", fontsize=5)
    selected_eigenval = eigenvals[eigenvals < 0]
    selected_eigenval = np.sort(selected_eigenval)
    selected_eigenval = selected_eigenval[-1]
    selected_eigenval_index = np.where(eigenvals == selected_eigenval)[0][0]
    colors = plt.cm.jet(np.linspace(0, 1, data.euler_angle.shape[0]))
    for i in range(data.euler_angle.shape[0]):
        angles = data.euler_angle[i]
        angles /= np.pi
        label = r"$\alpha=%s, \beta=%s, \gamma=%s$" % (
            f"{angles[0]:.1f}" + r"\pi",
            f"{angles[1]:.1f}" + r"\pi",
            f"{angles[2]:.1f}" + r"\pi",
        )
        ax[2].plot(
            np.abs(data.ortho_coeff_matrix[i, :, selected_eigenval_index]),
            np.abs(data.rotated_ortho_coeff_matrix[i, :, selected_eigenval_index]),
            "o",
            markerfacecolor="none",
            label=label,
            color=colors[i],
        )
    ax[2].legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=1,
        fontsize=3,
    )
    ax[2].plot(
        [0, 1],
        [0, 1],
        "--",
        color="black",
    )
    ax[2].set_xlabel(
        r"$\left|\mathbf{C}^{\mathrm{HOMO}} (\alpha, \beta, \gamma) \right|$",
        fontsize=5,
    )
    ax[2].set_ylabel(
        r"""
$\left|\mathbf{C}^{\mathrm{HOMO}} (\alpha_0, \beta_0, \gamma_0)\cdot \mathbf{D}^T(\alpha, \beta, \gamma)\right|$""",
        fontsize=5,
    )
    ax[2].tick_params(axis="x", labelsize=5)
    ax[2].tick_params(axis="y", labelsize=5)
    ax[0].tick_params(axis="both", which="both", length=0)
    ax[1].tick_params(axis="both", which="both", length=0)
    ax[2].set_aspect("equal", "box")
    ax[2].text(
        0.2,
        0.8,
        "c)",
        transform=ax[2].transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(__output_dir__ / f"figure1.png", dpi=400)
    fig.savefig(__output_dir__ / f"figure1.pdf", dpi=400, transparent=True)
