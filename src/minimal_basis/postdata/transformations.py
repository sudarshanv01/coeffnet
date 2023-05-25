try:
    from pyscf import gto
    from pyscf.dft.numint import eval_ao
except ImportError:
    raise ImportError("You need to install pyscf to use this module")

import numpy as np
import numpy.typing as npt

from ase.data import chemical_symbols


class OrthoCoeffMatrixToMolecularOrbitals:
    def __init__(
        self,
        grid: npt.ArrayLike,
        ortho_coeff_matrix: npt.ArrayLike,
        orthogonalization_matrix: npt.ArrayLike,
        positions: npt.ArrayLike,
        species: npt.ArrayLike,
        basis_name: str,
        indices_to_keep: npt.ArrayLike,
        charge: int = 0,
        uses_carterian_orbitals: bool = False,
    ):
        """Construct the molecular orbitals from the orthogonal coefficient matrix."""

        self.grid = np.asarray(grid)
        self.ortho_coeff_matrix = np.asarray(ortho_coeff_matrix)
        self.orthogonalization_matrix = np.asarray(orthogonalization_matrix)
        self.positions = np.asarray(positions)
        self.species = np.asarray(species)
        self.basis_name = basis_name
        self.indices_to_keep = np.asarray(indices_to_keep)
        self.uses_cartesian_orbitals = uses_carterian_orbitals
        self.charge = charge

    def store_gto_molecule(self):
        """Get the GTO molecule from PySCF."""
        if self.pyscf_mol:
            return self.pyscf_mol
        species_names = [
            chemical_symbols[int(self.species[i])] for i in range(len(self.species))
        ]
        atoms_real_input = ""
        for i in range(len(self.species)):
            atoms_real_input += f"{species_names[i]} {self.positions[i][0]} {self.positions[i][1]} {self.positions[i][2]}; "
        atoms_input = atoms_real_input[:-2]
        pyscf_mol = gto.M(
            atom=atoms_input,
            basis=self.basis_name,
            charge=self.charge,
            cart=self.uses_cartesian_orbitals,
        )
        self.pyscf_mol = pyscf_mol

    def store_atomic_orbitals_on_grid(self):
        """Gets the atomic orbitals on a grid. The output dimensions of this
        matrix should be Nxnao, where N is the number of grid points and nao
        is the number of atomic orbitals as shown in the documentation of
        pyscf.
        https://pyscf.org/pyscf_api_docs/pyscf.dft.html
        """
        pyscf_mol = self.get_gto_molecule()
        ao = eval_ao(pyscf_mol, self.grid)
        self.ao = ao[:, self.indices_to_keep]

    def reorder_atomic_orbitals(self):
        """The atomic orbitals that come from pyscf are ordered by the
        basis sets that define them. This method changes the order from
        this pre-defined order to all s-functions first, followed by all
        p-functions, followed by all d-functions.
        """
        self.orbital_labels = [a[4:6] for a in self.pyscf_mol.ao_labels()]
        self.orbital_labels = np.asarray(self.orbital_labels)
        self.orbital_labels = self.orbital_labels[self.indices_to_keep]

        # Store the array required to reorder the atomic orbitals
        # such that all s-orbitals come first, followed by all p-orbitals,
        # followed by all d-orbitals.
