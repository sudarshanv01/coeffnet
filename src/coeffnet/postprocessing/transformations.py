try:
    from pyscf import gto
    from pyscf.dft.numint import eval_ao
except ImportError:
    raise ImportError("You need to install pyscf to use this module")

import numpy as np
import numpy.typing as npt

import pandas as pd

from ase.data import chemical_symbols


class OrthoCoeffMatrixToGridQuantities:
    def __init__(
        self,
        ortho_coeff_matrix: npt.ArrayLike,
        orthogonalization_matrix: npt.ArrayLike,
        positions: npt.ArrayLike,
        species: npt.ArrayLike,
        basis_name: str,
        indices_to_keep: npt.ArrayLike,
        charge: int = 0,
        uses_carterian_orbitals: bool = False,
        buffer_grid: float = 5.0,
        grid_points: int = 100,
    ):
        """Construct the molecular orbitals from the orthogonal coefficient matrix.

        Args:
            ortho_coeff_matrix (npt.ArrayLike): The orthogonal coefficient matrix.
            orthogonalization_matrix (npt.ArrayLike): The orthogonalization matrix; S^(-1/2).
            positions (npt.ArrayLike): The positions of the atoms in the molecule.
            species (npt.ArrayLike): The species of the atoms in the molecule.
            basis_name (str): The name of the basis set.
            indices_to_keep (npt.ArrayLike): The indices of the orbitals to keep.
            charge (int, optional): The charge of the molecule. Defaults to 0.
            uses_carterian_orbitals (bool, optional): Whether the basis set uses cartesian orbitals.
             Passed directly to pyscf mol. Defaults to False.
            buffer_grid (float, optional): The buffer around the molecule to generate the grid.
                Defaults to 5.0 Angstroms.
            grid_points (int, optional): The number of grid points in each direction. Defaults to 100.
        """

        self.ortho_coeff_matrix = np.asarray(ortho_coeff_matrix)
        self.orthogonalization_matrix = np.asarray(orthogonalization_matrix)
        self.positions = np.asarray(positions)
        self.species = np.asarray(species)
        self.basis_name = basis_name
        self.indices_to_keep = np.asarray(indices_to_keep)
        self.uses_cartesian_orbitals = uses_carterian_orbitals
        self.charge = charge
        self.buffer_grid = buffer_grid
        self.grid_points = grid_points

        self.pyscf_mol = None
        self.atomic_orbitals = None

    def __call__(self):
        self.generate_grid()
        self.store_gto_molecule()
        self.store_atomic_orbitals_on_grid()
        self.store_coeff_matrix()
        self.store_molecular_orbital()

    def generate_grid(self):
        """Create a 3D grid around the molecule with a buffer."""
        x_min = np.min(self.positions[:, 0]) - self.buffer_grid
        x_max = np.max(self.positions[:, 0]) + self.buffer_grid
        y_min = np.min(self.positions[:, 1]) - self.buffer_grid
        y_max = np.max(self.positions[:, 1]) + self.buffer_grid
        z_min = np.min(self.positions[:, 2]) - self.buffer_grid
        z_max = np.max(self.positions[:, 2]) + self.buffer_grid

        grid = np.mgrid[
            x_min : x_max : self.grid_points * 1j,
            y_min : y_max : self.grid_points * 1j,
            z_min : z_max : self.grid_points * 1j,
        ]
        self.grid = grid.reshape(3, -1).T

    def get_atomic_orbitals(self):
        """Get the atomic orbitals."""
        return self.atomic_orbitals

    def get_molecular_orbital(self):
        """Get the molecular orbitals."""
        return self.molecular_orbital

    def get_grid(self):
        """Get the grid."""
        return self.grid

    def store_gto_molecule(self):
        """Store the GTO molecule from PySCF."""
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

        The order that pyscf uses to store its atomic orbitals is different from
        what QChem uses. We need to sort the atomic orbitals in the same order
        as this electronic structure code in order to make sure that the coefficients
        are dot-producted with the correct atomic orbitals.
        """
        ao = eval_ao(self.pyscf_mol, self.grid)
        pyscf_labels = self.pyscf_mol.ao_labels()
        pyscf_labels = [label.split()[0:3] for label in pyscf_labels]
        pyscf_labels = [
            d[:-1] + [d[-1][0]] + [d[-1][1]] + [d[-1][2:]] for d in pyscf_labels
        ]
        pyscf_labels = pd.DataFrame(
            pyscf_labels, columns=["atom_idx", "atom_name", "n", "l", "m"]
        )
        pyscf_labels["atom_idx"] = pyscf_labels["atom_idx"].astype(int)
        pyscf_labels["n"] = pyscf_labels["n"].astype(int)
        indices = pyscf_labels.sort_values(
            by=["atom_idx", "n", "l"], ascending=[True, True, False]
        ).index
        ao = ao[:, indices]
        self.atomic_orbitals = ao[:, self.indices_to_keep]

    def store_coeff_matrix(self):
        """Convert the Coefficient matrix from the orthogonal basis to the
        non-orthogonal basis."""
        self.coeff_matrix = np.matmul(
            self.orthogonalization_matrix.T, self.ortho_coeff_matrix
        )

    def store_molecular_orbital(self):
        """Get the molecular orbitals by multiplying the atomic orbitals with
        the coefficient matrix."""
        self.molecular_orbital = np.matmul(self.atomic_orbitals, self.coeff_matrix)


class NodeFeaturesToOrthoCoeffMatrix:
    def __init__(self, node_features: npt.ArrayLike, mask: npt.ArrayLike):
        """Convert the node features to the orthogonal coefficient matrix.
        Args:
            node_features (npt.ArrayLike): The node features.
            mask (npt.ArrayLike): The mask of the node features.
        """
        self.node_features = np.asarray(node_features)
        self.mask = np.asarray(mask)

        self.ortho_coeff_matrix = None

    def __call__(self):
        self.store_ortho_coeff_matrix()

    def get_ortho_coeff_matrix(self):
        """Get the orthogonal coefficient matrix."""
        return self.ortho_coeff_matrix

    def store_ortho_coeff_matrix(self):
        """Store the orthogonal coefficient matrix. First the mask is applied
        to the node features. Then the node features are flattened."""
        self.ortho_coeff_matrix = self.node_features[self.mask]
        self.ortho_coeff_matrix = self.ortho_coeff_matrix.flatten()


class DatapointStoredVectorToOrthogonlizationMatrix:
    def __init__(self, datapoint_stored_vector: npt.ArrayLike):
        """Convert the flattened vector stored in the datapoint to
        the orthogonalization matrix of the right dimensions."""
        self.datapoint_stored_vector = np.asarray(datapoint_stored_vector)
        self.orthogonalization_matrix = None

    def get_orthogonalization_matrix(self):
        """Get the orthogonalization matrix."""
        return self.orthogonalization_matrix

    def __call__(self):
        N = self.datapoint_stored_vector.shape[0]
        n = int(np.sqrt(N))
        self.orthogonalization_matrix = self.datapoint_stored_vector.reshape((n, n))
