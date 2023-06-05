from typing import List, Dict, Union, Any
import logging

import numpy.typing as npt
import numpy as np

from ase import data as ase_data

import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph

from e3nn import o3

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)
from minimal_basis.predata.cart_to_sph import cart_to_sph_d

logger = logging.getLogger(__name__)


def convert_to_tensor(
    x: Union[npt.ArrayLike, List[float]],
    dtype: TORCH_FLOATS = DTYPE,
) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    if isinstance(x, float):
        x = torch.tensor([x], dtype=dtype)
    else:
        x = torch.tensor(x, dtype=dtype)
    return x


class ReactionDataPoint(Data):
    def __init__(
        self,
        x: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        edge_index: Dict[str, Union[npt.ArrayLike, List[int]]] = None,
        pos: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        total_energies: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        species: Dict[str, Union[npt.ArrayLike, List[int]]] = None,
        orthogonalization_matrix: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        basis_mask: Union[npt.ArrayLike, List[bool]] = None,
        reactant_tag: str = None,
        product_tag: str = None,
        transition_state_tag: str = None,
        indices_to_keep: Union[npt.ArrayLike, List[int]] = None,
        **kwargs,
    ):
        """General purpose data class for reaction data."""

        if pos is not None:
            pos_initial_state = pos[reactant_tag]
            pos_final_state = pos[product_tag]
            pos_interpolated_transition_state = pos["interpolated_transition_state"]
            pos_transition_state = pos[transition_state_tag]

            pos_initial_state = convert_to_tensor(pos_initial_state)
            pos_final_state = convert_to_tensor(pos_final_state)
            pos_interpolated_transition_state = convert_to_tensor(
                pos_interpolated_transition_state
            )
            pos_transition_state = convert_to_tensor(pos_transition_state)
        else:
            pos_initial_state = None
            pos_final_state = None
            pos_interpolated_transition_state = None
            pos_transition_state = None

        if x is not None:
            x_initial_state = x[reactant_tag]
            x_final_state = x[product_tag]
            x_transition_state = x[transition_state_tag]

            x_initial_state = convert_to_tensor(x_initial_state)
            x_final_state = convert_to_tensor(x_final_state)
            x_transition_state = convert_to_tensor(x_transition_state)

        else:
            x_initial_state = None
            x_final_state = None
            x_transition_state = None

        if edge_index is not None:
            edge_index_initial_state = edge_index[reactant_tag]
            edge_index_final_state = edge_index[product_tag]
            edge_index_interpolated_transition_state = edge_index[
                "interpolated_transition_state"
            ]
            edge_index_transition_state = edge_index[transition_state_tag]

            edge_index_initial_state = convert_to_tensor(
                edge_index_initial_state, dtype=DTYPE_INT
            )
            edge_index_final_state = convert_to_tensor(
                edge_index_final_state, dtype=DTYPE_INT
            )
            edge_index_interpolated_transition_state = convert_to_tensor(
                edge_index_interpolated_transition_state, dtype=DTYPE_INT
            )
            edge_index_transition_state = convert_to_tensor(
                edge_index_transition_state, dtype=DTYPE_INT
            )
        else:
            edge_index_initial_state = None
            edge_index_final_state = None
            edge_index_interpolated_transition_state = None
            edge_index_transition_state = None

        if total_energies is not None:
            total_energy_initial_state = total_energies[reactant_tag]
            total_energy_final_state = total_energies[product_tag]
            total_energy_transition_state = total_energies[transition_state_tag]

            total_energy_initial_state = convert_to_tensor(total_energy_initial_state)
            total_energy_final_state = convert_to_tensor(total_energy_final_state)
            total_energy_transition_state = convert_to_tensor(
                total_energy_transition_state
            )

        else:
            total_energy_initial_state = None
            total_energy_final_state = None
            total_energy_transition_state = None

        if species is not None:
            species_initial_state = species[reactant_tag]
            species_final_state = species[product_tag]
            species_transition_state = species[transition_state_tag]

            species_initial_state = convert_to_tensor(species_initial_state)
            species_final_state = convert_to_tensor(species_final_state)
            species_transition_state = convert_to_tensor(species_transition_state)

        else:
            species_initial_state = None
            species_final_state = None
            species_transition_state = None

        if orthogonalization_matrix is not None:
            orthogonalization_matrix_initial_state = orthogonalization_matrix[
                reactant_tag
            ]
            orthogonalization_matrix_final_state = orthogonalization_matrix[product_tag]
            orthogonalization_matrix_transition_state = orthogonalization_matrix[
                transition_state_tag
            ]

            orthogonalization_matrix_initial_state = convert_to_tensor(
                orthogonalization_matrix_initial_state
            )
            orthogonalization_matrix_final_state = convert_to_tensor(
                orthogonalization_matrix_final_state
            )
            orthogonalization_matrix_transition_state = convert_to_tensor(
                orthogonalization_matrix_transition_state
            )

        else:
            orthogonalization_matrix_initial_state = None
            orthogonalization_matrix_final_state = None
            orthogonalization_matrix_transition_state = None

        if basis_mask is not None:
            basis_mask = convert_to_tensor(basis_mask, dtype=DTYPE_BOOL)
        else:
            basis_mask = None

        if indices_to_keep is not None:
            indices_to_keep = convert_to_tensor(indices_to_keep, dtype=DTYPE_INT)
        else:
            indices_to_keep = None

        super().__init__(
            x=x_initial_state,
            x_final_state=x_final_state,
            x_transition_state=x_transition_state,
            pos=pos_initial_state,
            pos_final_state=pos_final_state,
            pos_interpolated_transition_state=pos_interpolated_transition_state,
            pos_transition_state=pos_transition_state,
            edge_index=edge_index_initial_state,
            edge_index_final_state=edge_index_final_state,
            edge_index_interpolated_transition_state=edge_index_interpolated_transition_state,
            edge_index_transition_state=edge_index_transition_state,
            total_energy=total_energy_initial_state,
            total_energy_final_state=total_energy_final_state,
            total_energy_transition_state=total_energy_transition_state,
            species=species_initial_state,
            orthogonalization_matrix=orthogonalization_matrix_initial_state,
            orthogonalization_matrix_final_state=orthogonalization_matrix_final_state,
            orthogonalization_matrix_transition_state=orthogonalization_matrix_transition_state,
            basis_mask=basis_mask,
            indices_to_keep=indices_to_keep,
            **kwargs,
        )


class CoefficientMatrixSphericalBasis:

    l_to_n_basis = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    n_to_l_basis = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}

    def __init__(
        self,
        molecule_graph: MoleculeGraph,
        orbital_info: npt.ArrayLike,
        coefficient_matrix: npt.ArrayLike,
        store_idx_only: int = None,
        indices_to_keep: List[int] = None,
        **kwargs,
    ):
        """Store the coefficient matrix and provides some utilities to manipulate it.

        Args:
            molecule_graph (MoleculeGraph): The molecule graph.
            orbital_info (npt.ArrayLike): The orbital information [atom_symbol, m, n].
            coefficient_matrix (npt.ArrayLike): The coefficient matrix as computed.
            store_idx_only (int, optional): If not None, only store the coefficient matrix for this index. Defaults to None.
            indices_to_keep (List[int], optional): The indices of the basis functions to keep. Defaults to None.
        """

        if store_idx_only is not None:
            self.coefficient_matrix = coefficient_matrix[:, store_idx_only]
            self.coefficient_matrix = self.coefficient_matrix[:, np.newaxis]
        else:
            self.coefficient_matrix = coefficient_matrix

        self.molecule_graph = molecule_graph
        self.orbital_info = orbital_info
        self.cart_to_spherical_d = cart_to_sph_d()

        if indices_to_keep is None:
            self.indices_to_keep = list(range(len(self.coefficient_matrix)))
        else:
            self.indices_to_keep = indices_to_keep

        self.get_basis_index()

    def get_coefficient_matrix(self):
        """Return the coefficient matrix."""
        return self.coefficient_matrix

    def get_coefficient_matrix_for_basis_function(self, basis_idx: int):
        """Return the coefficient matrix for a given basis_function."""
        return self.coefficient_matrix[basis_idx, :]

    def get_coefficient_matrix_for_eigenstate(self, eigenstate_idx: int):
        """Return the coefficient matrix for a given eigenstate."""
        return self.coefficient_matrix[:, eigenstate_idx]

    def get_coefficient_matrix_for_atom(self, atom_idx: int):
        """Return the coefficient matrix for a given atom."""
        self.separate_coeff_matrix_to_atom_centers()
        return self.coefficient_matrix_atom[atom_idx]

    def get_irreps_for_atom(self, atom_idx: int):
        """Return the irreps for a given atom."""
        self.separate_coeff_matrix_to_atom_centers()
        return o3.Irreps(self.irreps_all_atom[atom_idx])

    def get_padded_coefficient_matrix_for_atom(self, atom_idx: int):
        """Return the coefficient matrix for a given atom."""
        self.separate_coeff_matrix_to_atom_centers()
        self.pad_split_coefficient_matrix()
        return self.coefficient_matrix_atom_centers_padded[atom_idx]

    def _get_atomic_number(cls, symbol):
        """Get the atomic number of an element based on the symbol."""
        return ase_data.atomic_numbers[symbol]

    def get_basis_index(self):
        """Get the basis index for each atom in the atom in the molecule."""
        atom_symbol, atom_idx, m, n = self.orbital_info.T
        atom_idx = atom_idx.astype(int)
        atom_idx_unique = np.unique(atom_idx)
        basis_atom = []
        for _atom_idx in atom_idx_unique:
            basis_atom.append(np.where(atom_idx == _atom_idx)[0])
        self.basis_atom = basis_atom
        basis_idx_s = []
        basis_idx_p = []
        basis_idx_d = []
        for _basis_atom in self.basis_atom:
            basis_idx_s.append(np.where(m[_basis_atom] == "s")[0].tolist())
            basis_idx_p.append(np.where(m[_basis_atom] == "p")[0].tolist())
            basis_idx_d.append(np.where(m[_basis_atom] == "d")[0].tolist())
        self.basis_idx_s = basis_idx_s
        self.basis_idx_p = basis_idx_p
        self.basis_idx_d = basis_idx_d

    def separate_coeff_matrix_to_atom_centers(self):
        """Split the coefficients to the atoms they belong to."""
        self.coefficient_matrix_atom = []
        for atom_idx, atom in enumerate(self.molecule_graph.molecule):
            _atom_basis = self.basis_atom[atom_idx]
            self.coefficient_matrix_atom.append(self.coefficient_matrix[_atom_basis, :])

    def pad_split_coefficient_matrix(self):
        """Once split, pad the coefficient matrix to the maximum number of basis."""
        max_basis = max([len(atom) for atom in self.basis_atom])

        self.coefficient_matrix_atom_centers_padded = []

        for atom_idx, atom in enumerate(self.molecule_graph.molecule):
            _atom_basis = self.basis_atom[atom_idx]
            if len(_atom_basis) < max_basis:
                pad = max_basis - len(_atom_basis)
                self.coefficient_matrix_atom_centers_padded.append(
                    np.pad(
                        self.coefficient_matrix_atom[atom_idx],
                        ((0, pad), (0, 0)),
                        "constant",
                    )
                )
            else:
                self.coefficient_matrix_atom_centers_padded.append(
                    np.array(self.coefficient_matrix_atom[atom_idx])
                )


class ModifiedCoefficientMatrixSphericalBasis(CoefficientMatrixSphericalBasis):

    minimal_basis_irrep = o3.Irreps("1x0e+1x1o")

    def __init__(
        self,
        molecule_graph: MoleculeGraph,
        orbital_info: npt.ArrayLike,
        coefficient_matrix: npt.ArrayLike,
        store_idx_only: int = None,
        set_to_absolute: bool = False,
        max_s_functions: Union[str, int] = None,
        max_p_functions: Union[str, int] = None,
        max_d_functions: Union[str, int] = None,
        indices_to_keep: List[int] = None,
        **kwargs,
    ):
        """Modify the coefficient matrix representing each atom by a fixed basis.
        
        Args:
            molecule_graph (MoleculeGraph): Molecule graph object.
            orbital_info (npt.ArrayLike): Orbital information.
            coefficient_matrix (npt.ArrayLike): Coefficient matrix.
            store_idx_only (int, optional): Store only the indices of the basis functions.
            set_to_absolute (bool, optional): Set the coefficients to absolute values.
            max_s_functions (Union[str, int], optional): Maximum number of s functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
            max_p_functions (Union[str, int], optional): Maximum number of p functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
            max_d_functions (Union[str, int], optional): Maximum number of d functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
        """
        super().__init__(
            molecule_graph=molecule_graph,
            orbital_info=orbital_info,
            coefficient_matrix=coefficient_matrix,
            store_idx_only=store_idx_only,
            set_to_absolute=set_to_absolute,
            indices_to_keep=indices_to_keep,
            **kwargs,
        )

        self.max_s_functions = max_s_functions
        self.max_p_functions = max_p_functions
        self.max_d_functions = max_d_functions

    def pad_split_coefficient_matrix(self):
        raise NotImplementedError(
            "This method is not implemented for this class\
                                  as the coefficient matrix for each atom is already\
                                  represented by a fixed basis and hence has the same \
                                  dimensions."
        )

    def get_node_features(self):
        """Get the node features based on the specifications."""
        return self.get_padded_representation()

    def get_padded_representation(self):
        """Return the padded representation of the coefficient matrix."""
        self.separate_coeff_matrix_to_atom_centers()
        self.generate_padded_representation()
        return self.coefficient_matrix_padded

    def generate_padded_representation(self):
        """In this representation, the coefficient matrix for each
        atom is padded with zeros to the maximum basis function for
        each orbital."""

        if self.max_s_functions == "all":
            max_s = max([len(atom) for atom in self.basis_idx_s])
            max_p = max([len(atom) for atom in self.basis_idx_p])
            max_d = max([len(atom) for atom in self.basis_idx_d])
        elif (
            isinstance(self.max_s_functions, int)
            and isinstance(self.max_p_functions, int)
            and isinstance(self.max_d_functions, int)
        ):
            max_s = self.max_s_functions
            max_p = self.max_p_functions
            max_d = self.max_d_functions
        else:
            raise ValueError(
                "The maximum number of s, p and d functions must be either 'all' or an integer."
            )
        num_atoms = len(self.molecule_graph.molecule)
        self.coefficient_matrix_padded = np.zeros(
            [
                num_atoms,
                max_s + 3 * max_p + 5 * max_d,
                self.coefficient_matrix.shape[1],
            ],
        )
        self.basis_mask = np.zeros(
            [
                num_atoms,
                max_s + 3 * max_p + 5 * max_d,
            ],
        )

        for atom_idx, atom in enumerate(self.molecule_graph.molecule):

            _s_basis_idx = self.basis_idx_s[atom_idx]
            _s_basis_idx = np.array(_s_basis_idx)
            _s_basis_idx = _s_basis_idx.flatten()
            s_basis_idx = self.basis_atom[atom_idx][_s_basis_idx]

            pad = max_s - len(_s_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of s functions is greater than the maximum number of s functions."
                )
            self.coefficient_matrix_padded[atom_idx, :max_s, :] = np.pad(
                self.coefficient_matrix[s_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, :max_s] = np.pad(
                np.ones(len(s_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )

            _p_basis_idx = self.basis_idx_p[atom_idx]
            _p_basis_idx = np.array(_p_basis_idx)
            _p_basis_idx = _p_basis_idx.flatten()
            if _p_basis_idx.size == 0:
                continue
            p_basis_idx = self.basis_atom[atom_idx][_p_basis_idx]

            pad = 3 * max_p - len(p_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of p functions is greater than the maximum number of p functions."
                )
            self.coefficient_matrix_padded[
                atom_idx, max_s : max_s + 3 * max_p, :
            ] = np.pad(
                self.coefficient_matrix[p_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, max_s : max_s + 3 * max_p] = np.pad(
                np.ones(len(p_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )

            _d_basis_idx = self.basis_idx_d[atom_idx]
            _d_basis_idx = np.array(_d_basis_idx)
            _d_basis_idx = _d_basis_idx.flatten()
            if _d_basis_idx.size == 0:
                continue
            d_basis_idx = self.basis_atom[atom_idx][_d_basis_idx]

            pad = 5 * max_d - len(_d_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of d functions is greater than the maximum number of d functions."
                )
            self.coefficient_matrix_padded[atom_idx, max_s + 3 * max_p :, :] = np.pad(
                self.coefficient_matrix[d_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, max_s + 3 * max_p :] = np.pad(
                np.ones(len(d_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )
