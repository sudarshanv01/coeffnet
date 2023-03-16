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
        basis_mask: Union[npt.ArrayLike, List[bool]] = None,
        **kwargs,
    ):
        """General purpose data class for reaction data."""

        if pos is not None:
            pos_initial_state = pos["initial_state"]
            pos_final_state = pos["final_state"]
            pos_interpolated_transition_state = pos["interpolated_transition_state"]
            pos_transition_state = pos["transition_state"]

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
            x_initial_state = x["initial_state"]
            x_final_state = x["final_state"]
            x_transition_state = x["transition_state"]

            x_initial_state = convert_to_tensor(x_initial_state)
            x_final_state = convert_to_tensor(x_final_state)
            x_transition_state = convert_to_tensor(x_transition_state)

        else:
            x_initial_state = None
            x_final_state = None
            x_transition_state = None

        if edge_index is not None:
            edge_index_initial_state = edge_index["initial_state"]
            edge_index_final_state = edge_index["final_state"]
            edge_index_interpolated_transition_state = edge_index[
                "interpolated_transition_state"
            ]
            edge_index_transition_state = edge_index["transition_state"]

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
            total_energy_initial_state = total_energies["initial_state"]
            total_energy_final_state = total_energies["final_state"]
            total_energy_transition_state = total_energies["transition_state"]

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
            species_initial_state = species["initial_state"]
            species_final_state = species["final_state"]
            species_transition_state = species["transition_state"]

            species_initial_state = convert_to_tensor(species_initial_state)
            species_final_state = convert_to_tensor(species_final_state)
            species_transition_state = convert_to_tensor(species_transition_state)

        else:
            species_initial_state = None
            species_final_state = None
            species_transition_state = None

        if basis_mask is not None:
            basis_mask = convert_to_tensor(basis_mask, dtype=DTYPE_BOOL)
        else:
            basis_mask = None

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
            basis_mask=basis_mask,
            **kwargs,
        )


class CoefficientMatrix:

    l_to_n_basis = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    n_to_l_basis = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}

    def __init__(
        self,
        molecule_graph: MoleculeGraph,
        basis_info_raw: Dict[str, Any],
        coefficient_matrix: npt.ArrayLike,
        store_idx_only: int = None,
        set_to_absolute: bool = False,
        **kwargs,
    ):
        """Store the coefficient matrix and provides some utilities to manipulate it."""

        if store_idx_only is not None:
            self.coefficient_matrix = coefficient_matrix[:, store_idx_only]
            self.coefficient_matrix = self.coefficient_matrix[:, np.newaxis]
        else:
            self.coefficient_matrix = coefficient_matrix

        if set_to_absolute:
            self.coefficient_matrix = np.abs(self.coefficient_matrix)

        self.molecule_graph = molecule_graph
        self.basis_info_raw = basis_info_raw

        self.parse_basis_data()
        self.get_basis_index()

    def get_coefficient_matrix(self):
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

    def parse_basis_data(self):
        """Parse the basis information from data from basissetexchange.org
        json format to a dict containing the number of s, p and d functions
        for each atom. The resulting dictionary, self.basis_info contains the
        total set of basis functions for each atom.
        """

        logger.debug(f"Parsing basis information from {self.basis_info_raw}")

        self.basis_info = {}

        for atom_number in self.basis_info_raw["elements"]:
            angular_momentum_all = []
            for basis_index, basis_functions in enumerate(
                self.basis_info_raw["elements"][atom_number]["electron_shells"]
            ):
                angular_momentum_all.extend(basis_functions["angular_momentum"])
            angular_momentum_all = [
                self.n_to_l_basis[element] for element in angular_momentum_all
            ]
            self.basis_info[int(atom_number)] = angular_momentum_all

    def _get_atomic_number(cls, symbol):
        """Get the atomic number of an element based on the symbol."""
        return ase_data.atomic_numbers[symbol]

    def get_basis_index(self):
        """Get the basis index for each atom in the atom in the molecule."""

        basis_idx_s = []
        basis_idx_p = []
        basis_idx_d = []
        basis_atom = []
        irreps_all_atom = []

        atom_basis_counter = 0

        for atom in self.molecule_graph.molecule:

            # Store the irreps
            irreps_atom = ""

            # Get the basis functions for this atom
            atomic_number = self._get_atomic_number(atom.species_string)
            basis_functions = self.basis_info[atomic_number]

            # Get the atom specific list
            basis_s_ = []
            basis_p_ = []
            basis_d_ = []

            # Store the initial basis functions
            counter_tot_basis_idx = atom_basis_counter

            for basis_function in basis_functions:
                if basis_function == "s":
                    range_idx = list(range(atom_basis_counter, atom_basis_counter + 1))
                    basis_s_.append(range_idx)
                    atom_basis_counter += 1
                    irreps_atom += "+1x0e"
                elif basis_function == "p":
                    range_idx = list(range(atom_basis_counter, atom_basis_counter + 3))
                    basis_p_.append(range_idx)
                    atom_basis_counter += 3
                    irreps_atom += "+1x1o"
                elif basis_function == "d":
                    range_idx = list(range(atom_basis_counter, atom_basis_counter + 6))
                    basis_d_.append(range_idx)
                    atom_basis_counter += 6
                    irreps_atom += "+1x0e+1x2e"

            irreps_atom = irreps_atom[1:]
            irreps_all_atom.append(irreps_atom)

            basis_atom.append(list(range(counter_tot_basis_idx, atom_basis_counter)))

            basis_idx_s.append(basis_s_)
            basis_idx_p.append(basis_p_)
            basis_idx_d.append(basis_d_)

        self.basis_idx_s = basis_idx_s
        self.basis_idx_p = basis_idx_p
        self.basis_idx_d = basis_idx_d
        self.basis_atom = basis_atom
        self.irreps_all_atom = irreps_all_atom

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
                    self.coefficient_matrix_atom[atom_idx]
                )


class ModifiedCoefficientMatrix(CoefficientMatrix):
    def __init__(
        self,
        molecule_graph: MoleculeGraph,
        basis_info_raw: Dict[str, Any],
        coefficient_matrix: npt.ArrayLike,
        store_idx_only: int = None,
        set_to_absolute: bool = False,
        max_s_functions: Union[str, int] = None,
        max_p_functions: Union[str, int] = None,
        max_d_functions: Union[str, int] = None,
        use_minimal_basis_node_features: bool = False,
        **kwargs,
    ):
        """Modify the coefficient matrix representing each atom by a fixed basis.
        
        Args:
            molecule_graph (MoleculeGraph): Molecule graph object.
            basis_info_raw (Dict[str, Any]): Dictionary containing the basis information.
            coefficient_matrix (npt.ArrayLike): Coefficient matrix.
            store_idx_only (int, optional): Store only the indices of the basis functions.
            set_to_absolute (bool, optional): Set the coefficients to absolute values.
            max_s_functions (Union[str, int], optional): Maximum number of s functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
            max_p_functions (Union[str, int], optional): Maximum number of p functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
            max_d_functions (Union[str, int], optional): Maximum number of d functions to use.\
                Use "all" to use all avail the basis functions [Not recommended].
            use_minimal_basis_node_features (bool, optional): Use minimal basis node features.
        """
        super().__init__(
            molecule_graph=molecule_graph,
            basis_info_raw=basis_info_raw,
            coefficient_matrix=coefficient_matrix,
            store_idx_only=store_idx_only,
            set_to_absolute=set_to_absolute,
            **kwargs,
        )

        self.max_s_functions = max_s_functions
        self.max_p_functions = max_p_functions
        self.max_d_functions = max_d_functions
        self.use_minimal_basis_node_features = use_minimal_basis_node_features

    def pad_split_coefficient_matrix(self):
        raise NotImplementedError(
            "This method is not implemented for this class\
                                  as the coefficient matrix for each atom is already\
                                  represented by a fixed basis and hence has the same \
                                  dimensions."
        )

    def get_node_features(self):
        """Get the node features based on the specifications."""
        if self.use_minimal_basis_node_features:
            return self.get_minimal_basis_representation()
        else:
            return self.get_padded_representation()

    def get_minimal_basis_representation(self):
        """Return the minimal basis representation of the coefficient matrix."""
        self.separate_coeff_matrix_to_atom_centers()
        self.generate_minimal_basis_representation()
        return self.coefficient_matrix_minimal_basis

    def get_minimal_basis_representation_atom(self, atom_idx):
        """Return the minimal basis representation of the coefficient matrix for a given atom."""
        self.separate_coeff_matrix_to_atom_centers()
        self.generate_minimal_basis_representation()
        return self.coefficient_matrix_minimal_basis[atom_idx]

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
                max_s + 3 * max_p + 6 * max_d,
                self.coefficient_matrix.shape[1],
            ],
        )
        # Also create a boolean mask to indicate which basis functions are present
        self.basis_mask = np.zeros(
            [
                num_atoms,
                max_s + 3 * max_p + 6 * max_d,
            ],
        )

        for atom_idx, atom in enumerate(self.molecule_graph.molecule):

            _s_basis_idx = self.basis_idx_s[atom_idx]
            _s_basis_idx = np.array(_s_basis_idx)
            _s_basis_idx = _s_basis_idx.flatten()

            pad = max_s - len(_s_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of s functions is greater than the maximum number of s functions."
                )
            self.coefficient_matrix_padded[atom_idx, :max_s, :] = np.pad(
                self.coefficient_matrix[_s_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, :max_s] = np.pad(
                np.ones(len(_s_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )

            _p_basis_idx = self.basis_idx_p[atom_idx]
            _p_basis_idx = np.array(_p_basis_idx)
            _p_basis_idx = _p_basis_idx.flatten()

            if _p_basis_idx.size == 0:
                continue

            pad = 3 * max_p - len(_p_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of p functions is greater than the maximum number of p functions."
                )
            self.coefficient_matrix_padded[
                atom_idx, max_s : max_s + 3 * max_p, :
            ] = np.pad(
                self.coefficient_matrix[_p_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, max_s : max_s + 3 * max_p] = np.pad(
                np.ones(len(_p_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )

            _d_basis_idx = self.basis_idx_d[atom_idx]
            _d_basis_idx = np.array(_d_basis_idx)
            _d_basis_idx = _d_basis_idx.flatten()

            if _d_basis_idx.size == 0:
                continue

            pad = 6 * max_d - len(_d_basis_idx)
            if pad < 0:
                raise ValueError(
                    "The number of d functions is greater than the maximum number of d functions."
                )
            self.coefficient_matrix_padded[atom_idx, max_s + 3 * max_p :, :] = np.pad(
                self.coefficient_matrix[_d_basis_idx, :],
                ((0, pad), (0, 0)),
                "constant",
                constant_values=0,
            )
            self.basis_mask[atom_idx, max_s + 3 * max_p :] = np.pad(
                np.ones(len(_d_basis_idx)),
                (0, pad),
                "constant",
                constant_values=0,
            )

    def generate_minimal_basis_representation(self):
        """Create a minimal basis representation of the coefficient matrix.
        This representation is created by summing up the s, p components of
        each atom.
        """

        self.coefficient_matrix_minimal_basis = np.zeros(
            [
                len(self.molecule_graph.molecule),
                4,  # 1 s + 3 p
                self.coefficient_matrix.shape[1],
            ],
        )

        for atom_idx, atom in enumerate(self.molecule_graph.molecule):

            _s_basis_idx = self.basis_idx_s[atom_idx]
            _s_basis_idx = np.array(_s_basis_idx)
            _s_basis_idx = _s_basis_idx.flatten()
            _s_basis_idx = _s_basis_idx - self.basis_atom[atom_idx][0]

            _p_basis_idx = self.basis_idx_p[atom_idx]
            _p_basis_idx = np.array(_p_basis_idx)
            if _p_basis_idx.size == 0:
                _pfunctions_exist = False
            else:
                _pfunctions_exist = True
                _px_basis_idx, _py_basis_idx, _pz_basis_idx = _p_basis_idx.T
                _px_basis_idx = _px_basis_idx - self.basis_atom[atom_idx][0]
                _py_basis_idx = _py_basis_idx - self.basis_atom[atom_idx][0]
                _pz_basis_idx = _pz_basis_idx - self.basis_atom[atom_idx][0]

            _s_coeff = np.max(
                self.coefficient_matrix_atom[atom_idx][_s_basis_idx, :], axis=0
            )
            if _pfunctions_exist:
                _px_coeff = np.max(
                    self.coefficient_matrix_atom[atom_idx][_px_basis_idx, :], axis=0
                )
                _py_coeff = np.max(
                    self.coefficient_matrix_atom[atom_idx][_py_basis_idx, :], axis=0
                )
                _pz_coeff = np.max(
                    self.coefficient_matrix_atom[atom_idx][_pz_basis_idx, :], axis=0
                )
            else:
                _px_coeff = np.zeros(self.coefficient_matrix_atom[atom_idx].shape[1])
                _py_coeff = np.zeros(self.coefficient_matrix_atom[atom_idx].shape[1])
                _pz_coeff = np.zeros(self.coefficient_matrix_atom[atom_idx].shape[1])

            self.coefficient_matrix_minimal_basis[atom_idx] = np.vstack(
                (_s_coeff, _px_coeff, _py_coeff, _pz_coeff)
            )
