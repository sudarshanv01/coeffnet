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


class HamiltonianDataPoint(Data):
    """Store the Hamiltonian data for the initial_state and final_state."""

    def __init__(
        self,
        x: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        edge_index: Dict[str, Union[npt.ArrayLike, List[int]]] = None,
        edge_attr: Dict[str, Union[npt.ArrayLike, List[int]]] = None,
        pos: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        global_attr: Dict[str, Union[npt.ArrayLike, List[float]]] = None,
        y: Union[npt.ArrayLike, List[float]] = None,
        edge_index_interpolated_TS: Union[npt.ArrayLike, List[int]] = None,
        pos_interpolated_TS: Union[npt.ArrayLike, List[float]] = None,
        **kwargs,
    ):
        if pos is not None:
            pos_initial_state, pos_final_state = (
                pos["initial_state"],
                pos["final_state"],
            )
            pos_initial_state = convert_to_tensor(pos_initial_state)
            pos_final_state = convert_to_tensor(pos_final_state)
            pos_interpolated_TS = convert_to_tensor(pos_interpolated_TS)
        else:
            pos_initial_state = None
            pos_final_state = None
            pos_interpolated_TS = None

        if x is not None:
            x_initial_state, x_final_state = x["initial_state"], x["final_state"]
            x_initial_state = convert_to_tensor(x_initial_state)
            x_final_state = convert_to_tensor(x_final_state)
        else:
            x_initial_state = None
            x_final_state = None

        if edge_index is not None:
            edge_index_initial_state, edge_index_final_state = (
                edge_index["initial_state"],
                edge_index["final_state"],
            )
            edge_index_initial_state = convert_to_tensor(
                edge_index_initial_state, dtype=DTYPE_INT
            )
            edge_index_final_state = convert_to_tensor(
                edge_index_final_state, dtype=DTYPE_INT
            )
        else:
            edge_index_initial_state = None
            edge_index_final_state = None

        if edge_index_interpolated_TS is not None:
            edge_index_interpolated_TS = convert_to_tensor(
                edge_index_interpolated_TS, dtype=DTYPE_INT
            )

        if edge_attr is not None:
            edge_attr_initial_state, edge_attr_final_state = (
                edge_attr["initial_state"],
                edge_attr["final_state"],
            )
            edge_attr_initial_state = convert_to_tensor(edge_attr_initial_state)
            edge_attr_final_state = convert_to_tensor(edge_attr_final_state)
        else:
            edge_attr_initial_state = None
            edge_attr_final_state = None

        if global_attr is not None:
            global_attr_initial_state, global_attr_final_state = (
                global_attr["initial_state"],
                global_attr["final_state"],
            )
            global_attr_initial_state = convert_to_tensor(global_attr_initial_state)
            global_attr_final_state = convert_to_tensor(global_attr_final_state)
        else:
            global_attr_initial_state = None
            global_attr_final_state = None

        if y is not None:
            y = convert_to_tensor(y)

        super().__init__(
            x=x_initial_state,
            x_final_state=x_final_state,
            edge_index=edge_index_initial_state,
            edge_index_final_state=edge_index_final_state,
            edge_index_interpolated_TS=edge_index_interpolated_TS,
            edge_attr=edge_attr_initial_state,
            edge_attr_final_state=edge_attr_final_state,
            pos=pos_initial_state,
            pos_final_state=pos_final_state,
            global_attr=global_attr_initial_state,
            global_attr_final_state=global_attr_final_state,
            y=y,
            pos_interpolated_TS=pos_interpolated_TS,
            **kwargs,
        )


class CoefficientMatrixToAtoms:
    def __init__(
        self,
        molecule_graph: MoleculeGraph,
        basis_info_raw: Dict[str, Any],
        coefficient_matrix: npt.ArrayLike,
        store_idx_only: int = None,
        **kwargs,
    ):
        """Store the coefficient matrix and provides some utilities manipulate it."""
        if store_idx_only:
            self.coefficient_matrix = coefficient_matrix[:, store_idx_only]
            self.coefficient_matrix = self.coefficient_matrix[:, np.newaxis, :]
        else:
            self.coefficient_matrix = coefficient_matrix

        self.molecule_graph = molecule_graph
        self.basis_info_raw = basis_info_raw

    def parse_basis_data(self):
        """Parse the basis information from data from basissetexchange.org
        json format to a dict containing the number of s, p and d functions
        for each atom. The resulting dictionary, self.basis_info contains the
        total set of basis functions for each atom.
        """
        # Create a new dict with the basis information
        logger.info(f"Parsing basis information from {self.basis_file}")

        self.basis_info = {}

        for atom_number in self.basis_info_raw["elements"]:
            angular_momentum_all = []
            for basis_index, basis_functions in enumerate(
                self.basis_info_raw[atom_number]["electron_shells"]
            ):
                angular_momentum_all.extend(basis_functions["angular_momentum"])
            angular_momentum_all = [
                self.n_to_l_basis[element] for element in angular_momentum_all
            ]
            self.basis_info[int(atom_number)] = angular_momentum_all

    def _get_atomic_number(cls, symbol):
        """Get the atomic number of an element based on the symbol."""
        return ase_data.atomic_numbers[symbol]

    def get_basis_index(self, molecule):
        """Get the basis index for each atom in the atom in the molecule."""

        # Basis information
        basis_s = []
        basis_p = []
        basis_d = []

        # Store the grouping of the basis index of each atom in the molecule
        basis_atom = []
        irreps_all_atom = []

        tot_basis_idx = 0
        for atom_name in molecule.species:

            # Store the irreps
            irreps_atom = ""

            # Get the basis functions for this atom
            atomic_number = self._get_atomic_number(atom_name.symbol)
            basis_functions = self.basis_info[atomic_number]

            # Get the atom specific list
            basis_s_ = []
            basis_p_ = []
            basis_d_ = []

            # Store the initial basis functions
            counter_tot_basis_idx = tot_basis_idx

            for basis_function in basis_functions:
                if basis_function == "s":
                    range_idx = list(range(tot_basis_idx, tot_basis_idx + 1))
                    basis_s_.append(range_idx)
                    tot_basis_idx += 1
                    irreps_atom += "+1x0e"
                elif basis_function == "p":
                    range_idx = list(range(tot_basis_idx, tot_basis_idx + 3))
                    basis_p_.append(range_idx)
                    tot_basis_idx += 3
                    irreps_atom += "+1x1o"
                elif basis_function == "d":
                    range_idx = list(range(tot_basis_idx, tot_basis_idx + 5))
                    basis_d_.append(range_idx)
                    tot_basis_idx += 5
                    irreps_atom += "+1x2e"

            irreps_atom = irreps_atom[1:]
            irreps_all_atom.append(irreps_atom)

            # Store the basis index for this atom
            basis_atom.append(list(range(counter_tot_basis_idx, tot_basis_idx)))

            # Store the s,p and d basis functions for this atom
            basis_s.append(basis_s_)
            basis_p.append(basis_p_)
            basis_d.append(basis_d_)

        # Split the Hamilonian into node and edge features
        all_basis_idx = [basis_s, basis_p, basis_d]


class MatrixSplitAtoms:
    def __init__(
        self,
        matrix: npt.ArrayLike,
        molecule_graph: MoleculeGraph,
        basis_info_atom: Dict,
        **kwargs,
    ):
        """Split an atomic matrix into node and edge features."""

        atom_basis = [
            basis_info_atom[atom.species_string] for atom in molecule_graph.molecule
        ]
        atom_basis = np.cumsum(atom_basis)

        atom_basis = np.insert(atom_basis, 0, 0)
        atom_basis = [
            list(range(atom_basis[i], atom_basis[i + 1]))
            for i in range(len(atom_basis) - 1)
        ]
        self.atom_basis = atom_basis

        node_features = []
        edge_features = []

        for atom_idx, atom in enumerate(molecule_graph.molecule):
            _atom_basis = atom_basis[atom_idx]
            diagonal_block = matrix.take(_atom_basis, axis=1).take(_atom_basis, axis=2)
            node_features.append(diagonal_block)

        for (src, dst, _) in molecule_graph.graph.edges:
            _src_basis = atom_basis[src]
            _dst_basis = atom_basis[dst]
            off_diagonal_block = matrix.take(_src_basis, axis=1).take(
                _dst_basis, axis=2
            )
            edge_features.append(off_diagonal_block)

        self.node_features = node_features
        self.edge_features = edge_features
