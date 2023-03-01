from typing import List, Dict, Union

import numpy.typing as npt
import numpy as np

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
        **kwargs
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
            **kwargs
        )


class MatrixSplitAtoms:
    def __init__(
        self,
        matrix: npt.ArrayLike,
        molecule_graph: MoleculeGraph,
        basis_info_atom: Dict,
        **kwargs
    ):
        """Split an atomic matrix into node and edge features."""

        atom_basis = [
            basis_info_atom[atom.species_string] for atom in molecule_graph.molecule
        ]
        atom_basis = np.cumsum(atom_basis)
        # Starting form 0, build the index as a range based on the end
        # of the basis
        atom_basis = np.insert(atom_basis, 0, 0)
        atom_basis = [
            list(range(atom_basis[i], atom_basis[i + 1]))
            for i in range(len(atom_basis) - 1)
        ]
        self.atom_basis = atom_basis

        # The dimensionality of the node and edge features wil be different
        # for now
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
