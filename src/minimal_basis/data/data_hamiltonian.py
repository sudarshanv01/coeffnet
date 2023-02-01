from typing import List, Dict, Union

import numpy.typing as nptyping
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)


class HamiltonianDataPoint(Data):
    """Store the Hamiltonian data for the initial_state and final_state."""

    def __init__(
        self,
        x: Dict[str, OptTensor] = None,
        edge_index: Dict[str, OptTensor] = None,
        edge_attr: Dict[str, OptTensor] = None,
        pos: Dict[str, OptTensor] = None,
        global_attr: Dict[str, OptTensor] = None,
        y: OptTensor = None,
        **kwargs
    ):
        if pos is not None:
            pos_initial_state, pos_final_state = (
                pos["initial_state"],
                pos["final_state"],
            )
        else:
            pos_initial_state = None
            pos_final_state = None

        if x is not None:
            x_initial_state, x_final_state = x["initial_state"], x["final_state"]
        else:
            x_initial_state = None
            x_final_state = None

        if edge_index is not None:
            edge_index_initial_state, edge_index_final_state = (
                edge_index["initial_state"],
                edge_index["final_state"],
            )
        else:
            edge_index_initial_state = None
            edge_index_final_state = None

        if edge_attr is not None:
            edge_attr_initial_state, edge_attr_final_state = (
                edge_attr["initial_state"],
                edge_attr["final_state"],
            )
        else:
            edge_attr_initial_state = None
            edge_attr_final_state = None

        if global_attr is not None:
            global_attr_initial_state, global_attr_final_state = (
                global_attr["initial_state"],
                global_attr["final_state"],
            )
        else:
            global_attr_initial_state = None
            global_attr_final_state = None

        super().__init__(
            x=x_initial_state,
            x_final_state=x_final_state,
            edge_index=edge_index_initial_state,
            edge_index_final_state=edge_index_final_state,
            edge_attr=edge_attr_initial_state,
            edge_attr_final_state=edge_attr_final_state,
            pos=pos_initial_state,
            pos_final_state=pos_final_state,
            global_attr=global_attr_initial_state,
            global_attr_final_state=global_attr_final_state,
            y=y,
            **kwargs
        )
