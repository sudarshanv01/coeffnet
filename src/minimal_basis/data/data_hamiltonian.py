from typing import List, Dict, Union

import numpy.typing as npt
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
