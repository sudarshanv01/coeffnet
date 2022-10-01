from typing import List, Dict

import numpy.typing as npt

import torch
from torch_geometric.data import Data

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)


class DataPoint(Data):
    """Base class for creating a reaction data point. This class
    is common to any scheme where the reaction is desired to be the
    datapoint. Because it is a reaction we are describing, every
    input is a dictionary.

    The key of the dictionary corresponds to the ordering of the
    molecular fragments in the reaction. If the sign of the key
    is negative, the fragment is a reactant. If the sign is positive,
    the fragment is a product.
    """

    def __init__(
        self,
        pos: Dict[str, npt.ArrayLike] = None,
        edge_index: Dict[str, npt.ArrayLike] = None,
        x: Dict[str, npt.ArrayLike] = None,
        edge_attr: Dict[str, npt.ArrayLike] = None,
        y: npt.ArrayLike = None,
        **kwargs,
    ):

        if pos is not None:
            tensor_pos = []
            for state_index in sorted(pos):
                if pos[state_index] is not None:
                    tensor_pos.extend(pos[state_index])
            tensor_pos = torch.as_tensor(tensor_pos, dtype=DTYPE)
            num_nodes = tensor_pos.shape[0]
        else:
            tensor_pos = None
            num_nodes = 0

        if x is not None:
            tensor_x = {}
            for state_index in sorted(x):
                tensor_x[state_index] = torch.as_tensor(x[state_index], dtype=DTYPE)
        else:
            tensor_x = None

        if edge_index is not None:
            tensor_edge_index = [[], []]
            for state_index in sorted(edge_index):
                if edge_index[state_index] is not None:
                    edge_scr, edge_dest = edge_index[state_index]
                    tensor_edge_index[0].extend(edge_scr)
                    tensor_edge_index[1].extend(edge_dest)
        else:
            tensor_edge_index = None

        if edge_attr is not None:
            tensor_edge_attr = {}
            for state_index in sorted(edge_attr):
                # Apply the same treatment to the edge_attr.
                tensor_edge_attr[state_index] = torch.as_tensor(
                    edge_attr[state_index],
                    dtype=DTYPE,
                )
        else:
            tensor_edge_attr = None

        if isinstance(y, torch.Tensor):
            tensor_y = y
        elif y == None:
            tensor_y = None
        else:
            tensor_y = torch.as_tensor(y, dtype=DTYPE)

        # convert kwargs to tensor
        tensor_kwargs = {}
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(
                    f"Only accepts np.ndarray or torch.Tensor. kwarg `{k}` is of type "
                    f" `{type(v)}`."
                )
            tensor_kwargs[k] = v

        # Convert tensor_edge_index to a tensor
        if tensor_edge_index is not None:
            tensor_edge_index = torch.as_tensor(tensor_edge_index, dtype=DTYPE_INT)

        super().__init__(
            num_nodes=num_nodes,
            pos=tensor_pos,
            edge_index=tensor_edge_index,
            x=tensor_x,
            edge_attr=tensor_edge_attr,
            y=tensor_y,
            **tensor_kwargs,
        )
