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
        else:
            tensor_pos = None
        if x is not None:
            tensor_x = {}
        else:
            tensor_x = None
        if edge_index is not None:
            tensor_edge_index = [[], []]
        else:
            tensor_edge_index = None
        if edge_attr is not None:
            tensor_edge_attr = {}
        else:
            tensor_edge_attr = None

        # This for loop assumes that the keys of pos, edge_index, and x are the same.
        # The sort function makes sure that the order of the molecules in the reaction
        # (from reactant to product) are always maintained.
        for state_index in sorted(pos):
            if pos[state_index] is not None:
                tensor_pos.extend(pos[state_index])

            if edge_index[state_index] is not None:
                edge_scr, edge_dest = edge_index[state_index]
                tensor_edge_index[0].extend(edge_scr)
                tensor_edge_index[1].extend(edge_dest)

            if x is not None:
                # Apply the same treatment to the x.
                tensor_x[state_index] = torch.as_tensor(x[state_index], dtype=DTYPE)

            if edge_attr is not None:
                # Apply the same treatment to the edge_attr.
                tensor_edge_attr[state_index] = torch.as_tensor(
                    edge_attr[state_index],
                    dtype=DTYPE,
                )

        if isinstance(y, torch.Tensor):
            tensor_y = y
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

        # Number of nodes the the total length of positions
        if tensor_pos is not None:
            tensor_pos = torch.as_tensor(tensor_pos, dtype=DTYPE)

        if tensor_pos is not None:
            num_nodes = tensor_pos.shape[0]
        else:
            num_nodes = 0

        super().__init__(
            num_nodes=num_nodes,
            pos=tensor_pos,
            edge_index=tensor_edge_index,
            x=tensor_x,
            edge_attr=tensor_edge_attr,
            y=tensor_y,
            **tensor_kwargs,
        )
