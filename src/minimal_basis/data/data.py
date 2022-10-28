from typing import List, Dict, Union

import numpy.typing as npt
import numpy as np

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
    """Base class for creating a reaction Data object.

    The main idea is to take the information from the elements
    from the reaction and store them in tensors, keeping track
    of which elements belong to which molecule and which state.

    Note that if a Dict is suppled to any input, it will sort
    the dictionary through its keys before converting it to a tensor.

    Parameters
    ----------
    pos : Dict[str, npt.ArrayLike]
        The positions of the atoms in a molecule. The keys represent
        the state index and the values are the positions of the atoms
        in the molecule in that state.
    edge_index : Dict[str, npt.ArrayLike]
        The index of the edge between two atoms.
    x : Dict[str, npt.ArrayLike]
        The features of the atoms in a molecule.
    edge_attr : Dict[str, npt.ArrayLike]
        The features of the edges between atoms.
    y : npt.ArrayLike
        The activation energy of the reaction.

    """

    def __init__(
        self,
        pos: Union[Dict[str, npt.ArrayLike], torch.Tensor] = None,
        edge_index: Union[Dict[str, npt.ArrayLike], torch.Tensor] = None,
        x: Dict[str, npt.ArrayLike] = None,
        edge_attr: Union[Dict[str, npt.ArrayLike], torch.Tensor] = None,
        y: npt.ArrayLike = None,
        **kwargs,
    ):

        if pos is not None:
            if isinstance(pos, dict):
                # Convert the positions to a tensor.
                tensor_pos = []
                for state_index in sorted(pos):
                    if pos[state_index] is not None:
                        tensor_pos.extend(pos[state_index])

                tensor_pos = torch.as_tensor(tensor_pos, dtype=DTYPE)
            elif isinstance(pos, torch.Tensor):
                tensor_pos = pos

            # The total number of nodes are the number of atoms in the
            # molecule for each state.
            num_nodes = tensor_pos.shape[0]
        else:
            # If there are no positions supplied, then there are no nodes
            # and here the number of nodes is set to zero.
            tensor_pos = None
            num_nodes = 0

        if x is not None:
            if isinstance(x, dict):
                # The features of the nodes are stored in a tensor.
                tensor_x = []
                for state_index in sorted(x):
                    if isinstance(x[state_index], np.ndarray):
                        tensor_x.extend(x[state_index].tolist())
                    elif isinstance(x[state_index], list):
                        tensor_x.extend(x[state_index])
                    elif isinstance(x[state_index], dict):
                        for feature in sorted(x[state_index]):
                            tensor_x.append(x[state_index][feature])
                    else:
                        raise TypeError(
                            f"The type of the feature {feature} is not supported."
                        )
                tensor_x = torch.as_tensor(tensor_x, dtype=DTYPE)
            elif isinstance(x, torch.Tensor):
                tensor_x = x
        else:
            # If there are no features supplied
            tensor_x = None

        if edge_index is not None:
            if isinstance(edge_index, dict):
                tensor_edge_index = [[], []]
                for state_index in sorted(edge_index):
                    if edge_index[state_index] is not None:
                        edge_scr, edge_dest = edge_index[state_index]
                        tensor_edge_index[0].extend(edge_scr)
                        tensor_edge_index[1].extend(edge_dest)
                # Convert tensor_edge_index to a tensor
                tensor_edge_index = torch.as_tensor(tensor_edge_index, dtype=DTYPE_INT)
            elif isinstance(edge_index, torch.Tensor):
                tensor_edge_index = edge_index
        else:
            tensor_edge_index = None

        if edge_attr is not None:
            if isinstance(edge_attr, dict):
                tensor_edge_attr = []
                for state_index in sorted(edge_attr):
                    # Apply the same treatment to the edge_attr.
                    if isinstance(edge_attr[state_index], np.ndarray):
                        tensor_edge_attr.extend(edge_attr[state_index].tolist())
                    elif isinstance(edge_attr[state_index], list):
                        tensor_edge_attr.append(edge_attr[state_index])
                    else:
                        raise TypeError(
                            f"The type of the edge_attr {edge_attr} is not supported."
                        )
                tensor_edge_attr = torch.as_tensor(tensor_edge_attr, dtype=DTYPE)
            elif isinstance(edge_attr, torch.Tensor):
                tensor_edge_attr = edge_attr
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

        super().__init__(
            num_nodes=num_nodes,
            pos=tensor_pos,
            edge_index=tensor_edge_index,
            x=tensor_x,
            edge_attr=tensor_edge_attr,
            y=tensor_y,
            **tensor_kwargs,
        )
