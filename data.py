"""Generate data for the TS prediction project."""
import torch
from torch_geometric.data import Data
from typing import List, Dict
import numpy as np
import numpy.typing as npt

class DataPoint(Data):
    """Base class for creating a reaction data point.
    
    Each of pos, edge_index and x correspond to a dictionary of arrays.
    The key of the dictionary corresponds to the ordering of the 
    molecular fragments in the reaction. If the sign of the key
    is negative, the fragment is a reactant. If the sign is positive,
    the fragment is a product. 
    TODO: Better way to specify the ordering?
    """

    def __init__(
        self,
        pos: Dict[str, npt.ArrayLike],
        edge_index: Dict[str, npt.ArrayLike],
        x: Dict[str, npt.ArrayLike],
        edge_attr: Dict[str, npt.ArrayLike],
        y: npt.ArrayLike, 
        **kwargs,
    ):

        # There are the matrices that will define the graph.
        tensor_pos = {}
        tensor_edge_index = {}
        tensor_x = {}
        tensor_edge_attr = {}
        num_nodes = 0 # Add up the number of nodes in the graph.

        # This for loop assumes that the keys of pos, edge_index, and x are the same.
        for state_index in pos:
            tensor_pos[state_index] = torch.as_tensor(pos[state_index], dtype=torch.float)
            assert tensor_pos[state_index].shape[1] == 3
            num_nodes += tensor_pos[state_index].shape[0]

            # Apply the same treatment to the edge_index.
            tensor_edge_index[state_index] = torch.as_tensor(edge_index[state_index], dtype=torch.int8)
            assert tensor_edge_index[state_index].shape[0] == 2

            # Apply the same treatment to the x.
            tensor_x[state_index] = torch.as_tensor(x[state_index], dtype=torch.float)

            # Apply the same treatment to the edge_attr.
            tensor_edge_attr[state_index] = torch.as_tensor(edge_attr[state_index], dtype=torch.float)

        if isinstance(y, torch.Tensor):
            tensor_y = y
        else:
            tensor_y = torch.as_tensor(y, dtype=torch.float)

        # convert kwargs to tensor
        # TODO: Is there a more robust way to check this?
        tensor_kwargs = {}
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(
                    f"Only accepts np.ndarray or torch.Tensor. kwarg `{k}` is of type "
                    f" `{type(v)}`."
                )
            tensor_kwargs[k] = v

        super().__init__(
            num_nodes = num_nodes,
            pos = tensor_pos,
            edge_index = tensor_edge_index,
            x = tensor_x,
            edge_attr = tensor_edge_attr,
            y = tensor_y,
            **tensor_kwargs,
        )