"""Generate data for the TS prediction project."""
import torch
from torch_geometric.data import Data
from typing import List, Dict
import numpy as np
import numpy.typing as npt

class DataPoint(Data):
    """Base class for creating a molecular graph data point."""

    def __init__(
        self,
        pos: List[npt.ArrayLike] = None,
        edge_index: npt.ArrayLike = None,
        x: Dict[str, npt.ArrayLike] = None,
        y: Dict[str, npt.ArrayLike] = None,
        **kwargs,
    ):

        # convert to tensors
        if pos is not None:
            pos = torch.as_tensor(pos, dtype=torch.float32)
            assert pos.shape[1] == 3
            num_nodes = pos.shape[0]
        else:
            num_nodes = None

        if edge_index is not None:
            edge_index = torch.as_tensor(edge_index, dtype=torch.int8)
            assert edge_index.shape[0] == 2
            num_edges = edge_index.shape[1]
        else:
            num_edges = None
        
        # convert input and output to tensors
        if isinstance(x, dict): 
            tensor_x = {}
            for k, v in x.items():
                # v = self._convert_to_tensor(v)
                if v is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of x is of "
                        f"type `{type(v)}`."
                    )
                tensor_x[k] = v
            # self._check_tensor_dict(tensor_x, dict_name="x")
        elif isinstance(x, torch.Tensor):
            tensor_x = x
        else:
            tensor_x = None

        if isinstance(y, dict):
            tensor_y = {}
            for k, v in y.items():
                v = self._convert_to_tensor(v)
                if v is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of y is of "
                        f"type `{type(v)}`."
                    )
                tensor_y[k] = v
            self._check_tensor_dict(tensor_y, dict_name="y")
        elif isinstance(y, torch.Tensor):
            tensor_y = y
        else:
            tensor_y = None

        # convert kwargs to tensor
        tensor_kwargs = {}
        for k, v in kwargs.items():
            # v = self._convert_to_tensor(v)
            if v is None:
                raise ValueError(
                    f"Only accepts np.ndarray or torch.Tensor. kwarg `{k}` is of type "
                    f" `{type(v)}`."
                )
            tensor_kwargs[k] = v
        # self._check_tensor_dict(tensor_kwargs, dict_name="kwargs")

        super().__init__(
            pos=pos,
            edge_index=edge_index,
            num_nodes=num_nodes,
            x=tensor_x,
            y=tensor_y,
            **tensor_kwargs,
        )