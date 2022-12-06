import torch

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class DatapointClassifier(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        **kwargs
    ):
        if y is not None:
            if y == "high":
                classify_y = torch.tensor([0], dtype=torch.long)
            elif y == "low":
                classify_y = torch.tensor([1], dtype=torch.long)
            else:
                raise ValueError("y must be 'high' or 'low'.")
        else:
            classify_y = None

        super().__init__(x, edge_index, edge_attr, classify_y, pos, **kwargs)
