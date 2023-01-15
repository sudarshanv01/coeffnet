import torch

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class InnerInterpolateDatapoint(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        **kwargs
    ):
        """Data structure for the inner interpolate model."""
        super().__init__(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, **kwargs
        )
