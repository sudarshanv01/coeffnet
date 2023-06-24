from typing import List, Optional, Union

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Absolute(BaseTransform):
    r"""Sets the node features to be absolute values of the original node
    features.

    Args:
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
    """

    def __init__(
        self,
        cat: bool = True,
    ):
        self.cat = cat

    def __call__(
        self,
        data: Data,
    ) -> Data:

        for store in data.node_stores:

            if store.x is not None:
                store.x = torch.abs(store.x)
            if store.x_final_state is not None:
                store.x_final_state = torch.abs(store.x_final_state)
            if store.x_transition_state is not None:
                store.x_transition_state = torch.abs(store.x_transition_state)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value})"
