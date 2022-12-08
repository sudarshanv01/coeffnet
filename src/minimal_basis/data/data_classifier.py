from typing import Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

from numpy import typing as nptying

from minimal_basis.data import InterpolateDiffDatapoint


class DiffClassifierDatapoint(InterpolateDiffDatapoint):
    def __init__(
        self,
        x: Dict[str, Union[OptTensor, nptying.ArrayLike]] = None,
        edge_index: Dict[str, Union[OptTensor, nptying.ArrayLike]] = None,
        edge_attr: Dict[str, Union[OptTensor, nptying.ArrayLike]] = None,
        pos: Dict[str, Union[OptTensor, nptying.ArrayLike]] = None,
        global_attr: Dict[str, Union[OptTensor, nptying.ArrayLike]] = None,
        y: OptTensor = None,
        **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, pos, global_attr, y, **kwargs)
