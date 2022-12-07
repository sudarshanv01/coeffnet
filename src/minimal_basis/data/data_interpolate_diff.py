from typing import Dict, List, Optional, Tuple, Union

import torch

from numpy import typing as nptying

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class InterpolateDiffDatapoint(Data):
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
        """Stores the graph structure of both the reactant and transition states
        as well as the correspoding node and edge features."""

        # Make sure there are exactly two keys in each dictionary
        # one for the `reactant` and one for the `transition_state`
        if x is not None:
            assert x.keys() == {"reactant", "transition_state"}
            tensor_x_reactant, tensor_x = x["reactant"], x["transition_state"]
        else:
            tensor_x = None
            tensor_x_reactant = None
        if edge_index is not None:
            assert edge_index.keys() == {"reactant", "transition_state"}
            tensor_edge_index_reactant, tensor_edge_index = (
                edge_index["reactant"],
                edge_index["transition_state"],
            )
        else:
            tensor_edge_index = None
            tensor_edge_index_reactant = None
        if edge_attr is not None:
            assert edge_attr.keys() == {"reactant", "transition_state"}
            tensor_edge_attr_reactant, tensor_edge_attr = (
                edge_attr["reactant"],
                edge_attr["transition_state"],
            )
        else:
            tensor_edge_attr = None
            tensor_edge_attr_reactant = None
        if pos is not None:
            tensor_pos_reactant, tensor_pos = pos["reactant"], pos["transition_state"]
        else:
            tensor_pos = None
            tensor_pos_reactant = None
        if global_attr is not None:
            tensor_global_attr_reactant, tensor_global_attr = (
                global_attr["reactant"],
                global_attr["transition_state"],
            )
        else:
            tensor_global_attr = None
            tensor_global_attr_reactant = None

        super().__init__(
            x=tensor_x,
            edge_index=tensor_edge_index,
            edge_attr=tensor_edge_attr,
            pos=tensor_pos,
            global_attr=tensor_global_attr,
            y=y,
            x_reactant=tensor_x_reactant,
            edge_index_reactant=tensor_edge_index_reactant,
            edge_attr_reactant=tensor_edge_attr_reactant,
            pos_reactant=tensor_pos_reactant,
            global_attr_reactant=tensor_global_attr_reactant,
            **kwargs
        )
