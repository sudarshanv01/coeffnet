from typing import List, Dict, Union

import numpy.typing as nptyping
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)


class HamiltonianDataPoint(Data):
    """Store the Hamiltonian data for the reactant and product."""

    def __init__(
        self,
        x: Dict[str, OptTensor] = None,
        edge_index: Dict[str, OptTensor] = None,
        edge_attr: Dict[str, OptTensor] = None,
        pos: Dict[str, OptTensor] = None,
        global_attr: Dict[str, OptTensor] = None,
        y: OptTensor = None,
        **kwargs
    ):

        if pos is not None:
            pos_reactant, pos_product = pos["reactant"], pos["product"]
        else:
            pos_reactant = None
            pos_product = None

        if x is not None:
            x_reactant, x_product = x["reactant"], x["product"]
        else:
            x_reactant = None

        if edge_index is not None:
            edge_index_reactant, edge_index_product = (
                edge_index["reactant"],
                edge_index["product"],
            )
        else:
            edge_index_reactant = None
            edge_index_product = None

        if edge_attr is not None:
            edge_attr_reactant, edge_attr_product = (
                edge_attr["reactant"],
                edge_attr["product"],
            )
        else:
            edge_attr_reactant = None
            edge_attr_product = None

        if global_attr is not None:
            global_attr_reactant, global_attr_product = (
                global_attr["reactant"],
                global_attr["product"],
            )
        else:
            global_attr_reactant = None
            global_attr_product = None

        super().__init__(
            x=x_reactant,
            x_product=x_product,
            edge_index=edge_index_reactant,
            edge_index_product=edge_index_product,
            edge_attr=edge_attr_reactant,
            edge_attr_product=edge_attr_product,
            pos=pos_reactant,
            pos_product=pos_product,
            global_attr=global_attr_reactant,
            global_attr_product=global_attr_product,
            y=y,
            **kwargs
        )
