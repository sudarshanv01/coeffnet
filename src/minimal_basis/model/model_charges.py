from typing import Dict, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing


class ChargeModel(MessagePassing):
    def __init__(self, out_channels: int):
        """Create a graph convolution neural network model
        to predict the activation barrier using the charges
        on the atoms."""
        super().__init__(aggr="add")

        # Construct the neural network
        self.lin = lambda in_channels: nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, dataset):
        """Forward pass of the Neural Network."""

        output = torch.zeros(dataset.len())

        for i, datapoint in enumerate(dataset.data):
            # Iterate over each graph.

            x = datapoint.x
            edge_index = datapoint.edge_index

            collated_x = torch.zeros(datapoint.num_nodes)
            overall_index = 0
            for j, react_index in enumerate(x):
                x_molecule = x[react_index]
                collated_x[
                    overall_index : overall_index + x_molecule.shape[0]
                ] = x_molecule
                overall_index += x_molecule.shape[0]

            collated_x = collated_x.view(-1, 1)

            # Add self-loops to the graph
            edge_index, _ = add_self_loops(edge_index, num_nodes=collated_x.size(1))

            # Linear transformation of the node features
            lin = self.lin(collated_x.size(1))
            collated_x = lin(collated_x)

            # Complete normalization
            row, col = edge_index
            deg = degree(col, collated_x.size(0), dtype=collated_x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Propagate the charges
            out = self.propagate(edge_index, x=collated_x, norm=norm)

            # Add the bias
            out += self.bias

            # Save the output
            output[i] = torch.abs(out.sum())

        return output

    def message(self, x_j, norm):
        """Message function for the graph convolution."""
        return norm.view(-1, 1) * x_j
