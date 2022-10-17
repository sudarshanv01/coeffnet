from typing import Dict, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear, ReLU, Softmax

from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing


class ChargeModel(MessagePassing):
    def __init__(self, out_channels: int):
        """Create a graph convolution neural network model
        to predict the activation barrier using the charges
        on the atoms."""
        super().__init__(aggr="add", node_dim=-1)

        # Construct the neural network
        # Note that for the first linear layer we have to
        # keep the in_channels as an input because the number of
        # channels is not fixed (different molecules can have different atoms).
        self.lin = lambda in_channels: nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, dataset):
        """Forward pass of the Neural Network."""

        # Get slices of data
        slices = dataset.slices

        # Get the data itself
        data = dataset.data

        # Splice up the different data sections to get the
        # different parts of the data.
        all_x = torch.tensor_split(data.x, slices["x"][1:-1])
        all_edge_index = torch.tensor_split(data.edge_index, slices["x"][1:-1], dim=-1)
        all_pos = torch.tensor_split(data.pos, slices["pos"][1:-1], dim=-1)

        # Loop over all the data points
        all_y_pred = torch.zeros(len(all_x))

        for i, (x, edge_index, pos) in enumerate(zip(all_x, all_edge_index, all_pos)):

            # Add self-loops to the graph.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index

            # Construct the first linear layer
            lin = self.lin(x.shape[0])

            # Linear transformation to the node features.
            x = lin(x)

            # Compute the degree.
            deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Propogate the message
            out = self.propagate(edge_index, x=x, norm=norm)

            # Apply a final bias layer
            out = out + self.bias

            # Sum up all the contributions to the activation barrier
            all_y_pred[i] = out.sum()

        return all_y_pred

    def message(self, x_j, norm):
        """Message function for the graph convolution."""
        return norm.view(-1, 1) * x_j
