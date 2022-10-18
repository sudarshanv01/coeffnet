from typing import Dict, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """Message passing graph convolutional network."""
        super().__init__(aggr="add", node_dim=-1)

        # Get the device from the kwargs
        self.device = kwargs.get("device", "cpu")

        # Construct the neural network
        # Note that for the first linear layer we have to
        # keep the in_channels as an input because the number of
        # channels is not fixed (different molecules can have different atoms).
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x, edge_index):
        """Forward pass of the Neural Network."""

        # Add self-loops to the graph.
        edge_index_selfloops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index_selfloops

        # Linear transformation to the node features.
        x = self.lin(x)

        # Compute the degree.
        deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propogate the message
        out = self.propagate(edge_index_selfloops, x=x, norm=norm)

        # Apply a final bias layer
        out = out + self.bias

        # Make out into the right dimension by summing
        out = out.mean(dim=0)

        return out

    def message(self, x_j, norm):
        """Message function for the graph convolution."""
        return norm.view(-1, 1) * x_j


class ChargeModel(nn.Module):
    def __init__(self, **kwargs):
        """Create a graph convolution neural network model
        to predict the activation barrier using the charges
        on the atoms."""
        super().__init__()

        # Check if a device is specified
        self.device = kwargs.get("device", "cpu")

        self.hidden_channel_1 = kwargs.get("hidden_channel_1", 64)
        self.hidden_channel_2 = kwargs.get("hidden_channel_2", 64)
        self.out_channels = kwargs.get("out_channels", 64)

        # Create a series of graph convolutional layers
        self.conv1 = lambda in_channels: GCNConv(
            in_channels, self.hidden_channel_1, device=self.device
        )
        self.conv2 = GCNConv(
            self.hidden_channel_1, self.hidden_channel_2, device=self.device
        )
        self.conv3 = GCNConv(
            self.hidden_channel_2, self.out_channels, device=self.device
        )

        # Transfer the model to the GPU
        self.conv2.to(self.device)
        self.conv3.to(self.device)

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

            # Move the data to the GPU
            edge_index = edge_index.to(self.device)

            # Get the number of atoms in the molecule
            num_atoms = x.size(0)

            conv1 = self.conv1(num_atoms)
            out_conv1 = conv1(x, edge_index)
            # Apply non-linearity
            out_conv1 = torch.sigmoid(out_conv1)
            # Apply dropout
            # out_conv1 = F.dropout(out_conv1, p=0.5, training=self.training)
            out_conv2 = self.conv2(out_conv1, edge_index)
            # Apply non-linearity
            out_conv2 = torch.sigmoid(out_conv2)

            # Linear layer to get the output
            out_conv3 = self.conv3(out_conv2, edge_index)

            # Get the predicted activation barrier
            y_pred = out_conv3.mean(dim=0)

            all_y_pred[i] = y_pred

        return all_y_pred
