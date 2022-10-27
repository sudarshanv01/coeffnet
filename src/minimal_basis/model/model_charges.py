import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class EdgeModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        num_targets: int,
    ):
        super().__init__()
        in_channels = 2 * num_node_features + num_edge_features + num_global_features
        self.edge_mlp = Seq(
            Lin(in_channels, hidden_channels),
            ReLU(),
            Lin(hidden_channels, num_targets),
        )

    def forward(self, ek, vrk, vsk, u):
        """Forward pass of the edge model.

        Args:
            ek: Edge features of shape (num_edges, num_edge_features)
            vrk: Edge source node features of shape (num_edges, num_node_features)
            vsk: Edge target node features of shape (num_edges, num_node_features)
            u: Global features of shape (num_graphs, num_global_features)
        """
        out = torch.cat([ek, vrk, vsk, u], 1)
        return self.edge_mlp(out)
