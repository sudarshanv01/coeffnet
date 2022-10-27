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

    def forward(self, ek, vrk, vsk, u, batch):
        """Forward pass of the edge model.

        Args:
            ek: Edge features of shape (num_edges, num_edge_features)
            vrk: Edge source node features of shape (num_edges, num_node_features)
            vsk: Edge target node features of shape (num_edges, num_node_features)
            u: Global features of shape (num_graphs, num_global_features)
        """
        out = torch.cat([ek, vrk, vsk, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        num_targets: int,
    ):
        super().__init__()
        in_channels_1 = num_node_features + num_edge_features
        self.node_mlp_1 = Seq(
            Lin(in_channels_1, hidden_channels),
            ReLU(),
            Lin(hidden_channels, hidden_channels),
        )
        in_channels_2 = num_node_features + hidden_channels + num_global_features
        self.node_mlp_2 = Seq(
            Lin(in_channels_2, hidden_channels),
            ReLU(),
            Lin(hidden_channels, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """Forward pass of the node model.
        Args:
            x: Node features of shape (num_nodes, num_node_features)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, num_edge_features)
            u: Global features of shape (num_graphs, num_global_features)
        """

        # Perform the first aggregation step.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))

        # Perform node update.
        out = torch.cat([x, out, u[batch]], dim=1)

        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_global_features: int,
        num_node_features: int,
        num_targets: int,
    ):
        super().__init__()
        in_channels = num_global_features + num_node_features
        self.global_mlp = Seq(
            Lin(in_channels, hidden_channels),
            ReLU(),
            Lin(hidden_channels, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)
