import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from minimal_basis.model.model_charges import Graph2GraphModel


class Graph2PropertyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr, u, batch):
        """Forward pass of the graph-to-property model. Takes in
        the attributes of the final graph and returns a single
        property prediction.
        Args:
            x: Node features of shape (num_nodes, num_node_features)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, num_edge_features)
            u: Global features of shape (num_graphs, num_global_features)
        """
        # Add the sum of the node features to the global features.
        averaged_node_features = global_mean_pool(x, batch)
        out = torch.cat([u, averaged_node_features], dim=1)
        # Sum the global features and the averaged node features
        out = out.sum(dim=1)
        return out


class MessagePassingInnerInterpolateModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        hidden_channels: int,
        num_updates: int,
    ):
        super().__init__()

        self.graph2graph = Graph2GraphModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            hidden_channels=hidden_channels,
            num_updates=num_updates,
        )

        # Store the number of features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features

        self.graph2property = Graph2PropertyModel()

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.global_attr,
            data.batch,
        )

        x = x.view(-1, self.num_node_features)
        edge_attr = edge_attr.view(-1, self.num_edge_features)
        u = u.view(-1, self.num_global_features)

        x, edge_attr, u = self.graph2graph(x, edge_index, edge_attr, u, batch)

        out = self.graph2property(x, edge_index, edge_attr, u, batch)

        return out
