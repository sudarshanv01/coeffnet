import logging

import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from minimal_basis.model.model_charges import Graph2GraphModel


class Graph2PropertyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_reactant, edge_index, edge_attr, u, u_reactant, batch):
        """Forward pass of the graph-to-property model. Takes in
        the attributes of the final graph and returns a single
        property prediction.
        Args:
            x: Node features of shape (num_nodes, num_node_features)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, num_edge_features)
            u: Global features of shape (num_graphs, num_global_features)
        """
        row, col = edge_index
        averaged_node_features = global_mean_pool(x, batch)
        averaged_edge_features = global_mean_pool(edge_attr, batch[row])
        out = torch.cat([u, averaged_node_features, averaged_edge_features], dim=1)

        averaged_node_features_reactant = global_mean_pool(x_reactant, batch)
        out_reactant = torch.cat(
            [u_reactant, averaged_node_features_reactant, averaged_edge_features], dim=1
        )

        # Take the difference between the two
        out = out - out_reactant
        return out


class MessagePassingDiffClassifierModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        hidden_channels: int,
        num_updates: int,
        **kwargs,
    ):

        self.logger = logging.getLogger(__name__)
        if kwargs.get("debug", False):
            self.logger.setLevel(logging.DEBUG)

        super().__init__()

        self.logger.debug("Creating model")
        self.logger.debug("Number of node features: {}".format(num_node_features))
        self.logger.debug("Number of edge features: {}".format(num_edge_features))
        self.logger.debug("Number of global features: {}".format(num_global_features))

        self.graph2graph = Graph2GraphModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            hidden_channels=hidden_channels,
            num_updates=num_updates,
        )
        self.graph2graph_reactants = Graph2GraphModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            hidden_channels=hidden_channels,
            num_updates=num_updates,
        )

        # Final linear layer
        self.lin = Linear(
            in_features=num_node_features + num_edge_features + num_global_features,
            out_features=1,
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
        # Reshape the tensors to the right dimensions
        u = u.view(-1, self.num_global_features)

        x_reactant, edge_index_reactant, edge_attr_reactant, u_reactant = (
            data.x_reactant,
            data.edge_index_reactant,
            data.edge_attr_reactant,
            data.global_attr_reactant,
        )
        u_reactant = u_reactant.view(-1, self.num_global_features)

        self.logger.debug("x shape: {}".format(x.shape))
        self.logger.debug("edge_attr shape: {}".format(edge_attr.shape))
        self.logger.debug("u shape: {}".format(u.shape))
        self.logger.debug("")
        self.logger.debug("x_reactant shape: {}".format(x_reactant.shape))
        self.logger.debug(
            "edge_attr_reactant shape: {}".format(edge_attr_reactant.shape)
        )
        self.logger.debug("")

        # Perform graph2graph separately on the reactant and transition state graphs
        self.logger.debug(f"u (reactant) before: {u_reactant}")
        x_reactant, edge_attr_reactant, u_reactant = self.graph2graph_reactants(
            x_reactant, edge_index_reactant, edge_attr_reactant, u_reactant, batch
        )
        self.logger.debug(f"u (reactant) after: {u_reactant}")
        self.logger.debug(f"u (transition state) before: {u}")
        x, edge_attr, u = self.graph2graph(x, edge_index, edge_attr, u, batch)
        self.logger.debug(f"u (transition state) after: {u}")

        out = self.graph2property(
            x, x_reactant, edge_index, edge_attr, u, u_reactant, batch
        )

        # Final linear layer
        out = self.lin(out)

        return out
