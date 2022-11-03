import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch_geometric.nn import MetaLayer

from torch_scatter import scatter_mean, scatter

import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace


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


class EquivariantNodeConv(torch.nn.Module):

    # The minimal basis representation will always
    # be 1s + 3p + 5d functions.
    minimal_basis_size = 9

    def __init__(self, irreps_out_per_basis, hidden_layers, num_basis):
        super().__init__()

        # Create the spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)

        # Create the tensor products needed for the equivariant convolution
        irreps_s = o3.Irreps("1x0e")
        irreps_p = o3.Irreps("3x1o")
        irreps_d = o3.Irreps("5x2e")

        self.tp_s = o3.FullyConnectedTensorProduct(
            irreps_s, self.irreps_sh, irreps_out_per_basis, shared_weights=False
        )
        self.tp_p = o3.FullyConnectedTensorProduct(
            irreps_p, self.irreps_sh, irreps_out_per_basis, shared_weights=False
        )
        self.tp_d = o3.FullyConnectedTensorProduct(
            irreps_d, self.irreps_sh, irreps_out_per_basis, shared_weights=False
        )

        # Create a fully connected layer
        self.fc_s = e3nn.nn.FullyConnectedNet(
            [num_basis, hidden_layers, self.tp_s.weight_numel], torch.relu
        )
        self.fc_p = e3nn.nn.FullyConnectedNet(
            [num_basis, hidden_layers, self.tp_p.weight_numel], torch.relu
        )
        self.fc_d = e3nn.nn.FullyConnectedNet(
            [num_basis, hidden_layers, self.tp_d.weight_numel], torch.relu
        )

        self.num_basis = num_basis

    def forward(self, f_in, edge_index, pos, max_radius, num_nodes):
        """Forward pass of Equivariant convolution."""

        row, col = edge_index

        # --- Create weights based on the distance between nodes ---
        # Create the edge vector based on the positions of the nodes
        edge_vec = pos[row] - pos[col]

        # Infer the number of neighbors for each node
        num_neighbors = len(row) / num_nodes

        # Start by creating the spherical harmonics
        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )

        # Embedding for the edge length
        edge_length_embedding = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start=0.0,
            end=max_radius,
            number=self.num_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        edge_length_embedding = edge_length_embedding.mul(self.num_basis**0.5)

        # Weights come from the fully connected layer
        weights_s = self.fc_s(edge_length_embedding)
        weights_p = self.fc_p(edge_length_embedding)
        weights_d = self.fc_d(edge_length_embedding)

        # --- Compute tensor products between the s, p and d channels ---
        # First reshape the input_tensor to the right shape, i.e.
        # (num_nodes, minimal_basis, minimal_basis, 2)
        f_in_matrix = f_in.reshape(
            num_nodes, self.minimal_basis_size, self.minimal_basis_size, 2
        )

        f_in_s = f_in_matrix[:, 0:1, 0:1, :]
        f_in_p = f_in_matrix[:, 1:4, 1:4, :]
        f_in_d = f_in_matrix[:, 4:9, 4:9, :]

        f_in_s_flatten = f_in_s.reshape(num_nodes, -1, 2)
        f_in_p_flatten = f_in_p.reshape(num_nodes, -1, 2)
        f_in_d_flatten = f_in_d.reshape(num_nodes, -1, 2)

        out_s = self.tp_s(f_in_s_flatten[..., 0][row], sh, weights_s)
        out_s += self.tp_s(f_in_s_flatten[..., 1][row], sh, weights_s)
        out_p = self.tp_p(f_in_p_flatten[..., 0][row], sh, weights_p)
        out_p += self.tp_p(f_in_p_flatten[..., 1][row], sh, weights_p)
        out_d = self.tp_d(f_in_d_flatten[..., 0][row], sh, weights_d)
        out_d += self.tp_d(f_in_d_flatten[..., 1][row], sh, weights_d)

        summand = torch.cat([out_s, out_p, out_d], dim=1)

        # # Get the output
        f_out = scatter(summand, col, dim=0, dim_size=num_nodes)

        f_out = f_out.div(num_neighbors**0.5)

        return f_out


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


class Graph2GraphModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        hidden_channels: int,
        num_updates: int,
    ):
        super().__init__()

        self.num_updates = num_updates

        # Define the model layers.
        self.edge_model = EdgeModel(
            hidden_channels=hidden_channels,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_targets=num_edge_features,
        )

        self.node_model = NodeModel(
            hidden_channels=hidden_channels,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_targets=num_node_features,
        )

        self.global_model = GlobalModel(
            hidden_channels=hidden_channels,
            num_global_features=num_global_features,
            num_node_features=num_node_features,
            num_targets=num_global_features,
        )

        # Define the model update function.
        self.meta_layer = MetaLayer(
            edge_model=self.edge_model,
            node_model=self.node_model,
            global_model=self.global_model,
        )

    def forward(self, x_, edge_index, edge_attr_, u_, batch_):
        # Perform a single GNN update.
        x, edge_attr, u = self.meta_layer(x_, edge_index, edge_attr_, u_, batch_)

        # Perform additional GNN updates if needed.
        for i in range(self.num_updates):
            x, edge_attr, u = self.meta_layer(x, edge_index, edge_attr, u, batch_)

        return x, edge_attr, u


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
        # Make a prediction based on the final graph.
        return u


class HamiltonianModel(torch.nn.Module):
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

        self.graph2property = Graph2PropertyModel()

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.global_attr,
            data.batch,
        )

        x = x.view(-1, data.num_node_features)
        edge_attr = edge_attr.view(-1, data.num_edge_features)
        u = u.view(-1, 1)

        x, edge_attr, u = self.graph2graph(x, edge_index, edge_attr, u, batch)

        out = self.graph2property(x, edge_index, edge_attr, u, batch)

        return out.view(len(out))
