import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch_geometric.nn import MetaLayer

from torch_scatter import scatter_mean, scatter

import e3nn
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet
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


class NodeEquiModel(torch.nn.Module):
    def __init__(self, irreps_out_per_basis, hidden_layers, num_basis):
        super().__init__()

        self.node_mlp = EquivariantConv(irreps_out_per_basis, hidden_layers, num_basis)

    def forward(self, x, edge_index, pos, max_radius, num_nodes):
        """Forward pass of the edge model."""

        out = self.node_mlp(
            f_in=x,
            edge_index=edge_index,
            pos=pos,
            max_radius=max_radius,
            num_nodes=num_nodes,
            target_dim=num_nodes,
        )
        return out


def generate_equi_rep_from_matrix(matrix):
    """For a minimal basis matrix, generate the 45 element vector representation."""

    # Split the components of the matrix
    s_comp = matrix[..., 0:1, 0:1]
    p_comp = matrix[..., 1:4, 1:4]
    d_comp = matrix[..., 4:, 4:]
    sp_comp = matrix[..., 0:1, 1:4]
    sd_comp = matrix[..., 0:1, 4:]
    pd_comp = matrix[..., 1:4, 4:]

    def create_voigt_notation_vector(component):
        """For a given component, create the voigt notation vector"""
        diagonal = torch.diagonal(component, dim1=-2, dim2=-1)
        indices_upper = torch.triu_indices(
            component.shape[-2], component.shape[-1], offset=1
        )
        off_diagonal = component[..., indices_upper[0], indices_upper[1]]
        return torch.cat([diagonal, off_diagonal], dim=-1)

    # Create the voigt notation vector for the s_component, p_component, d_component
    s_contrib = create_voigt_notation_vector(s_comp)
    p_contrib = create_voigt_notation_vector(p_comp)
    d_contrib = create_voigt_notation_vector(d_comp)

    sp_contrib = sp_comp.flatten(start_dim=-2)
    sd_contrib = sd_comp.flatten(start_dim=-2)
    pd_contrib = pd_comp.flatten(start_dim=-2)

    equi_rep = torch.cat(
        [s_contrib, p_contrib, d_contrib, sp_contrib, sd_contrib, pd_contrib], dim=-1
    )

    return equi_rep


class EquivariantConv(torch.nn.Module):

    minimal_basis_size = 9

    def __init__(self, irreps_in, irreps_out, hidden_layers) -> None:
        super().__init__()

        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_in,
            irreps_out=irreps_out,
        )

        if isinstance(irreps_in, str):
            irreps_in = e3nn.o3.Irreps(irreps_in)
        if isinstance(irreps_out, str):
            irreps_out = e3nn.o3.Irreps(irreps_out)

        self.fc = FullyConnectedNet([irreps_out.dim, hidden_layers, irreps_in.dim])

    def forward(self, f_nodes, f_edges, edge_index):
        """Forward pass of Equivariant convolution."""

        row, col = edge_index

        f_nodes_matrix = f_nodes.reshape(
            -1, 2, self.minimal_basis_size, self.minimal_basis_size
        )
        f_edges_matrix = f_edges.reshape(
            -1, 2, self.minimal_basis_size, self.minimal_basis_size
        )

        f_nodes_matrix = generate_equi_rep_from_matrix(f_nodes_matrix)
        f_nodes_matrix = f_nodes_matrix[row]

        f_edges_matrix = generate_equi_rep_from_matrix(f_edges_matrix)

        f_output = self.tp(f_nodes_matrix, f_edges_matrix)

        # Apply the fully connected network
        f_output = self.fc(f_output)

        return f_output


class SimpleHamiltonianModel(torch.nn.Module):
    def __init__(self, irreps_in, irreps_intermediate, hidden_layers) -> None:
        super().__init__()

        self.conv = EquivariantConv(
            irreps_in=irreps_in,
            irreps_out=irreps_intermediate,
            hidden_layers=hidden_layers,
        )

    def forward(self, data):
        """Forward pass of the Hamiltonian model."""

        # Parse data from the data object
        f_nodes_IS = data.x
        f_nodes_FS = data.x_final_state
        f_edges_IS = data.edge_attr
        f_edges_FS = data.edge_attr_final_state
        edge_index_IS = data.edge_index
        edge_index_FS = data.edge_index_final_state

        f_output_IS = self.conv(f_nodes_IS, f_edges_IS, edge_index_IS)
        f_output_FS = self.conv(f_nodes_FS, f_edges_FS, edge_index_FS)

        # Scatter the outputs to the nodes
        f_output_IS = scatter(
            f_output_IS, edge_index_IS[0], dim=0, dim_size=f_nodes_IS.size(0)
        )
        f_output_FS = scatter(
            f_output_FS, edge_index_FS[0], dim=0, dim_size=f_nodes_FS.size(0)
        )

        # Subtract the final state from the initial state
        f_output = f_output_IS - f_output_FS

        # Mean over all dimensions except the batch dimension
        f_output = f_output.mean(dim=tuple(range(1, f_output.dim())))

        # Scatter the output such that there is one output per graph
        f_output = scatter(f_output, data.batch, dim=0, reduce="mean")

        return f_output


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


class EdgeEquiModel(torch.nn.Module):
    def __init__(self, irreps_out_per_basis, hidden_layers, num_basis):
        super().__init__()

        self.edge_mlp = EquivariantConv(irreps_out_per_basis, hidden_layers, num_basis)

    def forward(self, edge_attr, edge_index, pos, max_radius, num_nodes):
        """Forward pass of the edge model."""

        out = self.edge_mlp(
            f_in=edge_attr,
            edge_index=edge_index,
            pos=pos,
            max_radius=max_radius,
            num_nodes=num_nodes,
            target_dim=edge_index.shape[1],
        )
        return out


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


class EquiGraph2GraphModel(torch.nn.Module):
    def __init__(
        self,
        irreps_out_per_basis,
        hidden_layers,
        num_basis,
        num_targets,
        num_updates,
        hidden_channels,
        num_global_features,
    ):
        super().__init__()

        # ---- Define the equivariant layers.
        self.equi_edge_model = EdgeEquiModel(
            irreps_out_per_basis, hidden_layers, num_basis
        )
        self.equi_node_model = NodeEquiModel(
            irreps_out_per_basis, hidden_layers, num_basis
        )
        num_features_tp = irreps_out_per_basis.dim * 3
        self.equi_global_model = GlobalModel(
            hidden_layers,
            num_global_features,
            num_features_tp,
            num_targets,
        )
        self.equi_meta_layer = MetaLayerEqui(
            self.equi_node_model,
            self.equi_edge_model,
            self.equi_global_model,
        )

        # ---- Define the invariant layers
        # Node features for the global model after the tensor product.
        self.edge_model = EdgeModel(
            hidden_channels=hidden_channels,
            num_node_features=num_features_tp,
            num_edge_features=num_features_tp,
            num_global_features=num_targets,
            num_targets=num_features_tp,
        )
        self.node_model = NodeModel(
            hidden_channels=hidden_channels,
            num_node_features=num_features_tp,
            num_edge_features=num_features_tp,
            num_global_features=num_targets,
            num_targets=num_features_tp,
        )
        self.global_model = GlobalModel(
            hidden_layers,
            num_targets,
            num_features_tp,
            num_targets,
        )
        self.meta_layer = MetaLayer(
            node_model=self.node_model,
            edge_model=self.edge_model,
            global_model=self.global_model,
        )

        self.num_updates = num_updates

    def forward(
        self, x_, edge_index, edge_attr_, u_, batch_, pos, max_radius, num_nodes
    ):
        # Perform a single GNN update for the equivariant network.
        x, edge_attr, u = self.equi_meta_layer(
            x_, edge_index, edge_attr_, u_, batch_, pos, max_radius, num_nodes
        )

        # Perform the invariant GNN updates.
        for i in range(self.num_updates):
            x, edge_attr, u = self.meta_layer(x, edge_index, edge_attr, u, batch_)

        return x, edge_attr, u


class MetaLayerEqui(torch.nn.Module):
    def __init__(self, node_model=None, edge_model=None, global_model=None):
        super().__init__()

        self.node_model = node_model
        self.edge_model = edge_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch, pos, max_radius, num_nodes):
        if self.node_model:
            x = self.node_model(
                x=x,
                edge_index=edge_index,
                pos=pos,
                max_radius=max_radius,
                num_nodes=num_nodes,
            )

        if self.edge_model:
            edge_attr = self.edge_model(
                edge_attr=edge_attr,
                edge_index=edge_index,
                pos=pos,
                max_radius=max_radius,
                num_nodes=num_nodes,
            )

        if self.global_model:
            u = self.global_model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u,
                batch=batch,
            )

        return x, edge_attr, u


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
    def __init__(self, global_features: int = 1):
        super().__init__()
        self.global_features = global_features

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
        averaged_node_features = scatter_mean(x, batch, dim=0)
        out = torch.cat([u, averaged_node_features], dim=1)
        # Take a mean of all the features of out
        out = torch.mean(out, dim=1)

        return out


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


class EquiHamiltonianModel(torch.nn.Module):
    def __init__(
        self,
        irreps_out_per_basis,
        hidden_layers,
        num_basis,
        num_targets,
        num_updates,
        hidden_channels,
        num_global_features,
        max_radius,
    ):
        super().__init__()

        self.graph2graph = EquiGraph2GraphModel(
            irreps_out_per_basis=irreps_out_per_basis,
            hidden_layers=hidden_layers,
            num_basis=num_basis,
            num_targets=num_targets,
            num_updates=num_updates,
            hidden_channels=hidden_channels,
            num_global_features=num_global_features,
        )

        self.graph2property = Graph2PropertyModel(global_features=num_global_features)

        self.max_radius = max_radius

    def forward(self, data):
        x, edge_index, edge_attr, u, batch, pos, num_nodes = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.global_attr,
            data.batch,
            data.pos,
            data.num_nodes,
        )

        u = u.view(-1, 1)

        x, edge_attr, u = self.graph2graph(
            x, edge_index, edge_attr, u, batch, pos, self.max_radius, num_nodes
        )

        out = self.graph2property(x, edge_index, edge_attr, u, batch)

        return out.view(len(out))
