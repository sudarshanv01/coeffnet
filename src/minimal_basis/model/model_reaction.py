import torch
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace


class EquivariantConv(torch.nn.Module):
    def __init__(
        self,
        irreps_sh: str,
        num_basis: int,
        max_radius: float,
        hidden_layers: int,
        irreps_in: str = "1x0e+1x1e",
        irreps_out: str = "1x0e+1x1e",
    ) -> None:
        super().__init__()

        if isinstance(irreps_in, str):
            irreps_in = o3.Irreps(irreps_in)
        if isinstance(irreps_out, str):
            irreps_out = o3.Irreps(irreps_out)
        if isinstance(irreps_sh, str):
            irreps_sh = o3.Irreps(irreps_sh)

        self.irreps_sh = irreps_sh

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            shared_weights=False,
        )

        self.num_basis = num_basis
        self.max_radius = max_radius

        self.fc = e3nn.nn.FullyConnectedNet(
            [self.num_basis, hidden_layers, self.tp.weight_numel], torch.relu
        )

    def forward(self, f_1, edge_index, pos):
        """Forward pass of Equivariant convolution."""

        row, col = edge_index
        num_nodes = f_1.shape[0]
        num_neighbors = len(row) / num_nodes

        edge_vec = pos[row] - pos[col]
        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )

        edge_length_embedding = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis="smooth_finite",
            cutoff=True,
        )

        weights_from_embedding = self.fc(edge_length_embedding)

        f_output = self.tp(f_1[row], sh, weights_from_embedding)
        f_output = scatter(f_output, col, dim=0, dim_size=num_nodes).div(
            num_neighbors**0.5
        )

        return f_output


class ReactionModel(torch.nn.Module):
    def __init__(
        self,
        irreps_sh: str,
        num_basis: int,
        max_radius: float,
        hidden_layers: int,
        irreps_in: str = "1x0e+1x1e",
        irreps_out: str = "1x0e+1x1e",
    ) -> None:
        """Initialize the reaction model."""
        super().__init__()

        self.equivariant_conv_IS = EquivariantConv(
            irreps_sh,
            num_basis,
            max_radius,
            hidden_layers,
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

        self.equivariant_conv_FS = EquivariantConv(
            irreps_sh,
            num_basis,
            max_radius,
            hidden_layers,
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    def forward(self, data):
        """Forward pass of the reaction model."""
        f_IS = data.x
        f_FS = data.x_final_state

        edge_index_interpolated_TS = data.edge_index_interpolated_transition_state
        pos_TS = data.pos_transition_state

        f_IS = self.equivariant_conv_IS(f_IS, edge_index_interpolated_TS, pos_TS)
        f_FS = self.equivariant_conv_FS(f_FS, edge_index_interpolated_TS, pos_TS)

        f_TS = f_IS + f_FS

        return f_TS
