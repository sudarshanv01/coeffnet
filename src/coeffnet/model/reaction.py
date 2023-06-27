from typing import Union, Dict, Optional

import logging

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from torch_cluster import radius_graph

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2102 import Network as GateNetwork
from e3nn.nn.models.gate_points_2102 import smooth_cutoff
from e3nn.nn.models.v2106.gate_points_networks import MessagePassing

from .equiformer.graph_attention import GraphAttention

logger = logging.getLogger(__name__)


def normalize_to_sum_squares_one(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Normalize the output such that the sum of squares of each graph is 1."""

    sum_squares_output = torch.sum(x**2, dim=1)
    sum_squares_graph = scatter(sum_squares_output, batch, dim=0, reduce="sum")
    normalization_factor = torch.sqrt(sum_squares_graph)
    x = x / normalization_factor[batch].unsqueeze(1)

    return x


class GateNetworkWithCustomEdges(GateNetwork):
    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        num_nodes,
        reduce_output=True,
    ) -> None:
        """Modify the GateNetwork class to use custom edges."""
        super().__init__(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            num_nodes=num_nodes,
            reduce_output=reduce_output,
        )

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
            - ``edge_index`` the edge index
        """
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_index = data["edge_index"]
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        scalar_z = self.ext_z(z)
        edge_features = torch.cat(
            [edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1
        )

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_features)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x


class GateReactionModel(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_node_attr: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        max_radius: float,
        num_neighbors: int,
        typical_number_of_nodes: int,
        mul: int,
        layers: Optional[int] = 3,
        number_of_basis: Optional[int] = 10,
        radial_layers: Optional[int] = 3,
        radial_neurons: Optional[int] = 128,
        reduce_output: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_output_to_initial_state: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Torch module for transition state properties prediction."""
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_node_attr = irreps_node_attr
        self.irreps_out = irreps_out
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_nodes = typical_number_of_nodes
        self.mul = mul
        self.layers = layers
        self.number_of_basis = number_of_basis
        self.radial_layers = radial_layers
        self.radial_neurons = radial_neurons

        self.reduce_output = reduce_output
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_output_to_initial_state = reference_output_to_initial_state

        if isinstance(irreps_in, str):
            self.irreps_in = o3.Irreps(irreps_in)
        if isinstance(irreps_node_attr, str):
            self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        if isinstance(irreps_out, str):
            self.irreps_out = o3.Irreps(irreps_out)

        irreps_ls = self.irreps_in.ls
        irreps_ls = list(set(irreps_ls))
        irreps_ls.sort()
        lp_irreps_hidden = [(l, (-1) ** l) for l in irreps_ls]
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l, p)) for l, p in lp_irreps_hidden]
        )
        logger.info(f"irreps_hidden: {self.irreps_hidden}")

        self.irreps_edge_attr = o3.Irreps(f"{self.number_of_basis}x0e")
        logger.info(f"irreps_edge_attr: {self.irreps_edge_attr}")

        self.network_initial_state = GateNetwork(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            max_radius=max_radius,
            number_of_basis=self.number_of_basis,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=False,
        )

        self.network_final_state = GateNetwork(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            max_radius=max_radius,
            number_of_basis=self.number_of_basis,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=False,
        )

        self.network_interpolated_transition_state = GateNetwork(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            max_radius=max_radius,
            number_of_basis=self.number_of_basis,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=False,
        )

    def forward(self, data):
        """Forward pass of the reaction model."""

        kwargs_initial_state = {
            "pos": data.pos,
            "x": data.x,
            "batch": data.batch,
            "edge_index": data.edge_index,
        }
        if "node_attr" in data:
            kwargs_initial_state["z"] = data.node_attr

        output_network_initial_state = self.network_initial_state(kwargs_initial_state)

        if self.mask_extra_basis:
            output_network_initial_state = (
                output_network_initial_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_initial_state = normalize_to_sum_squares_one(
                output_network_initial_state, data.batch
            )

        kwargs_final_state = {
            "pos": data.pos_final_state,
            "x": data.x_final_state,
            "batch": data.batch,
            "edge_index": data.edge_index_final_state,
        }
        if "node_attr" in data:
            kwargs_final_state["z"] = data.node_attr

        output_network_final_state = self.network_final_state(kwargs_final_state)

        if self.mask_extra_basis:
            output_network_final_state = output_network_final_state * data.basis_mask
        if self.normalize_sumsq:
            output_network_final_state = normalize_to_sum_squares_one(
                output_network_final_state, data.batch
            )

        p = data.p[0]
        p_prime = 1 - p
        x_interpolated_transition_state = (
            p_prime * output_network_initial_state + p * output_network_final_state
        )

        kwargs_interpolated_transition_state = {
            "pos": data.pos_interpolated_transition_state,
            "x": x_interpolated_transition_state,
            "batch": data.batch,
            "edge_index": data.edge_index_interpolated_transition_state,
        }
        if "node_attr" in data:
            kwargs_interpolated_transition_state["z"] = data.node_attr

        output_network_interpolated_transition_state = (
            self.network_interpolated_transition_state(
                kwargs_interpolated_transition_state
            )
        )

        if self.mask_extra_basis:
            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_interpolated_transition_state = normalize_to_sum_squares_one(
                output_network_interpolated_transition_state, data.batch
            )
        if self.reduce_output:
            output_network_interpolated_transition_state = scatter(
                output_network_interpolated_transition_state, data.batch, dim=0
            ).div(self.num_nodes**0.5)

            if self.reference_output_to_initial_state:
                output_network_initial_state = scatter(
                    output_network_initial_state, data.batch, dim=0
                ).div(self.num_nodes**0.5)

                output_network_interpolated_transition_state -= (
                    output_network_initial_state
                )

            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state.mean(dim=1)
            )
        else:
            if self.reference_output_to_initial_state:
                output_network_interpolated_transition_state -= (
                    output_network_initial_state
                )

        return output_network_interpolated_transition_state


class NetworkForAGraphWithNodeAttributes(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input: Union[str, o3.Irreps],
        irreps_node_attr: Union[str, o3.Irreps],
        irreps_node_output: Union[str, o3.Irreps],
        max_radius: float,
        num_neighbors: int,
        num_nodes: int,
        mul: int = 50,
        layers: int = 3,
        lmax: int = 2,
        pool_nodes: bool = True,
        number_of_basis: int = 10,
        radial_neurons: int = 64,
    ) -> None:
        """Modified from e3nn/nn/models/2106/gate_points_network.py to include _only_ node attributes.

        Args:
            irreps_node_input: Irreps of the input node features.
            irreps_node_attr: Irreps of the node attributes.
            irreps_node_output: Irreps of the output node features.
            max_radius: Maximum radius of the neighborhoods.
            num_neighbors: Number of neighbors to consider.
            num_nodes: Number of nodes in the graph.
            mul: Multiplicity to decide on hidden layer irreps.
            layers: Number of layers.
            lmax: Maximum degree of the spherical harmonics.
            pool_nodes: Whether to pool the node features.
            num_of_basis: Number of basis functions.
        """

        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_ls = irreps_node_input.ls
        irreps_ls = list(set(irreps_ls))
        irreps_ls.sort()
        lp_irreps_hidden = [(l, (-1) ** l) for l in irreps_ls]
        self.irreps_hidden = o3.Irreps([(mul, (l, p)) for l, p in lp_irreps_hidden])
        logger.info(f"irreps_hidden: {self.irreps_hidden}")

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input]
            + layers * [self.irreps_hidden]
            + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, radial_neurons],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        if "x" in data:
            node_input = data["x"]
        else:
            node_input = data["node_input"]

        if "node_attr" in data:
            node_attr = data["node_attr"]
        else:
            node_attr = node_input.new_ones(node_input.shape[0], 1)

        return batch, node_input, node_attr, edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_input, node_attr, edge_src, edge_dst, edge_vec = self.preprocess(
            data
        )
        del data

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )

        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",
            cutoff=True,
        ).mul(self.number_of_basis**0.5)

        node_outputs = self.mp(
            node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
        )

        if self.pool_nodes:
            return scatter(node_outputs, batch, int(batch.max()) + 1).div(
                self.num_nodes**0.5
            )
        else:
            return node_outputs


class MessagePassingReactionModel(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_node_attr: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        max_radius: float,
        num_neighbors: int,
        typical_number_of_nodes: int,
        mul: int,
        layers: int,
        lmax: int,
        reduce_output: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_reduced_output_to_initial_state: Optional[bool] = False,
        num_basis: Optional[int] = 10,
        radial_neurons: Optional[int] = 64,
    ) -> None:
        """Reaction model based on message passing."""

        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_node_attr = irreps_node_attr
        self.irreps_out = irreps_out
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_nodes = typical_number_of_nodes
        self.mul = mul
        self.layers = layers
        self.lmax = lmax
        self.number_of_basis = num_basis

        self.reduce_output = reduce_output
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )

        if isinstance(self.irreps_in, str):
            self.irreps_in = o3.Irreps(self.irreps_in)
        if isinstance(self.irreps_node_attr, str):
            self.irreps_node_attr = o3.Irreps(self.irreps_node_attr)
        if isinstance(self.irreps_out, str):
            self.irreps_out = o3.Irreps(self.irreps_out)

        self.network_initial_state = NetworkForAGraphWithNodeAttributes(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_in,
            max_radius=self.max_radius,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            mul=self.mul,
            layers=self.layers,
            number_of_basis=self.number_of_basis,
            lmax=self.lmax,
            radial_neurons=radial_neurons,
            pool_nodes=False,
        )

        self.network_final_state = NetworkForAGraphWithNodeAttributes(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_in,
            max_radius=self.max_radius,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            mul=self.mul,
            layers=self.layers,
            number_of_basis=self.number_of_basis,
            lmax=self.lmax,
            radial_neurons=radial_neurons,
            pool_nodes=False,
        )

        self.network_interpolated_transition_state = NetworkForAGraphWithNodeAttributes(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_out,
            max_radius=self.max_radius,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            mul=self.mul,
            layers=self.layers,
            number_of_basis=self.number_of_basis,
            lmax=self.lmax,
            radial_neurons=radial_neurons,
            pool_nodes=False,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model."""

        kwargs_initial_state = {
            "batch": data.batch,
            "pos": data.pos,
            "edge_index": data.edge_index,
            "x": data.x,
        }
        if "node_attr" in data:
            kwargs_initial_state["node_attr"] = data.node_attr

        kwargs_final_state = {
            "batch": data.batch,
            "pos": data.pos_final_state,
            "edge_index": data.edge_index_final_state,
            "x": data.x_final_state,
        }
        if "node_attr" in data:
            kwargs_final_state["node_attr"] = data.node_attr

        output_network_initial_state = self.network_initial_state(kwargs_initial_state)

        if self.mask_extra_basis:
            output_network_initial_state = (
                output_network_initial_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_initial_state = normalize_to_sum_squares_one(
                output_network_initial_state, data.batch
            )

        output_network_final_state = self.network_final_state(kwargs_final_state)

        if self.mask_extra_basis:
            output_network_final_state = output_network_final_state * data.basis_mask
        if self.normalize_sumsq:
            output_network_final_state = normalize_to_sum_squares_one(
                output_network_final_state, data.batch
            )

        p = data.p[0]
        p_prime = 1 - p
        x_interpolated_transition_state = (
            p_prime * output_network_initial_state + p * output_network_final_state
        )

        kwargs_interpolated_transition_state = {
            "batch": data.batch,
            "pos": data.pos_interpolated_transition_state,
            "edge_index": data.edge_index_interpolated_transition_state,
            "x": x_interpolated_transition_state,
        }
        if "node_attr" in data:
            kwargs_interpolated_transition_state["node_attr"] = data.node_attr

        output_network_interpolated_transition_state = (
            self.network_interpolated_transition_state(
                kwargs_interpolated_transition_state
            )
        )

        if self.mask_extra_basis:
            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_interpolated_transition_state = normalize_to_sum_squares_one(
                output_network_interpolated_transition_state, data.batch
            )
        if self.reduce_output:
            output_network_interpolated_transition_state = scatter(
                output_network_interpolated_transition_state, data.batch, dim=0
            ).div(self.num_nodes**0.5)

            if self.reference_reduced_output_to_initial_state:

                output_network_initial_state = scatter(
                    output_network_initial_state, data.batch, dim=0
                ).div(self.num_nodes**0.5)

                output_network_interpolated_transition_state -= (
                    output_network_initial_state
                )

        return output_network_interpolated_transition_state


class GraphAttentionReactionModel(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input: Union[str, o3.Irreps],
        irreps_node_attr: Union[str, o3.Irreps],
        irreps_edge_attr: Union[str, o3.Irreps],
        irreps_node_output: Union[str, o3.Irreps],
        fc_neurons: int,
        irreps_head: Union[str, o3.Irreps],
        num_heads: int,
        irreps_pre_attn: Optional[Union[str, o3.Irreps]] = None,
        rescale_degree: bool = False,
        nonlinear_message: bool = False,
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        reduce_output: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_reduced_output_to_initial_state: Optional[bool] = False,
        typical_number_of_nodes: Optional[int] = None,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_node_output)
        self.irreps_head = o3.Irreps(irreps_head)
        self.irreps_pre_attn = (
            o3.Irreps(irreps_pre_attn) if irreps_pre_attn is not None else None
        )
        self.num_heads = num_heads
        self.reduce_output = reduce_output
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )

        if typical_number_of_nodes:
            self.num_nodes = typical_number_of_nodes

        self.network_initial_state = GraphAttention(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_in,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.network_final_state = GraphAttention(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_in,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.network_interpolated_transition_state = GraphAttention(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_out,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""

        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_length = edge_vec.norm(dim=-1)
        edge_length = edge_length.view(-1, 1)
        edge_scalars = torch.ones_like(edge_vec[:, 0])
        edge_scalars = edge_scalars.view(-1, 1)
        kwargs_initial_state = {
            "node_input": data.x,
            "node_attr": data.node_attr,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_attr": edge_length,
            "edge_scalars": edge_scalars,
            "batch": data.batch,
        }

        output_network_initial_state = self.network_initial_state(
            **kwargs_initial_state
        )

        if self.mask_extra_basis:
            output_network_initial_state = (
                output_network_initial_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_initial_state = normalize_to_sum_squares_one(
                output_network_initial_state, data.batch
            )

        edge_src = data.edge_index_final_state[0]
        edge_dst = data.edge_index_final_state[1]
        edge_vec = data["pos_final_state"][edge_src] - data["pos_final_state"][edge_dst]
        edge_length = edge_vec.norm(dim=-1)
        edge_length = edge_length.view(-1, 1)
        edge_scalars = torch.ones_like(edge_vec[:, 0])
        edge_scalars = edge_scalars.view(-1, 1)
        kwargs_final_state = {
            "node_input": data.x_final_state,
            "node_attr": data.node_attr,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_attr": edge_length,
            "edge_scalars": edge_scalars,
            "batch": data.batch,
        }

        output_network_final_state = self.network_final_state(**kwargs_final_state)

        if self.mask_extra_basis:
            output_network_final_state = output_network_final_state * data.basis_mask

        if self.normalize_sumsq:
            output_network_final_state = normalize_to_sum_squares_one(
                output_network_final_state,
                data.batch,
            )

        p = data.p[0]
        p_prime = 1 - p
        x_interpolated_transition_state = (
            p_prime * output_network_initial_state + p * output_network_final_state
        )

        edge_src = data.edge_index_interpolated_transition_state[0]
        edge_dst = data.edge_index_interpolated_transition_state[1]
        edge_vec = (
            data["pos_interpolated_transition_state"][edge_src]
            - data["pos_interpolated_transition_state"][edge_dst]
        )
        edge_length = edge_vec.norm(dim=-1)
        edge_length = edge_length.view(-1, 1)
        edge_scalars = torch.ones_like(edge_vec[:, 0])
        edge_scalars = edge_scalars.view(-1, 1)
        kwargs_interpolated_transition_state = {
            "node_input": x_interpolated_transition_state,
            "node_attr": data.node_attr,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_attr": edge_length,
            "edge_scalars": edge_scalars,
            "batch": data.batch,
        }

        output_network_interpolated_transition_state = (
            self.network_interpolated_transition_state(
                **kwargs_interpolated_transition_state
            )
        )

        if self.mask_extra_basis:
            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_interpolated_transition_state = normalize_to_sum_squares_one(
                output_network_interpolated_transition_state, data.batch
            )
        if self.reduce_output:
            output_network_interpolated_transition_state = scatter(
                output_network_interpolated_transition_state, data.batch, dim=0
            ).div(self.num_nodes**0.5)

            if self.reference_reduced_output_to_initial_state:
                output_network_initial_state = scatter(
                    output_network_initial_state, data.batch, dim=0
                ).div(self.num_nodes**0.5)

                output_network_interpolated_transition_state -= (
                    output_network_initial_state
                )

            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state.mean(dim=1)
            )

        return output_network_interpolated_transition_state
