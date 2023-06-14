from typing import Union, Dict, Optional

import logging

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2102 import Network
from e3nn.nn.models.v2106.gate_points_networks import MessagePassing

logger = logging.getLogger(__name__)


def normalize_to_sum_squares_one(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Normalize the output such that the sum of squares of each graph is 1."""

    sum_squares_output = torch.sum(x**2, dim=1)
    sum_squares_graph = scatter(sum_squares_output, batch, dim=0, reduce="sum")
    normalization_factor = torch.sqrt(sum_squares_graph)
    x = x / normalization_factor[batch].unsqueeze(1)

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
        layers: int,
        number_of_basis: Optional[int] = 10,
        radial_layers: Optional[int] = 3,
        radial_neurons: Optional[int] = 128,
        reduce_output: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_reduced_output_to_initial_state: Optional[bool] = False,
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
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )

        if isinstance(irreps_in, str):
            self.irreps_in = o3.Irreps(irreps_in)
        if isinstance(irreps_node_attr, str):
            self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        if isinstance(irreps_out, str):
            self.irreps_out = o3.Irreps(irreps_out)

        lp_irreps_hidden = [(l, (-1) ** l) for l in self.irreps_in.ls]
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l, p)) for l, p in lp_irreps_hidden]
        )
        logger.info(f"irreps_hidden: {self.irreps_hidden}")

        self.network_initial_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=f"{self.number_of_basis}x0e",
            layers=self.layers,
            max_radius=max_radius,
            number_of_basis=self.number_of_basis,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=False,
        )

        self.network_final_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=f"{self.number_of_basis}x0e",
            layers=self.layers,
            max_radius=max_radius,
            number_of_basis=self.number_of_basis,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=False,
        )

        self.network_interpolated_transition_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=f"{self.number_of_basis}x0e",
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
        }

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
        }

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

            if self.reference_reduced_output_to_initial_state:

                output_network_initial_state = scatter(
                    output_network_initial_state, data.batch, dim=0
                ).div(self.num_nodes**0.5)

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

        lp_irreps_hidden = [(l, (-1) ** l) for l in irreps_node_input.ls]
        self.irreps_hidden = o3.Irreps([(mul, (l, p)) for l, p in lp_irreps_hidden])
        logger.info(f"irreps_hidden: {self.irreps_hidden}")

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input]
            + layers * [self.irreps_hidden]
            + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
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

        edge_src = data["edge_index"][0]
        edge_dst = data["edge_index"][1]

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
        number_of_basis: Optional[int] = 10,
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
        self.number_of_basis = number_of_basis

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
            kwargs_final_state["node_attr"] = data.node_attr_final_state

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
            kwargs_interpolated_transition_state[
                "node_attr"
            ] = data.node_attr_interpolated_transition_state

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
