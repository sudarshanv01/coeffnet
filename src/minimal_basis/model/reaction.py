from typing import Union, Dict, Optional

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2102 import Network
from e3nn.nn.models.v2106.gate_points_networks import MessagePassing


def normalize_to_sum_squares_one(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Normalize the output such that the sum of squares of each graph is 1."""

    sum_squares_output = torch.sum(x**2, dim=1)
    sum_squares_graph = scatter(sum_squares_output, batch, dim=0, reduce="sum")
    normalization_factor = torch.sqrt(sum_squares_graph)
    x = x / normalization_factor[batch].unsqueeze(1)

    return x


class ReactionModel(torch.nn.Module):

    max_species_embedding = 100

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_hidden: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        irreps_node_attr: Union[str, o3.Irreps],
        irreps_edge_attr: Union[str, o3.Irreps],
        radial_layers: int,
        max_radius: float,
        num_basis: int,
        radial_neurons: int,
        num_neighbors: int,
        typical_number_of_nodes: int,
        reduce_output: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_reduced_output_to_initial_state: Optional[bool] = False,
        use_atomic_masses: Optional[bool] = False,
    ) -> None:
        """Torch module for transition state properties prediction.

        Args:
            irreps_in (Union[str, o3.Irreps]): Irreps of the input.
            irreps_hidden (Union[str, o3.Irreps]): Irreps of the hidden layers.
            irreps_out (Union[str, o3.Irreps]): Irreps of the output.
            irreps_node_attr (Union[str, o3.Irreps]): Irreps of the node attributes.
            irreps_edge_attr (Union[str, o3.Irreps]): Irreps of the edge attributes.
            radial_layers (int): Number of radial layers.
            max_radius (float): Maximum radius cutoff
            num_basis (int): Number of basis functions for the network.
            radial_neurons (int): Number of neurons in the radial layers.
            num_neighbors (int): Number of neighbors to consider.
            typical_number_of_nodes (int): Typical number of nodes in the dataset.
            reduce_output (Optional[bool], optional): Whether to reduce the output. Useful for
                scalar predictions. Defaults to False.
            mask_extra_basis (Optional[bool], optional): Whether to mask the extra basis functions in
                the last layer. Defaults to False.
            normalize_sumsq (Optional[bool], optional): Whether to normalize the output by the sum of
                the squared output. Defaults to False.
            reference_reduced_output_to_initial_state (Optional[bool], optional): Whether to reference
                the reduced output to the initial state. Defaults to False.
            use_atomic_masses (Optional[bool], optional): Whether to use atomic masses as node attributes.
        """

        super().__init__()

        self.num_nodes = typical_number_of_nodes
        self.num_basis = num_basis
        self.reduce_output = reduce_output
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out
        self.irreps_node_attr = irreps_node_attr
        self.irreps_edge_attr = irreps_edge_attr
        self.radial_layers = radial_layers
        self.radial_neurons = radial_neurons
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.typical_number_of_nodes = typical_number_of_nodes
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )
        self.use_atomic_masses = use_atomic_masses

        self.network_initial_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            layers=radial_layers,
            max_radius=max_radius,
            number_of_basis=num_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            num_nodes=typical_number_of_nodes,
            reduce_output=False,
        )

        self.network_final_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_in,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            layers=radial_layers,
            max_radius=max_radius,
            number_of_basis=num_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            num_nodes=typical_number_of_nodes,
            reduce_output=False,
        )

        self.network_interpolated_transition_state = Network(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            layers=radial_layers,
            max_radius=max_radius,
            number_of_basis=num_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_nodes=typical_number_of_nodes,
            num_neighbors=num_neighbors,
            reduce_output=False,
        )

    def forward(self, data):
        """Forward pass of the reaction model."""

        x = data.x
        x_final_state = data.x_final_state

        kwargs_initial_state = {
            "pos": data.pos,
            "x": x,
            "batch": data.batch,
        }
        if hasattr(self, "use_atomic_masses"):
            if self.use_atomic_masses:
                kwargs_initial_state["z"] = data.species

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
            "x": x_final_state,
            "batch": data.batch,
        }
        if hasattr(self, "use_atomic_masses"):
            if self.use_atomic_masses:
                kwargs_final_state["z"] = data.species

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
        if hasattr(self, "use_atomic_masses"):
            if self.use_atomic_masses:
                kwargs_interpolated_transition_state["z"] = data.species

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
        num_of_basis: int = 10,
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
        self.number_of_basis = num_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        self.irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(self.lmax + 1) for p in [-1, 1]]
        )

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input]
            + layers * [self.irreps_node_hidden]
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

        node_attr = data["node_attr"]

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

        self.reduce_output = reduce_output
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )

        self.network_initial_state = NetworkForAGraphWithNodeAttributes(
            irreps_node_input=self.irreps_in,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_in,
            max_radius=self.max_radius,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            mul=self.mul,
            layers=self.layers,
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
            "node_attr": data.node_attr,
        }

        kwargs_final_state = {
            "batch": data.batch,
            "pos": data.pos_final_state,
            "edge_index": data.edge_index_final_state,
            "x": data.x,
            "node_attr": data.node_attr_final_state,
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

        output_network_final_state = self.network_final_state(kwargs_final_state)

        if self.mask_extra_basis:
            output_network_final_state = output_network_final_state * data.basis_mask
        if self.normalize_sumsq:
            output_network_final_state = normalize_to_sum_squares_one(
                output_network_final_state, data.batch
            )

        p = data.p[0]
        p_prime = 1 - p
        node_attr = (
            p_prime * output_network_initial_state + p * output_network_final_state
        )

        kwargs_interpolated_transition_state = {
            "batch": data.batch,
            "pos": data.pos_interpolated_transition_state,
            "edge_index": data.edge_index_interpolated_transition_state,
            "x": data.x,
            "node_attr": node_attr,
        }

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
