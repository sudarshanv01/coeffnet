from typing import List, Tuple, Union, Any, Optional

import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn.models.gate_points_2102 import Network
from e3nn.math import soft_one_hot_linspace


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
        make_absolute: Optional[bool] = False,
        mask_extra_basis: Optional[bool] = False,
        normalize_sumsq: Optional[bool] = False,
        reference_reduced_output_to_initial_state: Optional[bool] = False,
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
            make_absolute (Optional[bool], optional): Whether to report the absolute value of output.
                Defaults to False.
            mask_extra_basis (Optional[bool], optional): Whether to mask the extra basis functions in
                the last layer. Defaults to False.
            normalize_sumsq (Optional[bool], optional): Whether to normalize the output by the sum of
                the squared output. Defaults to False.
            reference_reduced_output_to_initial_state (Optional[bool], optional): Whether to reference
                the reduced output to the initial state. Defaults to False.
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
        self.make_absolute = make_absolute
        self.mask_extra_basis = mask_extra_basis
        self.normalize_sumsq = normalize_sumsq
        self.reference_reduced_output_to_initial_state = (
            reference_reduced_output_to_initial_state
        )

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

    def _normalize_to_sum_squares_one(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Normalize the output such that the sum of squares of each graph is 1."""

        sum_squares_output = torch.sum(x**2, dim=1)
        sum_squares_graph = scatter(sum_squares_output, batch, dim=0, reduce="sum")
        normalization_factor = torch.sqrt(sum_squares_graph)
        x = x / normalization_factor[batch].unsqueeze(1)

        return x

    def forward(self, data):
        """Forward pass of the reaction model."""

        if self.make_absolute:
            x = torch.abs(data.x)
            x_final_state = torch.abs(data.x_final_state)
        else:
            x = data.x
            x_final_state = data.x_final_state

        output_network_initial_state = self.network_initial_state(
            {
                "pos": data.pos,
                "x": x,
                "batch": data.batch,
            }
        )

        if self.make_absolute:
            output_network_initial_state = torch.abs(output_network_initial_state)

        if self.mask_extra_basis:
            output_network_initial_state = (
                output_network_initial_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_initial_state = self._normalize_to_sum_squares_one(
                output_network_initial_state, data.batch
            )

        output_network_final_state = self.network_final_state(
            {
                "pos": data.pos_final_state,
                "x": x_final_state,
                "batch": data.batch,
            }
        )

        if self.make_absolute:
            output_network_final_state = torch.abs(output_network_final_state)

        if self.mask_extra_basis:
            output_network_final_state = output_network_final_state * data.basis_mask
        if self.normalize_sumsq:
            output_network_final_state = self._normalize_to_sum_squares_one(
                output_network_final_state, data.batch
            )

        p = data.p
        p_prime = 1 - p
        x_interpolated_transition_state = (
            p[0] * output_network_initial_state
            + p_prime[0] * output_network_final_state
        )

        output_network_interpolated_transition_state = (
            self.network_interpolated_transition_state(
                {
                    "pos": data.pos_interpolated_transition_state,
                    "x": x_interpolated_transition_state,
                    "batch": data.batch,
                }
            )
        )

        if self.make_absolute:
            output_network_interpolated_transition_state = torch.abs(
                output_network_interpolated_transition_state
            )
        if self.mask_extra_basis:
            output_network_interpolated_transition_state = (
                output_network_interpolated_transition_state * data.basis_mask
            )
        if self.normalize_sumsq:
            output_network_interpolated_transition_state = (
                self._normalize_to_sum_squares_one(
                    output_network_interpolated_transition_state, data.batch
                )
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
