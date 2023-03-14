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
    ) -> None:
        """Initialize the reaction model."""
        super().__init__()

        self.num_nodes = typical_number_of_nodes
        self.num_basis = num_basis

        self.network_initial_state = Network(
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
            num_neighbors=num_neighbors,
            num_nodes=typical_number_of_nodes,
            reduce_output=False,
        )

        self.network_final_state = Network(
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
            reduce_output=reduce_output,
        )

    def forward(self, data):
        """Forward pass of the reaction model."""

        species_embedding = soft_one_hot_linspace(
            data.species.squeeze(),
            start=0.0,
            end=self.max_species_embedding,
            number=1,
            basis="smooth_finite",
            cutoff=True,
        )

        output_network_initial_state = self.network_initial_state(
            {
                "pos": data.pos,
                "x": data.x,
                # "z": species_embedding,
                "batch": data.batch,
            }
        )
        self.output_network_initial_state = output_network_initial_state

        output_network_final_state = self.network_final_state(
            {
                "pos": data.pos_final_state,
                "x": data.x_final_state,
                # "z": species_embedding,
                "batch": data.batch,
            }
        )
        self.output_network_final_state = output_network_final_state

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

        return output_network_interpolated_transition_state
