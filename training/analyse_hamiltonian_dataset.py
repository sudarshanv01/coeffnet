import os

import numpy as np

from e3nn import o3
from e3nn.math import soft_one_hot_linspace

import plotly.express as px

import torch
from torch_geometric.loader import DataLoader

from e3nn.nn.models.gate_points_2102 import Convolution, Network

from minimal_basis.dataset.dataset_reaction import ReactionDataset

from utils import (
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

from ase import units as ase_units

import matplotlib.pyplot as plt

import warnings

if __name__ == "__main__":

    inputs = read_inputs_yaml(os.path.join("input_files", "reaction_model.yaml"))

    train_json_filename = inputs["debug_train_json"]
    validate_json_filename = inputs["debug_validate_json"]

    train_dataset = ReactionDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        basis_filename=inputs["basis_file"],
    )

    validation_dataset = ReactionDataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        basis_filename=inputs["basis_file"],
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    irreps_in = o3.Irreps("1x0e+1x1o")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    irreps_out = o3.Irreps("1x0e+1x1o")
    irreps_hidden = o3.Irreps("4x0e+3x1o")
    number_of_edge_features = 1
    radial_layers = 4
    radial_neurons = 11
    num_neighbors = 4
    max_radius = 4.0
    num_basis = 4
    typical_number_of_nodes = 15

    convolution = Convolution(
        irreps_in=irreps_in,
        irreps_node_attr=irreps_node_attr,
        irreps_edge_attr=irreps_edge_attr,
        irreps_out=irreps_out,
        number_of_edge_features=number_of_edge_features,
        radial_layers=radial_layers,
        radial_neurons=radial_neurons,
        num_neighbors=num_neighbors,
    )
    print(convolution)

    network = Network(
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
    print(network)

    irreps_sh = o3.Irreps("1x0e+1x1o+1x2e")

    for idx, data in enumerate(train_loader):

        node_input = data.x
        node_attr = data.species_initial_state
        edge_src, edge_dst = data.edge_index
        edge_vec = (
            data.pos_interpolated_transition_state[edge_src]
            - data.pos_interpolated_transition_state[edge_dst]
        )
        sh = o3.spherical_harmonics(
            irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        edge_attr = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start=0.0,
            end=max_radius,
            number=num_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        edge_features = torch.norm(edge_vec, dim=1, keepdim=True)

        print(f"Shape of node_attr: {node_attr.shape}")
        print(f"Shape of node_input: {node_input.shape}")
        print(f"Shape of edge_src: {edge_src.shape}")
        print(f"Shape of edge_dst: {edge_dst.shape}")
        print(f"Shape of edge_attr: {edge_attr.shape}")
        print(f"Shape of edge_features: {edge_features.shape}")

        output = convolution(
            node_input=node_input,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_features=edge_features,
        )

        print(f"Shape of output: {output.shape}")

        output_network = network(
            {
                "pos": data.pos,
                "x": data.x,
                "z": data.species_initial_state,
                "batch": data.batch,
            }
        )
        print(f"Shape of output_network: {output_network.shape}")

        break
