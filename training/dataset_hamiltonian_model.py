import os
import logging
import argparse

import numpy as np

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import seaborn
import matplotlib.pyplot as plt

import networkx as nx

from utils import (
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
)

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "charge_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inspect_data",
    action="store_true",
    help="If set, the data is inspected through a histogram of the outputs.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
args = parser.parse_args()


def inspect_data(y, filename):
    """Make a histogram of y."""
    fig, ax = plt.subplots()
    seaborn.histplot(y, ax=ax)
    fig.savefig(f"output/{filename}.png", dpi=300)


if __name__ == "__main__":
    """Test a convolutional Neural Network based on the Hamiltonian model."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    folder_string, PLOT_FOLDER = create_plot_folder()

    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))

    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    # Create the training and test datasets
    train_dataset = HamiltonianDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=GRAPH_GENERATION_METHOD,
        basis_file=inputs["basis_file"],
    )
    train_dataset.process()
    print(train_dataset)

    if args.inspect_data:
        inspect_data(train_dataset.data.y, "barrier_data")
        inspect_data(train_dataset.data.global_attr, "reaction_energy_data")

    # Check if the batch loader works
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch in train_loader:
        print(batch)

    # Make plots of the data
    for idx, datapoint in enumerate(train_dataset):

        graph = convert.to_networkx(
            datapoint,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
        )

        # Make a plot of the graph
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        for j, nx_graph in enumerate(nx.weakly_connected_components(graph)):

            nx.draw(
                graph.subgraph(nx_graph),
                pos=nx.spring_layout(graph.subgraph(nx_graph), seed=42),
                with_labels=False,
                cmap="Set2",
                ax=ax[j],
            )

            # Draw node labels
            node_attrs_ = nx.get_node_attributes(graph.subgraph(nx_graph), "x")
            for node_idx, node_features in node_attrs_.items():

                fign, axn = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

                axn.set_title(f"Node {node_idx}")
                axn.set_xlabel("Basis functions")
                axn.set_ylabel("Basis functions")

                # Reshape the node features
                node_features = np.array(node_features).reshape(-1, 2)

                # Reshape the node features into a square matrix
                node_features = node_features.reshape(
                    int(np.sqrt(node_features.shape[0])),
                    int(np.sqrt(node_features.shape[0])),
                    -1,
                )
                node_features_spin_up = node_features[..., 0]
                node_features_spin_down = node_features[..., 1]

                # Make a contour plot of the node features
                cax = axn.imshow(node_features_spin_up, cmap="viridis")

                # Set basis function list as the axis labels for x,y axis
                basis_functions = ["s", "p", "p", "p", "d", "d", "d", "d", "d"]
                axn.set_xticks(np.arange(len(basis_functions)))
                axn.set_yticks(np.arange(len(basis_functions)))
                axn.set_xticklabels(basis_functions)
                axn.set_yticklabels(basis_functions)

                # Make colorbar
                fig.colorbar(cax, ax=axn)

                # Set equal aspect ratio
                axn.set_aspect("equal")
                fign.savefig(
                    f"output/hamiltonian_datapoint_{idx}_graph_{j}_node_index_{node_idx}.png",
                    dpi=300,
                )

                plt.close(fign)

            # Same treatment for the edge attributes
            edge_attrs_ = nx.get_edge_attributes(graph.subgraph(nx_graph), "edge_attr")
            for edge_idx, edge_features in edge_attrs_.items():
                fign, axn = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

                axn.set_title(f"Edge {edge_idx}")
                axn.set_xlabel("Basis functions")
                axn.set_ylabel("Basis functions")

                # Reshape the edge features
                edge_features = np.array(edge_features).reshape(-1, 2)

                # Reshape the edge features into a square matrix
                edge_features = edge_features.reshape(
                    int(np.sqrt(edge_features.shape[0])),
                    int(np.sqrt(edge_features.shape[0])),
                    -1,
                )
                edge_features_spin_up = edge_features[..., 0]
                edge_features_spin_down = edge_features[..., 1]

                # Make a contour plot of the edge features
                cax = axn.imshow(edge_features_spin_up, cmap="viridis")

                # Set basis function list as the axis labels for x,y axis
                basis_functions = ["s", "p", "p", "p", "d", "d", "d", "d", "d"]
                axn.set_xticks(np.arange(len(basis_functions)))
                axn.set_yticks(np.arange(len(basis_functions)))
                axn.set_xticklabels(basis_functions)
                axn.set_yticklabels(basis_functions)

                # Make colorbar
                fig.colorbar(cax, ax=axn)

                # Set equal aspect ratio
                axn.set_aspect("equal")
                fign.savefig(
                    f"output/hamiltonian_datapoint_{idx}_graph_{j}_edge_index_{edge_idx}.png",
                    dpi=300,
                )

                plt.close(fign)

        fig.savefig(f"output/hamiltonian_graph_{idx}.png", dpi=300)
        plt.close(fig)
