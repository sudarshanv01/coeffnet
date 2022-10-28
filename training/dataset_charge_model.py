import os
import logging
import argparse

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from minimal_basis.dataset.dataset_charges import ChargeDataset

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
    """Test a convolutional Neural Network based on the charge model."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    folder_string, PLOT_FOLDER = create_plot_folder()

    inputs = read_inputs_yaml(os.path.join("input_files", "charge_model.yaml"))

    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    # Create the training and test datasets
    train_dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=GRAPH_GENERATION_METHOD,
    )
    train_dataset.process()
    print(f"Number of training barriers: {len(train_dataset.data.y)}")

    if args.inspect_data:
        inspect_data(train_dataset.data.y, "barrier_data")
        inspect_data(train_dataset.data.global_attr, "reaction_energy_data")

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
            node_attrs = nx.get_node_attributes(graph.subgraph(nx_graph), "x")
            node_labels = {k: [round(a, 2) for a in v] for k, v in node_attrs.items()}
            nx.draw_networkx_labels(
                graph.subgraph(nx_graph),
                pos=nx.spring_layout(graph.subgraph(nx_graph), seed=42),
                labels=node_labels,
                ax=ax[j],
            )
            # Draw edge labels
            edge_attrs = nx.get_edge_attributes(graph.subgraph(nx_graph), "edge_attr")
            nx.draw_networkx_edge_labels(
                graph.subgraph(nx_graph),
                pos=nx.spring_layout(graph.subgraph(nx_graph), seed=42),
                edge_labels={k: round(v, 2) for k, v in edge_attrs.items()},
                ax=ax[j],
            )

        fig.savefig(f"output/graph_{idx}.png", dpi=300)
        plt.close(fig)
