import os
import logging

import numpy as np

import argparse

import torch
from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
from minimal_basis.model.model_hamiltonian import HamiltonianModel


from utils import (
    get_test_data_path,
    read_inputs_yaml,
)

import torch.nn.functional as F

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "hamiltonian_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
parser.add_argument(
    "--hidden_channels",
    type=int,
    default=64,
    help="Number of hidden channels in the neural network.",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=4,
    help="Number of layers in the neural network.",
)
args = parser.parse_args()


if not args.debug:
    import wandb

    wandb.init(project="minimal-basis-training", entity="sudarshanvj")


def train():
    """Train the model."""

    model.train()

    # Store all the loses
    losses = 0.0

    for train_batch in train_loader:
        data = train_batch.to(DEVICE)
        optim.zero_grad()
        predicted_y = model(data)
        loss = F.mse_loss(predicted_y, data.y)
        loss.backward()

        # Add up the loss
        losses += loss.item() * train_batch.num_graphs

        # Take an optimizer step
        optim.step()

    rmse = np.sqrt(losses / len(train_loader))

    return rmse


if __name__ == "__main__":

    if args.debug:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # --- Inputs
    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))
    graph_generation_method = inputs["graph_generation_method"]
    batch_size = inputs["batch_size"]
    learning_rate = inputs["learning_rate"]
    epochs = inputs["epochs"]

    if not args.debug:
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    # Create the training and test datasets
    train_dataset = HamiltonianDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=graph_generation_method,
        basis_file=inputs["basis_file"],
    )
    train_dataset.process()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Figure out the number of features
    num_node_features = train_dataset.num_node_features
    num_edge_features = train_dataset.num_edge_features
    num_global_features = 1
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    # Create the optimizer
    model = HamiltonianModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels=args.hidden_channels,
        num_updates=args.num_layers,
    )
    model = model.to(DEVICE)
    if not args.debug:
        wandb.watch(model)

    # Create the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs):
        # Train the model
        loss = train()
        print(f"Epoch: {epoch}, Loss: {loss}")

        if not args.debug:
            wandb.log({"loss": loss})
