import os
import logging
import json

import numpy as np

import argparse

import torch
from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
from minimal_basis.model.model_hamiltonian import HamiltonianModel, EquiHamiltonianModel

from e3nn import o3

from utils import (
    get_test_data_path,
    read_inputs_yaml,
)

import torch.nn.functional as F

import matplotlib.pyplot as plt

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "hamiltonian_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
parser.add_argument(
    "--use_equivariant",
    action="store_true",
    help="If set, the model will use equivariant layers.",
)
parser.add_argument(
    "--hidden_channels",
    type=int,
    default=64,
    help="Number of hidden channels in the neural network.",
)
parser.add_argument(
    "--num_basis",
    type=int,
    default=10,
    help="Number of basis functions to use in the model.",
)
parser.add_argument(
    "--num_targets",
    type=int,
    default=5,
    help="Number of targets to predict.",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=4,
    help="Number of layers in the neural network.",
)
parser.add_argument(
    "--max_radius",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--use_best_config",
    action="store_true",
    help="If set, the best configuration is used based on ray tune run.",
)

args = parser.parse_args()
print(args)


if not args.debug:
    import wandb

    if args.use_equivariant:
        wandb.init(project="equi_hamiltonian", entity="sudarshanvj")
    else:
        wandb.init(project="hamiltonian", entity="sudarshanvj")


def train(train_loader):
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


@torch.no_grad()
def validate(val_loader):
    """Validate the model."""
    model.eval()

    # Store all the loses
    losses = 0.0

    for val_batch in val_loader:
        data = val_batch.to(DEVICE)
        predicted_y = model(data)
        loss = F.mse_loss(predicted_y, data.y)

        # Add up the loss
        losses += loss.item() * val_batch.num_graphs

    rmse = np.sqrt(losses / len(val_loader))

    return rmse


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # --- Inputs
    inputs = read_inputs_yaml(os.path.join("input_files", "hamiltonian_model.yaml"))
    graph_generation_method = inputs["graph_generation_method"]

    if args.use_best_config:
        if args.use_equivariant:
            best_config = json.load(
                open("output/best_config_equi_hamiltonian.json", "r")
            )
        else:
            best_config = json.load(open("output/best_config_hamiltonian.json", "r"))
        batch_size = best_config["batch_size"]
        learning_rate = best_config["learning_rate"]
        # Replace args with best config
        args.hidden_channels = best_config["hidden_channels"]
        args.num_layers = best_config["num_layers"]
        if args.use_equivariant:
            args.num_basis = best_config["num_basis"]
            args.num_targets = best_config["num_targets"]
            args.max_radius = best_config["max_radius"]
            args.num_layers = best_config["num_layers"]
            args.hidden_channels = best_config["hidden_channels"]
    else:
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
        validate_json_filename = inputs["debug_validate_json"]
    else:
        train_json_filename = inputs["train_json"]
        validate_json_filename = inputs["validate_json"]

    # Create the training and test datasets
    train_dataset = HamiltonianDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=graph_generation_method,
        basis_file=inputs["basis_file"],
    )
    train_dataset.process()
    validate_dataset = HamiltonianDataset(
        root=get_test_data_path(),
        filename=validate_json_filename,
        graph_generation_method=graph_generation_method,
        basis_file=inputs["basis_file"],
    )
    validate_dataset.process()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(
        validate_dataset, batch_size=len(validate_dataset), shuffle=False
    )

    # Figure out the number of features
    num_node_features = train_dataset.num_node_features
    num_edge_features = train_dataset.num_edge_features
    num_global_features = 1
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    # Create the optimizer
    if args.use_equivariant:
        irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")
        model = EquiHamiltonianModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=args.hidden_channels,
            num_basis=args.num_basis,
            num_global_features=num_global_features,
            num_targets=args.num_targets,
            num_updates=args.num_layers,
            hidden_channels=10,
            max_radius=args.max_radius,
        )
    else:
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
        train_loss = train(train_loader=train_loader)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")

        # Validate the model
        validate_loss = validate(val_loader=validate_loader)
        print(f"Epoch: {epoch}, Validation Loss: {validate_loss}")

        if not args.debug:
            wandb.log({"train_loss": train_loss})
            wandb.log({"val_loss": validate_loss})

    # Plot the comparison between the predicted and actual activation energies
    model.eval()
    predicted_energies = []
    actual_energies = []
    for val_batch in validate_loader:
        data = val_batch.to(DEVICE)
        predicted_y = model(data)
        predicted_energies.extend(predicted_y.detach().cpu().numpy())
        actual_energies.extend(data.y.cpu().numpy())
    # Get the training loader as well
    train_predicted_energies = []
    train_actual_energies = []
    for train_batch in train_loader:
        data = train_batch.to(DEVICE)
        predicted_y = model(data)
        train_predicted_energies.extend(predicted_y.detach().cpu().numpy())
        train_actual_energies.extend(data.y.cpu().numpy())
    # Plot the training and validation data
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    ax.scatter(train_actual_energies, train_predicted_energies, label="Training Data")
    ax.scatter(actual_energies, predicted_energies, label="Validation Data")

    ax.set_xlabel("DFT Activation Energy (Ha)")
    ax.set_ylabel("Predicted Activation Energy (Ha)")

    # Plot the 1:1 line
    x_min = min(min(train_actual_energies), min(actual_energies))
    x_max = max(max(train_actual_energies), max(actual_energies))
    ax.plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--")

    ax.legend()
    if args.use_equivariant:
        fig.savefig("output/equi_hamiltonian_model.png", dpi=300)
    else:
        fig.savefig("output/hamiltonian_model.png", dpi=300)
