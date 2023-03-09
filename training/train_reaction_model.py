import os
import logging
import json

import numpy as np

import argparse

import torch
from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_reaction import ReactionDataset as Dataset
from minimal_basis.model.model_reaction import ReactionModel as Model

from e3nn import o3

from utils import (
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

import torch.nn.functional as F

import matplotlib.pyplot as plt

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "reaction_model.log"),
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
    "--use_wandb",
    action="store_true",
    help="If set, wandb is used for logging.",
)
parser.add_argument(
    "--use_best_config",
    action="store_true",
    help="If set, the best configuration is used based on ray tune run.",
)
parser.add_argument(
    "--reprocess_dataset",
    action="store_true",
    help="If set, the dataset is reprocessed.",
)
args = parser.parse_args()


if args.use_wandb:
    import wandb

    wandb.init(project="reaction", entity="sudarshanvj")


def train(train_loader):
    """Train the model."""

    model.train()

    # Store all the loses
    losses = 0.0

    for train_batch in train_loader:
        data = train_batch.to(DEVICE)
        optim.zero_grad()

        predicted_y = model(data)
        real_y = data.x_transition_state

        loss = F.mse_loss(predicted_y, real_y)
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
        real_y = data.x_transition_state
        loss = F.mse_loss(predicted_y, real_y)

        # Add up the loss
        losses += loss.item() * val_batch.num_graphs

    rmse = np.sqrt(losses / len(val_loader))

    return rmse


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # --- Inputs
    inputs = read_inputs_yaml(os.path.join("input_files", "reaction_model.yaml"))

    if args.use_best_config:
        best_config = json.load(open("output/best_config_reaction.json", "r"))
        batch_size = best_config["batch_size"]
        learning_rate = best_config["learning_rate"]

        # Replace args with best config
        hidden_channels = best_config["hidden_channels"]
        num_basis = best_config["num_basis"]

    else:
        batch_size = inputs["batch_size"]
        learning_rate = inputs["learning_rate"]
        hidden_channels = inputs["hidden_channels"]
        num_basis = inputs["num_basis"]

    epochs = inputs["epochs"]

    if args.use_wandb:
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
    train_dataset = Dataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        basis_filename=inputs["basis_file"],
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = Dataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        basis_filename=inputs["basis_file"],
    )
    if args.reprocess_dataset:
        validate_dataset.process()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(
        validate_dataset, batch_size=len(validate_dataset), shuffle=False
    )
    # Figure out the number of features
    num_node_features = train_dataset.num_node_features
    num_edge_features = train_dataset.num_edge_features
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    model = Model(
        irreps_sh=inputs["irreps_sh"],
        irreps_in=inputs["irreps_in"],
        irreps_out=inputs["irreps_out"],
        hidden_layers=hidden_channels,
        num_basis=num_basis,
        max_radius=inputs["max_radius"],
    )
    model = model.to(DEVICE)

    if args.use_wandb:
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

        if args.use_wandb:
            wandb.log({"train_loss": train_loss})
            wandb.log({"val_loss": validate_loss})

    # Save the model
    torch.save(model, "output/reaction_model.pt")

    # Store wandb model as artifact
    if args.use_wandb:
        artifact = wandb.Artifact("reaction_model", type="model")
        artifact.add_file("output/reaction_model.pt")
        wandb.run.log_artifact(artifact)
        wandb.finish()
