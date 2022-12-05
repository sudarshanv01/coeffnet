import os
import logging
import argparse
from pprint import pformat

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torch_geometric
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from minimal_basis.dataset.dataset_classifier import ActivationBarrierDataset
from minimal_basis.model.model_classifier import (
    ActivationBarrierModel,
    MessagePassingActivationBarrierModel,
)

from utils import (
    read_inputs_yaml,
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
)

from plot_params import get_plot_params
from minimal_basis.data._dtype import TORCH_FLOATS

get_plot_params()

if not os.path.exists("output"):
    os.makedirs("output")

if not os.path.exists(os.path.join("output", "log_files")):
    os.makedirs(os.path.join("output", "log_files"))

LOGFILES_FOLDER = os.path.join("output", "log_files")
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "classifier_model.log"),
    filemode="w",
    level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())

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
    "--num_updates",
    type=int,
    default=10,
    help="Number of updates to perform on the model.",
)
parser.add_argument(
    "--use_cpu",
    action="store_true",
)
parser.add_argument(
    "--reprocess_dataset",
    action="store_true",
)
args = parser.parse_args()

if not args.debug:
    import wandb

    wandb.init(project="activation_barrier_model", entity="sudarshanvj")


def train(loader):
    """Train the model."""

    model.train()

    # Store all the loses
    losses = 0.0

    for train_batch in loader:
        data = train_batch.to(device)
        optimizer.zero_grad()
        predicted_y = model(data)
        predicted_y = predicted_y.view(-1)
        loss = F.mse_loss(predicted_y, data.y)
        loss.backward()

        # Add up the loss
        losses += loss.item() * train_batch.num_graphs

        # Take an optimizer step
        optimizer.step()

    rmse = np.sqrt(losses / len(loader))

    return rmse


@torch.no_grad()
def validate(loader):
    """Validate the model."""
    model.eval()

    # Store all the loses
    losses = 0.0

    for val_batch in loader:
        data = val_batch.to(device)
        predicted_y = model(data)
        predicted_y = predicted_y.view(-1)
        loss = F.mse_loss(predicted_y, data.y)

        # Add up the loss
        losses += loss.item() * val_batch.num_graphs

    rmse = np.sqrt(losses / len(loader))

    return rmse


if __name__ == "__main__":
    """Train a simple classifier model."""

    if not args.use_cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # --- Inputs
    inputs = read_inputs_yaml(
        os.path.join("input_files", "activation_barrier_model.yaml")
    )
    batch_size = inputs["batch_size"]
    learning_rate = inputs["learning_rate"]
    if args.debug:
        epochs = 200
    else:
        epochs = inputs["epochs"]

    if not args.debug:
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]
    dataset_parameters_filename = inputs["dataset_json"]

    # Create the training and test datasets
    train_dataset = ActivationBarrierDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        filename_classifier_parameters=dataset_parameters_filename,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = ActivationBarrierDataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        filename_classifier_parameters=dataset_parameters_filename,
    )
    if args.reprocess_dataset:
        validate_dataset.process()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    # Figure out the number of features
    num_node_features = train_dataset.num_node_features
    logger.info(f"Number of node features: {num_node_features}")
    num_edge_features = train_dataset.num_edge_features
    logger.info(f"Number of edge features: {num_edge_features}")
    num_global_features = train_dataset.num_global_features
    logger.info(f"Number of global features: {num_global_features}")
    num_classes = train_dataset.num_classes

    # Create the model
    model = MessagePassingActivationBarrierModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels=args.hidden_channels,
        num_updates=args.num_updates,
    )
    model = model.to(device)
    if not args.debug:
        wandb.watch(model)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(1, epochs):
        # Train the model
        train(loader=train_loader)
        train_acc = validate(loader=train_loader)
        logger.info(f"Epoch: {epoch}, Train Loss: {train_acc}")

        # Validate the model
        val_acc = validate(loader=validate_loader)
        logger.info(f"Epoch: {epoch}, Validation Loss: {val_acc}")

        if not args.debug:
            wandb.log({"train_loss": train_acc})
            wandb.log({"val_loss": val_acc})

        # scheduler.step()

    # Save the model
    if not os.path.exists("model_files"):
        os.mkdir("model_files")
    torch.save(
        model.state_dict(), os.path.join("model_files", "activation_barrier_model.pt")
    )
