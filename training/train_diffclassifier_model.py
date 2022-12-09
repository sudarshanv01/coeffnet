import os
import logging
import argparse
import json

import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from minimal_basis.dataset import DiffClassifierDataset
from minimal_basis.model import MessagePassingDiffClassifierModel

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
parser.add_argument(
    "--use_best_config",
    action="store_true",
    help="If set, the best configuration is used based on ray tune run.",
)
parser.add_argument(
    "--use_wandb",
    action="store_true",
)
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.init(project="diffclassifier_model", entity="sudarshanvj")


def train(loader: DataLoader):
    model.train()
    losses = 0.0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        out = out.view(-1)
        actual = data.y
        actual = actual.to(TORCH_FLOATS[1])

        loss = criterion(out, actual)
        loss.backward()
        losses += loss.item() * data.num_graphs

        optimizer.step()

    return losses


@torch.no_grad()
def validate(loader: DataLoader, theshold: float = 0.5):
    model.eval()

    # Store the number of correct predictions
    correct = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        # Apply a sigmoid to the out values
        out = torch.sigmoid(out)
        pred = (out > theshold).float()
        pred = pred.view(-1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


if __name__ == "__main__":
    """Train a simple classifier model."""

    if not args.use_cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    inputs = read_inputs_yaml(os.path.join("input_files", "diffclassifier_model.yaml"))

    # --- Inputs
    if args.use_best_config:
        best_config = json.load(open("output/best_config_diffclassifier.json", "r"))
        batch_size = best_config["batch_size"]
        learning_rate = best_config["learning_rate"]
        # Replace args with best config
        args.hidden_channels = best_config["hidden_channels"]
        args.num_updates = best_config["num_layers"]
    else:
        batch_size = inputs["batch_size"]
        learning_rate = inputs["learning_rate"]

    if args.debug:
        epochs = 200
    else:
        epochs = inputs["epochs"]

    if args.use_wandb:
        wandb.init(project="diffclassifier_model", entity="sudarshanvj")
        wandb.config.update(
            {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_channels": args.hidden_channels,
                "num_updates": args.num_updates,
            }
        )

    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]
    dataset_parameters_filename = inputs["dataset_json"]
    pretrain_params_json = inputs["pretrain_params_json"]

    # Create the training and test datasets
    train_dataset = DiffClassifierDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        pretrain_params_json=pretrain_params_json,
        debug=args.debug,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = DiffClassifierDataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        pretrain_params_json=pretrain_params_json,
        debug=args.debug,
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
    model = MessagePassingDiffClassifierModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels=args.hidden_channels,
        num_updates=args.num_updates,
        debug=args.debug,
    )
    model = model.to(device)

    if args.use_wandb:
        wandb.watch(model)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs):
        # Train the model
        loss_training = train(loader=train_loader)
        logger.info(f"Epoch: {epoch}, Training Loss: {loss_training}")
        train_acc = validate(loader=train_loader)
        logger.info(f"Epoch: {epoch}, Train Accuracy: {train_acc}")

        # Validate the model
        val_acc = validate(loader=validate_loader)
        logger.info(f"Epoch: {epoch}, Validation Accuracy: {val_acc}")

        if args.use_wandb:
            wandb.log({"train_acc": train_acc})
            wandb.log({"val_acc": val_acc})

    # Save the model
    if not os.path.exists("model_files"):
        os.mkdir("model_files")
    torch.save(
        model.state_dict(), os.path.join("model_files", "diffclassifier_model.pt")
    )
