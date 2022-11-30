import os
import logging
import argparse

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn

from minimal_basis.dataset.dataset_classifier import ClassifierDataset
from minimal_basis.model.model_classifier import ClassifierModel

from utils import (
    read_inputs_yaml,
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
)

LOGFILES_FOLDER = "log_files"
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "classifier_model.log"),
    filemode="w",
    level=logging.INFO,
)

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
    "--reprocess_dataset",
    action="store_true",
)
args = parser.parse_args()

if not args.debug:
    import wandb

    wandb.init(project="classifier_model", entity="sudarshanvj")


def train(loader: DataLoader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(loader: DataLoader, theshold: float = 0.5):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = (out > theshold).float()
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


if __name__ == "__main__":
    """Train a simple classifier model."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Inputs
    inputs = read_inputs_yaml(os.path.join("input_files", "classifier_model.yaml"))
    batch_size = inputs["batch_size"]
    learning_rate = inputs["learning_rate"]
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
    train_dataset = ClassifierDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        filename_classifier_parameters=dataset_parameters_filename,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = ClassifierDataset(
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
    num_edge_features = train_dataset.num_edge_features
    num_global_features = train_dataset.num_global_features
    num_classes = train_dataset.num_classes

    # Create the model
    model = ClassifierModel(
        num_node_features=num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
    )
    model = model.to(device)

    if not args.debug:
        wandb.watch(model)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs):
        # Train the model
        train_loss = train(loader=train_loader)
        train_acc = validate(loader=train_loader)
        logger.info(f"Epoch: {epoch}, Train Accuracy: {train_acc}")

        # Validate the model
        val_acc = validate(loader=validate_loader)
        logger.info(f"Epoch: {epoch}, Validation Accuracy: {val_acc}")

        if not args.debug:
            wandb.log({"train_loss": train_acc})
            wandb.log({"val_loss": val_acc})
