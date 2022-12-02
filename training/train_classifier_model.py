import os
import logging
import argparse
from pprint import pformat

import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from minimal_basis.dataset.dataset_classifier import ClassifierDataset
from minimal_basis.model.model_classifier import ClassifierModel

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

    wandb.init(project="classifier_model", entity="sudarshanvj")


def train(loader: DataLoader):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        # Run the model
        out = model(data)

        # Apply a sigmoid to the out values
        out = torch.sigmoid(out)
        # Make out into a 1D tensor
        out = out.view(-1)

        # Get the actial data
        actual = data.y

        # Convert the actual values to torch float
        actual = actual.to(TORCH_FLOATS[1])

        loss = criterion(out, actual)
        loss.backward()


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
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


def validation_curve(loader: DataLoader, threshold: float = 0.5):
    """Return the predicted and actual values for the dataset."""

    # Make a tensor to store the predicted and actual values
    pred = []
    actual = []
    output_model = []
    out = []

    for data in loader:
        data = data.to(device)
        _out = model(data)
        output_model.append(_out)

        # Apply a sigmoid to the out values
        _out = torch.sigmoid(_out)
        out.append(_out)

        # Make out into a 1D tensor
        _out = _out.view(-1)
        _pred = (_out > threshold).float()
        _actual = data.y

        pred.append(_pred)
        actual.append(_actual)

    # Concatenate the tensors
    pred = torch.cat(pred)
    actual = torch.cat(actual)
    out = torch.cat(out)
    output_model = torch.cat(output_model)

    return pred, actual, [output_model, out]


def metrics(pred: torch.Tensor, actual: torch.Tensor):
    """Calculate the metrics for the model."""
    if pred.shape != actual.shape:
        raise ValueError("The shapes of the prediction and actual are not the same.")

    # Calculate the true positives, true negatives, false positives, and false negatives
    tp = ((pred == 1) & (actual == 1)).sum().item()
    tn = ((pred == 0) & (actual == 0)).sum().item()
    fp = ((pred == 1) & (actual == 0)).sum().item()
    fn = ((pred == 0) & (actual == 1)).sum().item()
    logger.debug(f"True positives: {tp}")
    logger.debug(f"True negatives: {tn}")
    logger.debug(f"False positives: {fp}")
    logger.debug(f"False negatives: {fn}")

    # Calculate the precision, recall, and f1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)

    data_dict = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return data_dict


if __name__ == "__main__":
    """Train a simple classifier model."""

    if not args.use_cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # --- Inputs
    inputs = read_inputs_yaml(os.path.join("input_files", "classifier_model.yaml"))
    batch_size = inputs["batch_size"]
    learning_rate = inputs["learning_rate"]
    if args.debug:
        epochs = 2
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
        out_channels=1,
    )
    model = model.to(device)

    if not args.debug:
        wandb.watch(model)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    for epoch in range(1, epochs):
        # Train the model
        train(loader=train_loader)
        train_acc = validate(loader=train_loader)
        logger.info(f"Epoch: {epoch}, Train Accuracy: {train_acc}")

        # Validate the model
        val_acc = validate(loader=validate_loader)
        logger.info(f"Epoch: {epoch}, Validation Accuracy: {val_acc}")

        if not args.debug:
            wandb.log({"train_loss": train_acc})
            wandb.log({"val_loss": val_acc})

    # Save the model
    if not os.path.exists("model_files"):
        os.mkdir("model_files")
    torch.save(model.state_dict(), os.path.join("model_files", "classifier_model.pt"))

    # Get the metrics for the model
    pred, actual, output_sigmoid = validation_curve(loader=validate_loader)
    metrics_dict = metrics(pred, actual)
    # Make a plot with the sigmoid values
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.75), constrained_layout=True)
    ax.plot(output_sigmoid[0].cpu().detach(), output_sigmoid[1].cpu().detach(), "o")
    # Plot also a generic sigmoid curve
    ax.plot(np.linspace(-5, 5, 100), 1 / (1 + np.exp(-np.linspace(-5, 5, 100))), "k--")
    ax.set_xlabel("Model output after linear layer")
    ax.set_ylabel("Sigmoid output")
    fig.savefig("output/sigmoid.png", dpi=300)
    plt.close(fig)

    logger.info(f"Metrics: {pformat(metrics_dict)}")

    # Compute the AUCROC for the validation dataset
    recall = []  # True positive rate
    fpr = []  # False positive rate
    for threshold in np.linspace(0, 1, inputs["threshold_num"]):
        logger.info(f"Threshold: {threshold}")
        pred, actual, _ = validation_curve(loader=validate_loader, threshold=threshold)
        try:
            data_dict = metrics(pred=pred, actual=actual)
        except ZeroDivisionError:
            logger.info("No predictions were made.")
            continue
        logger.info(f"Threshold: {threshold}, Metrics: {data_dict}")

        recall.append(data_dict["recall"])
        fpr.append(data_dict["fpr"])

    # Plot the precision recall curve
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.75), constrained_layout=True)
    ax.plot(fpr, recall, "o-")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Precision Recall Curve")

    # Plot the random guess line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    fig.savefig(os.path.join("output", "precision_recall_curve.png"), dpi=300)
