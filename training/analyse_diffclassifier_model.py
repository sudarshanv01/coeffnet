import os
import logging
import argparse
import json
from pprint import pformat

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from torchmetrics.classification import BinaryAUROC

from minimal_basis.dataset import DiffClassifierDataset
from minimal_basis.model import MessagePassingDiffClassifierModel


from utils import (
    read_inputs_yaml,
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def statistics_model(loader: DataLoader, threshold: float = 0.5):
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


def comparison_plots(loader, ax):
    """Plot the activation energies against the reaction energies and
    see how good the model prediction is."""

    pred, actual, _ = statistics_model(loader)

    reaction_barrier = []
    reaction_energy = []
    for data in loader:
        reaction_barrier.append(data.reaction_barrier.cpu().detach().numpy())
        reaction_energy.append(data.reaction_energy.cpu().detach().numpy())

    reaction_energy = np.concatenate(reaction_energy)
    reaction_barrier = np.concatenate(reaction_barrier)
    sns.scatterplot(
        x=reaction_energy,
        y=reaction_barrier,
        hue=pred.cpu().detach().numpy(),
        ax=ax,
    )
    ax.set_xlabel("Reaction energy")
    ax.set_ylabel("Activation energy")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reprocess_dataset",
        action="store_true",
        help="Reprocess the dataset and save it to a file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Load the `diffclassifier` model and gauge how good it is."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_cli()

    # Filename of the model
    saved_model_filename = "model_files/diffclassifier_model.pt"

    inputs = read_inputs_yaml(os.path.join("input_files", "diffclassifier_model.yaml"))

    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]
    dataset_parameters_filename = inputs["dataset_json"]
    pretrain_params_json = inputs["pretrain_params_json"]

    best_config = json.load(open("output/best_config_diffclassifier.json", "r"))
    batch_size = best_config["batch_size"]
    learning_rate = best_config["learning_rate"]
    # Replace args with best config
    hidden_channels = best_config["hidden_channels"]
    num_updates = best_config["num_layers"]

    # Create the training and test datasets
    train_dataset = DiffClassifierDataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        pretrain_params_json=pretrain_params_json,
    )
    if args.reprocess_dataset:
        train_dataset.process()
    validate_dataset = DiffClassifierDataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        pretrain_params_json=pretrain_params_json,
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
        hidden_channels=hidden_channels,
        num_updates=num_updates,
    )
    model = model.to(device)

    # Load the model
    model.load_state_dict(torch.load(saved_model_filename))

    # Plot `data.y`, which is the activation energy against the
    # difference of global_attr and global_attr_reactant with the
    # colors being `pred`
    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2.75), constrained_layout=True)
    comparison_plots(train_loader, ax[0])
    comparison_plots(validate_loader, ax[1])
    ax[0].set_title("Training set")
    ax[1].set_title("Validation set")
    logger.info("Saving the comparison plot")
    fig.savefig("output/diffclassifier_comparison.png", dpi=300)

    # Get the metrics for the model
    pred, actual, output_sigmoid = statistics_model(loader=validate_loader)
    # Get the AUCROC score through torchmetrics
    aucroc_metric = BinaryAUROC(thresholds=None)
    aucroc = aucroc_metric(pred, actual)
    logger.info(f"AUCROC: {aucroc}")
    # Get other metrics
    metrics_dict = metrics(pred, actual)
    logger.info(f"Metrics: {pformat(metrics_dict)}")

    # Compute the AUCROC for the validation dataset
    recall = []  # True positive rate
    fpr = []  # False positive rate
    for threshold in np.linspace(0, 1, inputs["threshold_num"]):
        logger.info(f"Threshold: {threshold}")
        pred, actual, _ = statistics_model(loader=validate_loader, threshold=threshold)
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
