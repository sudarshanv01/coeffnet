import os
import logging

import numpy as np

import argparse

import wandb

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.model.reaction import ReactionModel as Model

from utils import (
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

from model_functions import construct_model_name, construct_irreps, train, validate

import torch.nn.functional as F


def get_command_line_arguments() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, the calculation is a DEBUG calculation.",
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
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/interp_sn2_model.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--wandb_username",
        type=str,
        default="sudarshanvj",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_command_line_arguments()

    model_name = construct_model_name(args.model_config, debug=args.debug)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("minimal_basis").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    inputs = read_inputs_yaml(os.path.join(args.model_config))
    construct_irreps(inputs)
    transform = T.ToDevice(DEVICE)

    wandb.init(project=model_name, entity=args.wandb_username)
    wandb.config = {
        "learning_rate": inputs["learning_rate"],
        "epochs": inputs["epochs"],
        "batch_size": inputs["batch_size"],
    }

    if args.debug:
        train_json_filename = inputs["debug_train_json"]
        validate_json_filename = inputs["debug_validate_json"]
    else:
        train_json_filename = inputs["train_json"]
        validate_json_filename = inputs["validate_json"]

    kwargs_dataset = inputs["dataset_options"]
    kwargs_dataset["use_minimal_basis_node_features"] = inputs[
        "use_minimal_basis_node_features"
    ]

    train_dataset = Dataset(
        root=get_train_data_path(model_name),
        filename=train_json_filename,
        basis_filename=inputs["basis_file"],
        transform=transform,
        **kwargs_dataset,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = Dataset(
        root=get_validation_data_path(model_name),
        filename=validate_json_filename,
        basis_filename=inputs["basis_file"],
        transform=transform,
        **kwargs_dataset,
    )

    if args.reprocess_dataset:
        validate_dataset.process()

    train_loader = DataLoader(
        train_dataset, batch_size=inputs["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=len(validate_dataset), shuffle=False
    )

    model = Model(**inputs["model_options"])
    model = model.to(DEVICE)

    wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=inputs["learning_rate"])

    for epoch in range(1, inputs["epochs"] + 1):
        train_loss = train(
            train_loader=train_loader, model=model, optim=optim, inputs=inputs
        )
        validate_loss = validate(val_loader=validate_loader, model=model, inputs=inputs)

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": validate_loss})
        logger.info(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validate_loss}"
        )

    torch.save(model, f"output/{model_name}.pt")

    artifact = wandb.Artifact(f"{model_name}", type="model")
    artifact.add_file(f"output/{model_name}.pt")
    logger.debug(f"Added model to artifact: {artifact}.")

    wandb.run.log_artifact(artifact)
    logger.debug(f"Logged artifact: {artifact}.")

    wandb.finish()
