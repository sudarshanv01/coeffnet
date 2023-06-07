import os
import logging

from pathlib import Path

import argparse

import wandb

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.model.reaction import ReactionModel as Model

from utils import (
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

from model_functions import construct_model_name, construct_irreps, train, validate

__input_folder__ = "input"


def get_command_line_arguments() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, the calculation is a DEBUG calculation.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Name of the folder to store the input files.",
        default=__input_folder__,
    )
    parser.add_argument(
        "--reprocess_dataset",
        action="store_true",
        help="If set, the dataset is reprocessed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to use.",
        default="reaction",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/rudorff_lilienfeld_sn2_dataset.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="coeff_matrix",
        help="Mode of prediction. Can be either 'coeff_matrix' or 'relative_energy'.",
    )
    parser.add_argument(
        "--wandb_username",
        type=str,
        default="sudarshanvj",
    )
    parser.add_argument(
        "--basis_set_type",
        type=str,
        default="full",
        help="Type of basis set. Can be either 'full' or 'minimal'.",
    )
    parser.add_argument(
        "--basis_set",
        type=str,
        default="6-31g*",
        help="Name of the basis set to use.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_command_line_arguments()
    basis_set_name = args.basis_set.replace("*", "star")
    basis_set_name = basis_set_name.replace("+", "plus")
    basis_set_name = basis_set_name.replace("(", "")
    basis_set_name = basis_set_name.replace(")", "")
    basis_set_name = basis_set_name.replace(",", "")
    basis_set_name = basis_set_name.replace(" ", "_")
    basis_set_name = basis_set_name.lower()

    model_name = construct_model_name(
        dataset_name=args.dataset_name,
        basis_set_type=args.basis_set_type,
        debug=args.debug,
        basis_set=basis_set_name,
    )

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("minimal_basis").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    inputs = read_inputs_yaml(os.path.join(args.model_config))
    input_foldername = (
        Path(args.input_folder)
        / args.dataset_name
        / args.basis_set_type
        / basis_set_name
    )
    dataset_options = inputs["dataset_options"][f"{args.basis_set_type}_basis"]
    learning_options = inputs["learning_options"]
    model_options = inputs["model_options"][args.prediction_mode][
        f"{args.basis_set_type}_basis"
    ]

    construct_irreps(
        model_options=model_options,
        dataset_options=dataset_options,
        prediction_mode=args.prediction_mode,
    )
    transform = T.ToDevice(DEVICE)

    wandb.init(project=model_name, entity=args.wandb_username)
    wandb.config.update(args)
    wandb.config.update({"dataset_options": dataset_options})
    wandb.config.update({"model_options": model_options})

    if args.debug:
        train_json_filename = input_foldername / "train_debug.json"
        validate_json_filename = input_foldername / "validate_debug.json"
    else:
        train_json_filename = input_foldername / "train.json"
        validate_json_filename = input_foldername / "validate.json"

    train_dataset = Dataset(
        root=get_train_data_path(model_name),
        filename=train_json_filename,
        transform=transform,
        **dataset_options,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = Dataset(
        root=get_validation_data_path(model_name),
        filename=validate_json_filename,
        transform=transform,
        **dataset_options,
    )
    if args.reprocess_dataset:
        validate_dataset.process()

    train_loader = DataLoader(
        train_dataset, batch_size=learning_options["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=learning_options["batch_size"], shuffle=False
    )

    model = Model(**model_options)
    model = model.to(DEVICE)

    wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=learning_options["learning_rate"])

    for epoch in range(1, learning_options["epochs"] + 1):

        train_loss = train(
            train_loader=train_loader,
            model=model,
            optim=optim,
            prediction_mode=args.prediction_mode,
        )
        validate_loss = validate(
            val_loader=validate_loader,
            model=model,
            prediction_mode=args.prediction_mode,
        )

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
