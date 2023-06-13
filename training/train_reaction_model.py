import os
import logging

from datetime import datetime

from pathlib import Path

import wandb

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from e3nn import o3

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.model.reaction import MessagePassingReactionModel as Model

from utils import (
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

from model_functions import construct_model_name, construct_irreps, train, validate
from cli_functions import get_command_line_arguments, get_basis_set_name


def create_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("minimal_basis").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    args = get_command_line_arguments()
    basis_set_name = get_basis_set_name(args.basis_set)
    logger.info(f"Basis set name: {basis_set_name}")

    model_name = construct_model_name(
        dataset_name=args.dataset_name,
        debug=args.debug,
    )

    timestamp = create_timestamp()

    logger.info(f"Model name: {model_name}")

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
    learning_options = {}
    learning_options["batch_size"] = args.batch_size
    learning_options["learning_rate"] = args.learning_rate
    learning_options["num_epochs"] = args.num_epochs
    logger.info(f"Learning options: {learning_options}")
    model_options = inputs["model_options"][args.prediction_mode][
        f"{args.basis_set_type}_basis"
    ]

    transform = T.ToDevice(DEVICE)

    # If args.debug is set, then the calculation is a DEBUG calculation, set wandb to dryrun
    if args.debug:
        wandb.init(project=model_name, entity=args.wandb_username, mode="dryrun")
    else:
        wandb.init(project=model_name, entity=args.wandb_username)
    wandb.config.update(args)
    wandb.config.update({"dataset_options": dataset_options})

    if args.debug:
        train_json_filename = input_foldername / "train_debug.json"
        validate_json_filename = input_foldername / "validate_debug.json"
    else:
        train_json_filename = input_foldername / "train.json"
        validate_json_filename = input_foldername / "validate.json"

    train_dataset = Dataset(
        root=get_train_data_path(
            model_name + "_" + args.basis_set_type + "_basis" + "_" + basis_set_name
        ),
        filename=train_json_filename,
        transform=transform,
        **dataset_options,
    )
    validate_dataset = Dataset(
        root=get_validation_data_path(
            model_name + "_" + args.basis_set_type + "_basis" + "_" + basis_set_name
        ),
        filename=validate_json_filename,
        transform=transform,
        **dataset_options,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=learning_options["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=learning_options["batch_size"], shuffle=False
    )

    model_options["irreps_in"] = "1x0e"
    model_options["irreps_node_attr"] = train_dataset.irreps_node_attr
    model_options["irreps_out"] = train_dataset.irreps_out
    model_options["lmax"] = o3.Irreps(model_options["irreps_out"]).lmax

    wandb.config.update({"model_options": model_options})
    logger.info(f"Model Options: {model_options}")
    model = Model(**model_options)
    model = model.to(DEVICE)
    wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=learning_options["learning_rate"])

    for epoch in range(1, learning_options["num_epochs"] + 1):

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

    torch.save(model, f"output/{model_name}_{timestamp}.pt")

    artifact = wandb.Artifact(f"{model_name}", type="model")
    artifact.add_file(f"output/{model_name}_{timestamp}.pt")
    logger.debug(f"Added model to artifact: {artifact}.")

    wandb.run.log_artifact(artifact)
    logger.debug(f"Logged artifact: {artifact}.")

    wandb.finish()
