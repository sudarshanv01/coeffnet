"""Train any model based on the specifications of the input file."""
import os
import json
import torch
from typing import Dict, List, Tuple
from filelock import FileLock
import logging
import argparse

import numpy as np

import torch.nn.functional as F

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.reaction import ReactionDataset
from minimal_basis.model.reaction import ReactionModel

import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

from e3nn import o3

from utils import (
    get_train_data_path,
    get_validation_data_path,
    read_inputs_yaml,
)
from model_functions import construct_model_name, construct_irreps, train, validate


def load_data() -> Tuple[ReactionDataset, ReactionDataset, Dict]:
    """Load the data for the model."""

    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]

    kwargs_dataset = inputs["dataset_options"]
    kwargs_dataset["use_minimal_basis_node_features"] = inputs[
        "use_minimal_basis_node_features"
    ]
    transform = T.ToDevice(DEVICE)

    with FileLock(os.path.expanduser("~/.data.lock")):

        train_dataset = ReactionDataset(
            root=get_train_data_path(model_name),
            filename=os.path.join(BASEDIR, train_json_filename),
            basis_filename=os.path.join(BASEDIR, inputs["basis_file"]),
            transform=transform,
            **kwargs_dataset,
        )

        validate_dataset = ReactionDataset(
            root=get_validation_data_path(model_name),
            filename=os.path.join(BASEDIR, validate_json_filename),
            basis_filename=os.path.join(BASEDIR, inputs["basis_file"]),
            transform=transform,
            **kwargs_dataset,
        )

    return (
        train_dataset,
        validate_dataset,
        inputs,
    )


def train_model(config: Dict[str, float]):
    """Train the model."""

    train_dataset, validate_dataset, inputs = load_data()

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(validate_dataset, len(validate_dataset), shuffle=True)

    _inputs = inputs.copy()
    construct_irreps(_inputs)
    _inputs["model_options"]["irreps_edge_attr"] = f"{config['num_basis']}x0e"
    _inputs["model_options"]["radial_layers"] = config["radial_layers"]
    _inputs["model_options"]["max_radius"] = config["max_radius"]
    _inputs["model_options"]["num_basis"] = config["num_basis"]
    _inputs["model_options"]["radial_neurons"] = config["radial_neurons"]
    _inputs["model_options"][
        "irreps_hidden"
    ] = f"{config['hidden_s_functions']}x0e+{config['hidden_p_functions']}x1o+{config['hidden_d_functions']}x2e"
    _inputs["epochs"] = args.max_num_epochs

    model = ReactionModel(**_inputs["model_options"])
    model.to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, f"checkpoint.pt")
            )
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    model.train()
    for epoch in range(1, _inputs["epochs"] + 1):

        logger.info(f"Epoch {epoch}")

        train_loss = train(
            train_loader=train_loader, model=model, optim=optim, inputs=_inputs
        )
        validate_loss = validate(
            val_loader=validate_loader, model=model, inputs=_inputs
        )
        session.report(
            {"loss": validate_loss, "train_loss": train_loss},
        )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        (model.state_dict(), optim.state_dict()), f"{args.output_dir}/checkpoint.pt"
    )
    checkpoint = Checkpoint.from_directory(args.output_dir)
    session.report(
        {"loss": validate_loss, "train_loss": train_loss}, checkpoint=checkpoint
    )
    logger.info("Finished Training")


def main(
    num_samples: int = 1,
    max_num_epochs: int = 100,
    gpus_per_trial: int = 1,
    grace_period: int = 5,
    reduction_factor: int = 2,
):
    """Construct the hyperparameter search space and run the experiment."""

    config = {
        "batch_size": tune.grid_search([5, 10, 15, 20, 25, 30]),
        "learning_rate": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "radial_layers": tune.grid_search([1, 2, 3, 4, 5]),
        "max_radius": tune.grid_search([1, 2, 3, 4, 5]),
        "num_basis": tune.grid_search([2, 4, 8, 16]),
        "radial_neurons": tune.grid_search([32, 64, 128]),
        "hidden_s_functions": tune.grid_search([16, 32, 64, 128, 256]),
        "hidden_p_functions": tune.grid_search([16, 32, 64, 128, 256]),
        "hidden_d_functions": tune.grid_search([16, 32, 64, 128, 256]),
    }

    scheduler = ASHAScheduler(
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )

    config["wandb"] = {
        "api_key_file": api_key_file,
    }

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 2, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(
            local_dir=args.output_dir,
            name=model_name,
            callbacks=[
                WandbLoggerCallback(
                    project=model_name,
                    api_key_file=api_key_file,
                    log_config=True,
                    upload_checkpoints=True,
                )
            ],
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    logger.info("Best trial config: {}".format(best_result.config))
    logger.info("Best trial final loss: {}".format(best_result.metrics["loss"]))
    best_config = best_result.config
    json.dump(best_config, open(f"output/best_config_{model_name}.json", "w"))


def get_command_line_arguments():
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Name of the output directory.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to run.",
    )
    parser.add_argument(
        "--max_num_epochs",
        type=int,
        default=20,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--grace_period",
        type=int,
        default=15,
        help="Number of epochs to run before early stopping.",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=2,
        help="Reduction factor for the ASHA scheduler.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/interp_sn2_model.yaml",
        help="Path to the model config file.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_command_line_arguments()

    model_name = construct_model_name(args.model_config)

    inputs = read_inputs_yaml(args.model_config)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BASEDIR = os.path.dirname(os.path.abspath(__file__))

    api_key_file = "~/.wandb_api_key"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(
        num_samples=args.num_samples,
        max_num_epochs=args.max_num_epochs,
        gpus_per_trial=1,
    )
