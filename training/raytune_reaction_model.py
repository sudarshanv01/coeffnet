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
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import create_scheduler

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
    construct_irreps(_inputs, prediction_mode=args.prediction_mode)
    model_options = _inputs[f"model_options_{args.prediction_mode}"]
    model_options["irreps_edge_attr"] = f"{config['num_basis']}x0e"
    model_options["radial_layers"] = config["radial_layers"]
    model_options["max_radius"] = config["max_radius"]
    model_options["num_basis"] = config["num_basis"]
    model_options["radial_neurons"] = config["radial_neurons"]
    model_options[
        "irreps_hidden"
    ] = f"{config['hidden_s_functions']}x0e+{config['hidden_p_functions']}x1o+{config['hidden_d_functions']}x2e"
    _inputs["epochs"] = args.max_num_epochs

    model = ReactionModel(**_inputs[f"model_options_{args.prediction_mode}"])
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
        session.report(
            {"validate_loss": validate_loss, "train_loss": train_loss},
        )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        (model.state_dict(), optim.state_dict()), f"{args.output_dir}/checkpoint.pt"
    )
    checkpoint = Checkpoint.from_directory(args.output_dir)
    session.report(
        {"validate_loss": validate_loss, "train_loss": train_loss},
        checkpoint=checkpoint,
    )
    logger.info("Finished Training")


def main(
    gpus_per_trial: int = 1,
    grace_period: int = 5,
    reduction_factor: int = 2,
):
    """Construct the hyperparameter search space and run the experiment."""

    config = {
        "batch_size": tune.grid_search([16, 32, 64]),
        "learning_rate": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "hidden_s_functions": tune.grid_search([64, 128, 256]),
        "hidden_p_functions": tune.grid_search([64, 128, 256]),
        "hidden_d_functions": tune.grid_search([64, 128, 256]),
    }

    scheduler = create_scheduler(
        args.scheduler_name,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        time_attr="training_iteration",
        metric="train_loss",
        mode="min",
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
            scheduler=scheduler,
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

    best_result = results.get_best_result("train_loss", "min")

    logger.info("Best trial config: {}".format(best_result.config))
    logger.info(
        "Best trial final train loss: {}".format(best_result.metrics["train_loss"])
    )
    best_config = best_result.config
    logger.info(
        "Corresponding validate loss: {}".format(best_result.metrics["validate_loss"])
    )
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
        "--max_num_epochs",
        type=int,
        default=200,
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
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="coeff_matrix",
        help="Mode of prediction. Can be either 'coeff_matrix' or 'relative_energy'.",
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        default="asha",
        help="Scheduler to use. See ray for more details.",
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

    main(gpus_per_trial=1)
