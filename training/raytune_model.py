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

from minimal_basis.dataset.dataset_reaction import ReactionDataset
from minimal_basis.model.model_reaction import ReactionModel
from minimal_basis.data._dtype import DTYPE, TORCH_FLOATS

import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import wandb_mixin

from e3nn import o3

import wandb

from utils import (
    get_train_data_path,
    get_validation_data_path,
    get_test_data_path,
    read_inputs_yaml,
)


def load_data(data_dir: str = "input_files"):
    """Load the data for the model."""

    input_file = os.path.join(BASEDIR, data_dir, f"reaction_model.yaml")
    inputs = read_inputs_yaml(input_file)

    if args.debug:
        if "debug_train_json" not in inputs:
            train_json_filename = inputs["debug_train_json"]
        else:
            train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    validate_json_filename = inputs["validate_json"]

    kwargs_dataset = inputs["dataset_options"]
    kwargs_dataset["use_minimal_basis_node_features"] = inputs[
        "use_minimal_basis_node_features"
    ]

    with FileLock(os.path.expanduser("~/.data.lock")):

        train_dataset = ReactionDataset(
            root=get_train_data_path(),
            filename=os.path.join(BASEDIR, train_json_filename),
            basis_filename=os.path.join(BASEDIR, inputs["basis_file"]),
            **kwargs_dataset,
        )

        validate_dataset = ReactionDataset(
            root=get_validation_data_path(),
            filename=os.path.join(BASEDIR, validate_json_filename),
            basis_filename=os.path.join(BASEDIR, inputs["basis_file"]),
            **kwargs_dataset,
        )

    return (
        train_dataset,
        validate_dataset,
        inputs,
    )


def construct_irreps(inputs):
    """Construct the inputs if needed."""

    if (
        inputs["model_options"]["irreps_in"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"]["irreps_in"] = "1x0e+1x1e"
    elif (
        inputs["model_options"]["irreps_in"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"][
            "irreps_in"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        inputs["model_options"][
            "irreps_in"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1o"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            inputs["model_options"]["irreps_in"] += f"+1x0e+1x2e"
    if (
        inputs["model_options"]["irreps_out"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"]["irreps_out"] = "1x0e+1x1e"
    elif (
        inputs["model_options"]["irreps_out"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"][
            "irreps_out"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        inputs["model_options"][
            "irreps_out"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1o"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            inputs["model_options"]["irreps_out"] += f"+1x0e+1x2e"

    if inputs["model_options"]["irreps_edge_attr"] == "@construct":
        inputs["model_options"][
            "irreps_edge_attr"
        ] = f"{inputs['model_options']['num_basis']}x0e"


@wandb_mixin
def train_model(config: Dict[str, float]):
    """Train the model."""

    train_dataset, validate_dataset, inputs = load_data("input_files")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(validate_dataset, len(validate_dataset), shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _inputs = inputs.copy()
    construct_irreps(_inputs)
    _inputs["model_options"]["irreps_edge_attr"] = f"{config['num_basis']}x0e"
    _inputs["model_options"]["radial_layers"] = config["radial_layers"]
    _inputs["model_options"]["max_radius"] = config["max_radius"]
    _inputs["model_options"]["num_basis"] = config["num_basis"]
    _inputs["model_options"]["radial_neurons"] = config["radial_neurons"]

    model = ReactionModel(**_inputs["model_options"])
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, f"{args.model}_checkpoint.pt")
            )
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    model.train()
    for epoch in range(1, inputs["epochs"]):

        logger.info(f"Epoch {epoch}")

        train_params = {
            "model": model,
            "train_loader": train_loader,
            "optim": optim,
            "device": device,
        }
        train_loss = train(**train_params)
        wandb.log({"train_loss": train_loss})

        validation_params = {
            "model": model,
            "validation_loader": test_loader,
            "device": device,
        }
        val_metric = validation(**validation_params)
        wandb.log({"val_loss": val_metric})

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        (model.state_dict(), optim.state_dict()), f"{args.output_dir}/checkpoint.pt"
    )
    checkpoint = Checkpoint.from_directory(args.output_dir)
    session.report({"loss": train_loss}, checkpoint=checkpoint)
    logger.info("Finished Training")


def get_loss(
    predicted_y: torch.Tensor,
    real_y: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Get the loss."""
    loss_positive = F.mse_loss(predicted_y, real_y, reduction=reduction)
    loss_negative = F.mse_loss(-predicted_y, real_y, reduction=reduction)
    loss = loss_positive if loss_positive < loss_negative else loss_negative
    return loss


def validation(
    model,
    validation_loader: DataLoader,
    device: torch.device,
):
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for test_batch in validation_loader:
        data = test_batch.to(device)
        predicted_y = model(data)
        loss = get_loss(predicted_y, data.x_transition_state, reduction="sum")
        losses += loss.item()
        num_graphs += test_batch.num_graphs

    output_metric = losses / num_graphs

    return output_metric


def train(
    model,
    train_loader: DataLoader,
    optim,
    device: torch.device,
):
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0
    for train_batch in train_loader:
        data = train_batch.to(device)
        optim.zero_grad()

        predicted_y = model(data)
        real_y = data.x_transition_state

        predicted_y = torch.abs(predicted_y)
        loss = get_loss(predicted_y, real_y, reduction="sum")
        loss.backward()

        losses += loss.item()
        num_graphs += train_batch.num_graphs

        optim.step()

    output_metric = losses / num_graphs
    return output_metric


def main(
    num_samples: int = 10,
    max_num_epochs: int = 100,
    gpus_per_trial: int = 1,
    grace_period: int = 5,
    reduction_factor: int = 2,
):
    """Construct the hyperparameter search space and run the experiment."""

    # config = {
    #     "batch_size": tune.choice([5, 10, 15, 20, 25, 30]),
    #     "learning_rate": tune.loguniform(1e-4, 1e-1),
    #     "hidden_channels": tune.choice([32, 64, 128]),
    #     "radial_layers": tune.choice([1, 2, 3, 4, 5]),
    #     "max_radius": tune.choice([1, 2, 3, 4, 5]),
    #     "num_basis": tune.choice([2, 4, 8, 16]),
    #     "radial_neurons": tune.choice([2, 4, 6, 8]),
    # }

    config = {
        "batch_size": tune.choice([15]),
        # "learning_rate": tune.loguniform(1e-4, 1e-3),
        "learning_rate": tune.choice([1e-4, 1e-3]),
        "hidden_channels": tune.choice([32]),
        "radial_layers": tune.choice([1]),
        "max_radius": tune.choice([1]),
        "num_basis": tune.choice([2]),
        "radial_neurons": tune.choice([2]),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )

    config["wandb"] = {
        "api_key_file": api_key_file,
        "project": f"raytune_reaction_model",
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
        run_config=RunConfig(local_dir="./results", name="test_experiment"),
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    logger.info("Best trial config: {}".format(best_result.config))
    logger.info("Best trial final loss: {}".format(best_result.metrics["loss"]))
    best_config = best_result.config
    json.dump(best_config, open(f"output/best_config_{args.model}.json", "w"))


def parse_cli():
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, the calculation is a DEBUG calculation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my_model",
        help="Name of the output directory.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to run.",
    )
    parser.add_argument(
        "--max_num_epochs",
        type=int,
        default=500,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--grace_period",
        type=int,
        default=30,
        help="Number of epochs to run before early stopping.",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=2,
        help="Reduction factor for the ASHA scheduler.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="reaction_model",
        help="Name of the model to train.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_cli()

    BASEDIR = os.path.dirname(os.path.abspath(__file__))

    api_key_file = "~/.wandb_api_key"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(
        num_samples=args.num_samples,
        max_num_epochs=args.max_num_epochs,
        gpus_per_trial=1,
    )
