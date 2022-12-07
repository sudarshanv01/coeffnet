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

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
from minimal_basis.dataset.dataset_interpolate import InterpolateDataset
from minimal_basis.model.model_charges import ChargeModel
from minimal_basis.model.model_hamiltonian import HamiltonianModel, EquiHamiltonianModel
from minimal_basis.model.model_interpolate import MessagePassingInterpolateModel

import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import wandb_mixin

from e3nn import o3

import wandb

from utils import (
    get_test_data_path,
    read_inputs_yaml,
)


def load_data(data_dir: str = "input_files", model: str = "charge"):
    """Load the data for the model."""

    # Load the input file
    # get the absolute path to the input file
    input_file = os.path.join(
        os.path.dirname(__file__), data_dir, f"{model}_model.yaml"
    )
    inputs = read_inputs_yaml(input_file)

    if args.debug:
        if "debug_train_json" not in inputs:
            train_json_filename = inputs["train_json"]
        else:
            train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    validate_json_filename = inputs["validate_json"]

    # Decipher the name of the Dataset based on the model
    if model == "charge":
        DatasetModule = ChargeDataset
        graph_generation_method = inputs["graph_generation_method"]
        kwargs = {"graph_generation_method": graph_generation_method}
    elif model == "hamiltonian":
        DatasetModule = HamiltonianDataset
        graph_generation_method = inputs["graph_generation_method"]
        kwargs = {
            "basis_file": inputs["basis_file"],
            "graph_generation_method": graph_generation_method,
        }
    elif model == "equi_hamiltonian":
        DatasetModule = HamiltonianDataset
        graph_generation_method = inputs["graph_generation_method"]
        kwargs = {
            "basis_file": inputs["basis_file"],
            "graph_generation_method": graph_generation_method,
        }
    elif model == "interpolate":
        DatasetModule = InterpolateDataset
        pretrain_params_json = inputs["pretrain_params_json"]
        kwargs = {"pretrain_params_json": pretrain_params_json}
    else:
        raise ValueError(f"Model {model} not recognized.")

    with FileLock(os.path.expanduser("~/.data.lock")):
        # Create the training and test datasets
        train_dataset = DatasetModule(
            root=get_test_data_path(),
            filename=train_json_filename,
            **kwargs,
        )

        validate_dataset = DatasetModule(
            root=get_test_data_path(),
            filename=validate_json_filename,
            **kwargs,
        )

    return (
        train_dataset,
        validate_dataset,
        {
            "num_node_features": train_dataset.num_node_features,
            "num_edge_features": train_dataset.num_edge_features,
            "num_global_features": train_dataset.num_global_features,
        },
    )


@wandb_mixin
def train_model(config: Dict[str, float]):
    """Train the model."""

    train_dataset, validate_dataset, dataset_info = load_data("input_files", args.model)

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(validate_dataset, len(validate_dataset), shuffle=True)

    if args.model == "charge":
        model = ChargeModel(
            num_node_features=dataset_info["num_node_features"],
            num_edge_features=dataset_info["num_edge_features"],
            num_global_features=dataset_info["num_global_features"],
            hidden_channels=config["hidden_channels"],
            num_updates=config["num_layers"],
        )
    elif args.model == "hamiltonian":
        model = HamiltonianModel(
            num_node_features=dataset_info["num_node_features"],
            num_edge_features=dataset_info["num_edge_features"],
            num_global_features=dataset_info["num_global_features"],
            hidden_channels=config["hidden_channels"],
            num_updates=config["num_layers"],
        )
    elif args.model == "equi_hamiltonian":
        irreps_out = o3.Irreps("1x0e + 3x1o + 5x2e")
        model = EquiHamiltonianModel(
            irreps_out_per_basis=irreps_out,
            hidden_layers=config["hidden_channels"],
            num_basis=config["num_basis"],
            num_global_features=dataset_info["num_global_features"],
            num_targets=config["num_targets"],
            num_updates=config["num_layers"],
            hidden_channels=config["hidden_channels"],
            max_radius=config["max_radius"],
        )
    elif args.model == "interpolate":
        model = MessagePassingInterpolateModel(
            num_node_features=dataset_info["num_node_features"],
            num_edge_features=dataset_info["num_edge_features"],
            num_global_features=dataset_info["num_global_features"],
            hidden_channels=config["hidden_channels"],
            num_updates=config["num_layers"],
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    # Train the model
    model.train()
    for epoch in range(1, config["epochs"]):
        # Train the model
        train_loss = train(model, train_loader, optim, device)
        wandb.log({"train_loss": train_loss})

        # Validation loss
        val_loss = validation(model, test_loader, device)
        wandb.log({"val_loss": val_loss})

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        (model.state_dict(), optim.state_dict()), f"{args.output_dir}/checkpoint.pt"
    )
    checkpoint = Checkpoint.from_directory(args.output_dir)
    session.report({"loss": val_loss}, checkpoint=checkpoint)
    logger.info("Finished Training")


@torch.no_grad()
def validation(model, validation_loader, device):
    """Validate the model."""

    # Store all the loses
    losses = 0.0

    for test_batch in validation_loader:
        data = test_batch.to(device)
        predicted_y = model(data)
        loss = F.mse_loss(predicted_y, data.y)

        # Add up the loss
        losses += loss.item() * test_batch.num_graphs

    rmse = np.sqrt(losses / len(validation_loader))

    return rmse


def train(model, train_loader, optim, device):
    """Train the model."""

    # Store all the loses
    losses = 0.0

    for train_batch in train_loader:
        data = train_batch.to(device)
        optim.zero_grad()
        predicted_y = model(data)
        loss = F.mse_loss(predicted_y, data.y)
        loss.backward()

        # Add up the loss
        losses += loss.item() * train_batch.num_graphs

        # Take an optimizer step
        optim.step()

    rmse = np.sqrt(losses / len(train_loader))

    return rmse


def main(num_samples=10, max_num_epochs=100, gpus_per_trial=1):
    # Define the search space
    if not args.debug:
        config = {
            "batch_size": tune.choice([5, 10, 15, 20]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "hidden_channels": tune.choice([32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3, 4, 5]),
            "epochs": max_num_epochs,
        }
    else:
        config = {
            "batch_size": tune.choice([5, 10]),
            "learning_rate": tune.loguniform(1e-3, 1e-4),
            "hidden_channels": tune.choice([32, 64]),
            "num_layers": tune.choice([1, 2]),
            "epochs": max_num_epochs,
        }

    # Add additonal config options for the equi-hamiltonian model
    if args.model == "equi_hamiltonian":
        config.update(
            {
                "num_basis": tune.choice([10, 20, 30]),
                "num_targets": tune.choice([5, 10, 20]),
                "max_radius": tune.choice([5, 10, 20]),
            }
        )

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)

    # Add wandb to the config
    config["wandb"] = {
        "api_key_file": api_key_file,
        "project": args.wandb_project,
        "group": args.model,
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

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(best_result.metrics["loss"])
    )

    # Write out the best performing model as a checkpoint
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
        "--wandb_project",
        type=str,
        default="minimal_basis_ray_tune",
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="charge",
        help="Name of the model to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my_model",
        help="Name of the output directory.",
    )
    parser.add_argument(
        "--test_setup", action="store_true", help="Test the imports and quit."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_cli()

    if args.test_setup:
        quit()

    api_key_file = "~/.wandb_api_key"

    LOGFILES_FOLDER = "log_files"
    logging.basicConfig(
        filename=os.path.join(LOGFILES_FOLDER, f"{args.wandb_project}_model.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    main(num_samples=5, max_num_epochs=500, gpus_per_trial=1)
