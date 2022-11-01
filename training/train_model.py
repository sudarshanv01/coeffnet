"""Train any model based on the specifications of the input file."""
import os
import torch
from typing import Dict, List, Tuple
from filelock import FileLock
import logging
import argparse

import numpy as np

import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel
from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
from minimal_basis.model.model_hamiltonian import HamiltonianModel

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


from utils import (
    get_test_data_path,
    read_inputs_yaml,
)


def load_data(data_dir="input_files", model="charge"):
    """Load the data for the model."""

    # Load the input file
    inputs = read_inputs_yaml(os.path.join(data_dir, f"{model}_model.yaml"))

    # Graph generation method
    graph_generation_method = inputs["graph_generation_method"]

    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    test_json_filename = inputs["test_json"]

    # Decipher the name of the Dataset based on the model
    if model == "charge":
        DatasetModule = ChargeDataset
    elif model == "hamiltonian":
        DatasetModule = HamiltonianDataset
    else:
        raise ValueError(f"Model {model} not recognized.")

    with FileLock(os.path.expanduser("~/.data.lock")):
        # Create the training and test datasets
        train_dataset = DatasetModule(
            root=get_test_data_path(),
            filename=train_json_filename,
            graph_generation_method=graph_generation_method,
        )
        train_dataset.process()

        test_dataset = DatasetModule(
            root=get_test_data_path(),
            filename=test_json_filename,
            graph_generation_method=graph_generation_method,
        )
        test_dataset.process()

    return (
        train_dataset,
        test_dataset,
        {
            "num_node_features": train_dataset.num_node_features,
            "num_edge_features": train_dataset.num_edge_features,
        },
    )


def train_model(config):
    """Train the model."""

    train_dataset, test_dataset, dataset_info = load_data()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True
    )

    if args.model == "charge":
        model = ChargeModel(
            num_node_features=dataset_info["num_node_features"],
            num_edge_features=dataset_info["num_edge_features"],
            num_global_features=num_global_features,
            hidden_channels=config["hidden_channels"],
            num_updates=config["num_layers"],
        )
    elif args.model == "hamiltonian":
        model = HamiltonianModel(
            num_node_features=dataset_info["num_node_features"],
            num_edge_features=dataset_info["num_edge_features"],
            num_global_features=num_global_features,
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

        if not args.debug:
            wandb.log({"loss": train_loss})

    # Validation loss
    val_loss = validation(model, test_loader, device)

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        (model.state_dict(), optim.state_dict()), f"{args.output_dir}/checkpoint.pt"
    )
    checkpoint = Checkpoint.from_directory(args.output_dir)
    session.report({"loss": val_loss}, checkpoint=checkpoint)
    print("Finished Training")


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
            "batch_size": 5,
            "learning_rate": 1e-3,
            "hidden_channels": 32,
            "num_layers": 1,
            "epochs": max_num_epochs,
        }

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)

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
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]
        )
    )


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
        default="minimal-basis-training",
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

    LOGFILES_FOLDER = "log_files"
    logging.basicConfig(
        filename=os.path.join(LOGFILES_FOLDER, f"{args.wandb_project}_model.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    if not args.debug:
        import wandb

        wandb.init(project=args.wandb_project, entity="sudarshanvj")

    num_global_features = 1

    main(num_samples=2, max_num_epochs=2, gpus_per_trial=0)
