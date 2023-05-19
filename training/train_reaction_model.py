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

import torch.nn.functional as F


def train(train_loader: DataLoader) -> float:
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for data in train_loader:
        optim.zero_grad()
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
            if inputs["model_options"]["make_absolute"]:
                real_y = torch.abs(real_y)
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        loss = F.l1_loss(predicted_y, real_y, reduction="sum")
        loss.backward()

        losses += loss.item()

        num_graphs += data.num_graphs

        optim.step()

    output_metric = losses / num_graphs

    return output_metric


@torch.no_grad()
def validate(val_loader: DataLoader) -> float:
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for data in val_loader:
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
            if inputs["model_options"]["make_absolute"]:
                real_y = torch.abs(real_y)
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        loss = F.l1_loss(predicted_y, real_y, reduction="sum")

        losses += loss.item()
        num_graphs += data.num_graphs

    output_metric = losses / num_graphs

    return output_metric


def construct_irreps(inputs: dict) -> None:
    """Construct the inputs if there is an @construct in the inputs.

    Args:
        inputs (dict): The inputs dictionary.

    """

    if inputs["model_options"]["make_absolute"]:
        parity = "e"
    else:
        parity = "o"

    if (
        inputs["model_options"]["irreps_in"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"]["irreps_in"] = f"1x0e+1x1{parity}"
    elif (
        inputs["model_options"]["irreps_in"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"][
            "irreps_in"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        inputs["model_options"][
            "irreps_in"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1{parity}"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            inputs["model_options"]["irreps_in"] += f"+1x2e"
    if (
        inputs["model_options"]["irreps_out"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"]["irreps_out"] = f"1x0e+1x1{parity}"
    elif (
        inputs["model_options"]["irreps_out"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        inputs["model_options"][
            "irreps_out"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        inputs["model_options"][
            "irreps_out"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1{parity}"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            inputs["model_options"]["irreps_out"] += f"+1x2e"

    if inputs["model_options"]["irreps_edge_attr"] == "@construct":
        inputs["model_options"][
            "irreps_edge_attr"
        ] = f"{inputs['model_options']['num_basis']}x0e"

    logger.debug(f"irreps_in: {inputs['model_options']['irreps_in']}")
    logger.debug(f"irreps_out: {inputs['model_options']['irreps_out']}")
    logger.debug(f"irreps_edge_attr: {inputs['model_options']['irreps_edge_attr']}")
    logger.debug(f"irreps_node_attr: {inputs['model_options']['irreps_node_attr']}")


def construct_model_name() -> str:
    """Construct the model name based on the config filename and
    the debug flag."""

    model_name = args.model_config.split("/")[-1].split(".")[0]
    if args.debug:
        model_name += "_debug"

    return model_name


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

    model_name = construct_model_name()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("minimal_basis").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
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
        train_loss = train(train_loader=train_loader)
        validate_loss = validate(val_loader=validate_loader)

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": validate_loss})

    torch.save(model, f"output/{model_name}.pt")

    artifact = wandb.Artifact(f"{model_name}", type="model")
    artifact.add_file(f"output/f{model_name}.pt")
    logger.debug(f"Added model to artifact: {artifact}.")

    wandb.run.log_artifact(artifact)
    logger.debug(f"Logged artifact: {artifact}.")

    wandb.finish()
