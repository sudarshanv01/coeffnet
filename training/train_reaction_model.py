import os
import logging
from pprint import pprint

import numpy as np

import argparse

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.dataset_reaction import ReactionDataset as Dataset
from minimal_basis.model.model_reaction import ReactionModel as Model
from minimal_basis.transforms.absolute import Absolute

from utils import (
    get_test_data_path,
    get_validation_data_path,
    get_train_data_path,
    read_inputs_yaml,
)

import torch.nn.functional as F

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "reaction_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
parser.add_argument(
    "--use_wandb",
    action="store_true",
    help="If set, wandb is used for logging.",
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
args = parser.parse_args()


if args.use_wandb:
    import wandb

    wandb.init(project="reaction", entity="sudarshanvj")


def train(train_loader):
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for data in train_loader:
        optim.zero_grad()
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        elif inputs["prediction_mode"] == "forces":
            real_y = data.forces_transition_state
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        loss = F.l1_loss(predicted_y, real_y, reduction="sum")
        # loss_negative = F.l1_loss(-predicted_y, real_y, reduction="sum")
        # loss = loss_positive if loss_positive < loss_negative else loss_negative
        # loss.backward()

        losses += loss.item()

        num_graphs += data.num_graphs

        optim.step()

    output_metric = losses / num_graphs

    return output_metric


@torch.no_grad()
def validate(val_loader):
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for data in val_loader:
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        elif inputs["prediction_mode"] == "forces":
            real_y = data.forces_transition_state
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        loss_positive = F.l1_loss(predicted_y, real_y, reduction="sum")
        loss_negative = F.l1_loss(-predicted_y, real_y, reduction="sum")
        loss = loss_positive if loss_positive < loss_negative else loss_negative

        losses += loss.item()
        num_graphs += data.num_graphs

    output_metric = losses / num_graphs

    return output_metric


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
        inputs["model_options"][
            "irreps_in"
        ] += f"+{inputs['dataset_options']['max_d_functions']}x0e"
        inputs["model_options"][
            "irreps_in"
        ] += f"+{inputs['dataset_options']['max_d_functions']}x2e"
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
        inputs["model_options"][
            "irreps_out"
        ] += f"+{inputs['dataset_options']['max_d_functions']}x0e"
        inputs["model_options"][
            "irreps_out"
        ] += f"+{inputs['dataset_options']['max_d_functions']}x2e"

    if inputs["model_options"]["irreps_edge_attr"] == "@construct":
        inputs["model_options"][
            "irreps_edge_attr"
        ] = f"{inputs['model_options']['num_basis']}x0e"

    pprint(inputs)


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    inputs = read_inputs_yaml(os.path.join("input_files", "reaction_model.yaml"))
    construct_irreps(inputs)
    transform = T.ToDevice(DEVICE)

    if args.use_wandb:
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
        root=get_train_data_path(),
        filename=train_json_filename,
        basis_filename=inputs["basis_file"],
        transform=transform,
        **kwargs_dataset,
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = Dataset(
        root=get_validation_data_path(),
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
    print(model)

    if args.use_wandb:
        wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=inputs["learning_rate"])

    for epoch in range(1, inputs["epochs"] + 1):
        train_loss = train(train_loader=train_loader)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")

        validate_loss = validate(val_loader=validate_loader)
        print(f"Epoch: {epoch}, Validation Loss: {validate_loss}")

        if args.use_wandb:
            wandb.log({"train_loss": train_loss})
            wandb.log({"val_loss": validate_loss})

    torch.save(model, "output/reaction_model.pt")

    if args.use_wandb:
        artifact = wandb.Artifact("reaction_model", type="model")
        artifact.add_file("output/reaction_model.pt")
        wandb.run.log_artifact(artifact)
        wandb.finish()
