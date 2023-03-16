import os
import logging

import numpy as np

import argparse

import torch
from torch_geometric.loader import DataLoader

from minimal_basis.dataset.dataset_reaction import ReactionDataset as Dataset
from minimal_basis.model.model_reaction import ReactionModel as Model

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

    for train_batch in train_loader:
        data = train_batch.to(DEVICE)
        optim.zero_grad()
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        # loss = F.mse_loss(predicted_y, real_y, reduction="sum")
        loss = F.l1_loss(predicted_y, real_y, reduction="sum")
        loss.backward()

        losses += loss.item()

        num_graphs += train_batch.num_graphs

        optim.step()

    output_metric = losses / num_graphs
    # output_metric = np.sqrt(output_metric)

    return output_metric


@torch.no_grad()
def validate(val_loader):
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for val_batch in val_loader:
        data = val_batch.to(DEVICE)
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            real_y = data.x_transition_state
        elif inputs["prediction_mode"] == "relative_energy":
            real_y = data.total_energy_transition_state - data.total_energy
            predicted_y = predicted_y.mean(dim=1)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        # loss = F.mse_loss(predicted_y, real_y, reduction="sum")
        loss = F.l1_loss(predicted_y, real_y, reduction="sum")

        losses += loss.item()
        num_graphs += val_batch.num_graphs

    output_metric = losses / num_graphs
    # output_metric = np.sqrt(output_metric)

    return output_metric


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    inputs = read_inputs_yaml(os.path.join("input_files", "reaction_model.yaml"))
    if inputs["prediction_mode"] == "coeff_matrix":
        reduce_output = False
    elif inputs["prediction_mode"] == "relative_energy":
        reduce_output = True
    else:
        raise ValueError(f"Prediction mode {inputs['prediction_mode']} not recognized.")

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

    train_dataset = Dataset(
        root=get_train_data_path(),
        filename=train_json_filename,
        basis_filename=inputs["basis_file"],
        max_s_functions=inputs["max_s_functions"],
        max_p_functions=inputs["max_p_functions"],
        max_d_functions=inputs["max_d_functions"],
    )
    if args.reprocess_dataset:
        train_dataset.process()

    validate_dataset = Dataset(
        root=get_validation_data_path(),
        filename=validate_json_filename,
        basis_filename=inputs["basis_file"],
        max_s_functions=inputs["max_s_functions"],
        max_p_functions=inputs["max_p_functions"],
        max_d_functions=inputs["max_d_functions"],
    )

    irreps_in = f"{inputs['max_s_functions']}x0e+{inputs['max_p_functions']}x1e+{inputs['max_d_functions']}x0e+{inputs['max_d_functions']}x2e"
    irreps_out = f"{inputs['max_s_functions']}x0e+{inputs['max_p_functions']}x1e+{inputs['max_d_functions']}x0e+{inputs['max_d_functions']}x2e"

    if args.reprocess_dataset:
        validate_dataset.process()

    train_loader = DataLoader(
        train_dataset, batch_size=inputs["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=len(validate_dataset), shuffle=False
    )

    typical_number_of_nodes = 0
    for data in train_loader:
        typical_number_of_nodes += data.x.shape[0]
    typical_number_of_nodes = typical_number_of_nodes / len(train_dataset)
    typical_number_of_nodes = int(typical_number_of_nodes)

    model = Model(
        irreps_in=irreps_in,
        irreps_hidden=inputs["irreps_hidden"],
        irreps_out=irreps_out,
        irreps_node_attr=inputs["irreps_node_attr"],
        irreps_edge_attr=f"{inputs['num_basis']}x0e",
        radial_layers=inputs["radial_layers"],
        max_radius=inputs["max_radius"],
        num_basis=inputs["num_basis"],
        radial_neurons=inputs["radial_neurons"],
        num_neighbors=inputs["num_neighbors"],
        typical_number_of_nodes=typical_number_of_nodes,
        reduce_output=reduce_output,
    )
    model = model.to(DEVICE)
    print(model)

    if args.use_wandb:
        wandb.watch(model)

    # Create the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=inputs["learning_rate"])

    for epoch in range(1, inputs["epochs"] + 1):
        # Train the model
        train_loss = train(train_loader=train_loader)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")

        # Validate the model
        validate_loss = validate(val_loader=validate_loader)
        print(f"Epoch: {epoch}, Validation Loss: {validate_loss}")

        if args.use_wandb:
            wandb.log({"train_loss": train_loss})
            wandb.log({"val_loss": validate_loss})

    # Save the model
    torch.save(model, "output/reaction_model.pt")

    if args.use_wandb:
        artifact = wandb.Artifact("reaction_model", type="model")
        artifact.add_file("output/reaction_model.pt")
        wandb.run.log_artifact(artifact)
        wandb.finish()
