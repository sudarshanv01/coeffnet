import torch

from torch_geometric.loader import DataLoader
from torch.nn import functional as F

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.model.reaction import ReactionModel as Model
from minimal_basis.loss.eigenvectors import Unsigned_MSELoss


def construct_model_name(model_config: str, debug: bool = False) -> str:
    """Construct the model name based on the config filename and
    the debug flag."""

    model_name = model_config.split("/")[-1].split(".")[0]
    if debug:
        model_name += "_debug"

    return model_name


def construct_irreps(inputs: dict, prediction_mode: str) -> None:
    """Construct the inputs if there is an @construct in the inputs.

    Args:
        inputs (dict): The inputs dictionary.

    """

    model_options = inputs[f"model_options_{prediction_mode}"]

    if model_options["make_absolute"]:
        parity = "e"
    else:
        parity = "o"

    if (
        model_options["irreps_in"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        model_options["irreps_in"] = f"1x0e+1x1{parity}"
    elif (
        model_options["irreps_in"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        model_options[
            "irreps_in"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        model_options[
            "irreps_in"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1{parity}"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            model_options["irreps_in"] += f"+1x2e"
    if (
        model_options["irreps_out"] == "@construct"
        and inputs["use_minimal_basis_node_features"]
    ):
        model_options["irreps_out"] = f"1x0e+1x1{parity}"
    elif (
        model_options["irreps_out"] == "@construct"
        and not inputs["use_minimal_basis_node_features"]
    ):
        model_options[
            "irreps_out"
        ] = f"{inputs['dataset_options']['max_s_functions']}x0e"
        model_options[
            "irreps_out"
        ] += f"+{inputs['dataset_options']['max_p_functions']}x1{parity}"
        for i in range(inputs["dataset_options"]["max_d_functions"]):
            model_options["irreps_out"] += f"+1x2e"

    if model_options["irreps_edge_attr"] == "@construct":
        model_options["irreps_edge_attr"] = f"{model_options['num_basis']}x0e"


def signed_coeff_matrix_loss(data, predicted_y, do_backward=True):
    """Get the loss when converting the coefficient matrix to the density."""

    real_y = data.x_transition_state
    batch = data.batch
    batch_size = data.num_graphs

    loss = Unsigned_MSELoss()(predicted_y, real_y, batch, batch_size)
    if do_backward:
        loss.backward()

    return loss.item()


def relative_energy_loss(data, predicted_y, do_backward=True):
    """Get the loss when predicting the relative energy."""

    real_y = data.total_energy_transition_state - data.total_energy
    predicted_y = predicted_y.mean(dim=1)
    loss = F.l1_loss(predicted_y, real_y, reduction="sum")

    if do_backward:
        loss.backward()

    return loss.item()


def train(model: Model, train_loader: DataLoader, optim, prediction_mode: str) -> float:
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for data in train_loader:
        optim.zero_grad()
        predicted_y = model(data)

        if prediction_mode == "coeff_matrix":
            losses += signed_coeff_matrix_loss(data, predicted_y)
        elif prediction_mode == "relative_energy":
            losses += relative_energy_loss(data, predicted_y)
        else:
            raise ValueError(f"Prediction mode {prediction_mode} not recognized.")

        num_graphs += data.num_graphs
        optim.step()

    output_metric = losses / num_graphs

    return output_metric


@torch.no_grad()
def validate(model: Model, val_loader: DataLoader, prediction_mode: str) -> float:
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for data in val_loader:
        predicted_y = model(data)

        if prediction_mode == "coeff_matrix":
            losses += signed_coeff_matrix_loss(data, predicted_y, do_backward=False)
        elif prediction_mode == "relative_energy":
            losses += relative_energy_loss(data, predicted_y, do_backward=False)
        else:
            raise ValueError(f"Prediction mode {prediction_mode} not recognized.")

        num_graphs += data.num_graphs

    output_metric = losses / num_graphs

    return output_metric
