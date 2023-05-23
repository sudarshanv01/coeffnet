import torch

from torch_geometric.loader import DataLoader
from torch.nn import functional as F

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.model.reaction import ReactionModel as Model


def construct_model_name(model_config: str, debug: bool = False) -> str:
    """Construct the model name based on the config filename and
    the debug flag."""

    model_name = model_config.split("/")[-1].split(".")[0]
    if debug:
        model_name += "_debug"

    return model_name


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


def coeff2density_loss(data, predicted_y, do_backward=True):
    """Get the loss when converting the coefficient matrix to the density."""

    real_y = data.x_transition_state
    losses = 0

    for i in range(data.num_graphs):
        real_c_ij = real_y[data.batch == i]
        predicted_c_ij = predicted_y[data.batch == i]

        real_c_ij = real_c_ij.reshape(-1, 1)
        predicted_c_ij = predicted_c_ij.reshape(-1, 1)

        real_cij_dot_cij_T = real_c_ij @ real_c_ij.T
        predicted_c_ij_dot_c_ij_T = predicted_c_ij @ predicted_c_ij.T

        loss = F.l1_loss(
            predicted_c_ij_dot_c_ij_T, real_cij_dot_cij_T, reduction="mean"
        )
        if do_backward:
            loss.backward(retain_graph=True)
        losses += loss.item()
        yield losses


def relative_energy_loss(data, predicted_y, do_backward=True):
    """Get the loss when predicting the relative energy."""

    real_y = data.total_energy_transition_state - data.total_energy
    predicted_y = predicted_y.mean(dim=1)
    loss = F.l1_loss(predicted_y, real_y, reduction="sum")

    if do_backward:
        loss.backward(retain_graph=True)

    return loss.item()


def train(model: Model, train_loader: DataLoader, optim, inputs: dict) -> float:
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for data in train_loader:
        optim.zero_grad()
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            losses += sum(coeff2density_loss(data, predicted_y))
        elif inputs["prediction_mode"] == "relative_energy":
            losses += relative_energy_loss(data, predicted_y)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        num_graphs += data.num_graphs
        optim.step()

    output_metric = losses / num_graphs

    return output_metric


@torch.no_grad()
def validate(model: Model, val_loader: DataLoader, inputs: dict) -> float:
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for data in val_loader:
        predicted_y = model(data)

        if inputs["prediction_mode"] == "coeff_matrix":
            losses += sum(coeff2density_loss(data, predicted_y, do_backward=False))
        elif inputs["prediction_mode"] == "relative_energy":
            losses += relative_energy_loss(data, predicted_y, do_backward=False)
        else:
            raise ValueError(
                f"Prediction mode {inputs['prediction_mode']} not recognized."
            )

        num_graphs += data.num_graphs

    output_metric = losses / num_graphs

    return output_metric
