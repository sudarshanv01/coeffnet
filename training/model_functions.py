import torch

from torch_geometric.loader import DataLoader
from torch.nn import functional as F

from minimal_basis.dataset.reaction import ReactionDataset as Dataset
from minimal_basis.loss.eigenvectors import (
    UnsignedMSELoss,
    UnsignedL1Loss,
    UnsignedDotProductPreservingMSELoss,
    UnsignedDotProductPreservingL1Loss,
)


def construct_model_name(dataset_name: str, debug: bool = False) -> str:
    """Construct the model name based on the config filename and
    the debug flag."""

    model_name = dataset_name
    model_name += "_model"
    if debug:
        model_name += "_debug"

    return model_name


def construct_irreps(
    model_options: dict,
) -> None:
    """Construct the inputs if there is an @construct in the inputs.

    Args:
        model_options (dict): The model options.
        dataset_options (dict): The dataset options.
        prediction_mode (str): The prediction mode.

    """

    if model_options["irreps_edge_attr"] == "@construct":
        model_options["irreps_edge_attr"] = f"{model_options['num_basis']}x0e"


def signed_coeff_matrix_loss(data, predicted_y, do_backward=True):
    """Get the loss when converting the coefficient matrix to the density."""

    real_y = data.x_transition_state
    # batch = data.batch
    # batch_size = data.num_graphs

    # loss = UnsignedDotProductPreservingL1Loss()(
    #     predicted_y, real_y, batch, batch_size, reduction="sum"
    # )
    loss = F.l1_loss(predicted_y.abs(), real_y.abs(), reduction="sum")
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


def train(model, train_loader: DataLoader, optim, prediction_mode: str) -> float:
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for idx, data in enumerate(train_loader):
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
def validate(model, val_loader: DataLoader, prediction_mode: str) -> float:
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for idx, data in enumerate(val_loader):
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
