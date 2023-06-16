import torch

from inspect import signature

from torch_geometric.loader import DataLoader
from torch.nn import functional as F


def construct_model_name(dataset_name: str, debug: bool = False) -> str:
    """Construct the model name based on the config filename and
    the debug flag."""

    model_name = dataset_name
    model_name += "_model"
    if debug:
        model_name += "_debug"

    return model_name


def signed_coeff_matrix_loss(data, predicted_y, loss_function, do_backward=True):
    """Get the loss when converting the coefficient matrix to the density."""

    real_y = data.x_transition_state
    batch = data.batch
    batch_size = data.num_graphs

    # If the loss_function takes only two inputs, pass it `predicted_y` and
    # `real_y`. If it takes four inputs, pass it `predicted_y`, `real_y`, and
    # `batch` and `batch_size`.
    if len(signature(loss_function.forward).parameters) == 2:
        loss = loss_function(predicted_y, real_y)
    elif len(signature(loss_function.forward).parameters) == 4:
        loss = loss_function(predicted_y, real_y, batch, batch_size)
    else:
        raise ValueError(
            f"Loss function {loss_function} has an invalid number of inputs."
        )

    if do_backward:
        loss.backward()

    return loss.item()


def relative_energy_loss(data, predicted_y, loss_function, do_backward=True):
    """Get the loss when predicting the relative energy."""

    real_y = data.total_energy_transition_state - data.total_energy
    loss = loss_function(predicted_y, real_y)

    if do_backward:
        loss.backward()

    return loss.item()


def train(
    model,
    train_loader: DataLoader,
    optim,
    prediction_mode: str,
    loss_function: torch.nn.Module,
) -> float:
    """Train the model."""

    model.train()

    losses = 0.0
    num_graphs = 0

    for idx, data in enumerate(train_loader):
        optim.zero_grad()
        predicted_y = model(data)

        if prediction_mode == "coeff_matrix":
            losses += signed_coeff_matrix_loss(data, predicted_y, loss_function)
        elif prediction_mode == "relative_energy":
            losses += relative_energy_loss(data, predicted_y, loss_function)
        else:
            raise ValueError(f"Prediction mode {prediction_mode} not recognized.")

        num_graphs += data.num_graphs
        optim.step()

    output_metric = losses / num_graphs

    return output_metric


@torch.no_grad()
def validate(
    model, val_loader: DataLoader, prediction_mode: str, loss_function: torch.nn.Module
) -> float:
    """Validate the model."""
    model.eval()

    losses = 0.0
    num_graphs = 0

    for idx, data in enumerate(val_loader):
        predicted_y = model(data)

        if prediction_mode == "coeff_matrix":
            losses += signed_coeff_matrix_loss(
                data, predicted_y, loss_function, do_backward=False
            )
        elif prediction_mode == "relative_energy":
            losses += relative_energy_loss(
                data, predicted_y, loss_function, do_backward=False
            )
        else:
            raise ValueError(f"Prediction mode {prediction_mode} not recognized.")

        num_graphs += data.num_graphs

    output_metric = losses / num_graphs

    return output_metric
