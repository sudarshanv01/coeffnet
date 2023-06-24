import argparse

import pandas as pd

import wandb

wandb_api = wandb.Api()

import plotly.express as px

from model_functions import construct_model_name


def get_cli_arguments():
    """Get command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        default="rudorff_lilienfeld_sn2_dataset",
        help="Name of the dataset to train on.",
    )
    arg_parser.add_argument(
        "--basis_set_type",
        type=str,
        default="full",
        help="Type of basis set to use.",
    )
    arg_parser.add_argument(
        "--basis_set",
        type=str,
        default="def2-svp",
        help="Basis set to use.",
    )
    arg_parser.add_argument(
        "--prediction_mode",
        type=str,
        default="coeff_matrix",
        help="Type of prediction to make.",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    """Pull runs from the W&B API and analyze the hyperparameters using a
    parallel coordinates plot."""

    args = get_cli_arguments()

    hyperparameters_to_analyze = [
        "max_radius",
        "radial_neurons",
        "layers",
        "num_basis",
        "mul",
    ]

    model_name = construct_model_name(
        dataset_name=args.dataset_name,
        debug=False,
    )
    runs = wandb_api.runs(f"sudarshanvj/{model_name}")
    df = pd.DataFrame()
    for run in runs:
        if run.state != "finished":
            continue
        if (
            run.config.get("basis_set_type") == args.basis_set_type
            and run.config.get("basis_set") == args.basis_set
            and run.config.get("prediction_mode") == args.prediction_mode
        ):
            data_to_store = {}
            data_to_store.update(run.config)
            train_loss = run.summary.get("train_loss", None)
            val_loss = run.summary.get("val_loss", None)
            data_to_store.update({"train_loss": train_loss, "val_loss": val_loss})
            data_to_store.update({"wandb_model_name": run.name})
            df = pd.concat(
                [df, pd.DataFrame(data_to_store, index=[0])], ignore_index=True
            )

    df = df.sort_values(by=["val_loss"])

    # Plot a parallel coordinates plot
    fig = px.parallel_coordinates(
        df,
        color="val_loss",
        dimensions=hyperparameters_to_analyze + ["val_loss"],
        color_continuous_scale=px.colors.diverging.Tealrose,
        template="simple_white",
        color_continuous_midpoint=df["val_loss"].median(),
    )
    # Add a color bar which maps values to colors
    fig.update_layout(coloraxis_colorbar=dict(title="Validation Loss"))
    # For the best visualization, we want to have the best models at the top
    fig.update_yaxes(autorange="reversed")

    # Lower dimensions of the plot
    fig.update_layout(
        height=800,
        width=1200,
        font=dict(size=18),
    )
    fig.show()
