import wandb

api = wandb.Api()

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from plot_params import get_plot_params

get_plot_params()

from ase import units


if __name__ == "__main__":
    """Plot the variation of the validation loss with the fraction of the training data used."""

    # Get the runs
    runs = api.runs("sudarshanvj/rudorff_lilienfeld_sn2_dataset_model_datatesting")

    # Store everything in a dataframe
    df = pd.DataFrame(columns=["Fraction of training data", "Validation loss"])

    for run in runs:

        # Get the run summary
        summary = run.summary
        config = run.config

        # Get the fraction of training data
        fraction = config["limit_ratio"]

        # Get the validation loss
        validation_loss = summary["val_loss"]

        # Concatenate the data
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[fraction, validation_loss]],
                    columns=["fraction", "val_loss"],
                ),
            ]
        )

    # Multiply the validation loss by the unit
    df["val_loss"] *= units.Hartree

    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

    # Make a scatter plot
    sns.scatterplot(
        data=df,
        x="fraction",
        y="val_loss",
        ax=ax,
        color="black",
    )

    # Set the x and y labels
    ax.set_xlabel("Fraction of training data")
    ax.set_ylabel("Validation loss (eV)")

    fig.savefig("output/learning_curve.png", dpi=300)
    fig.savefig("output/learning_curve.pdf", dpi=300)
