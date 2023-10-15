from pathlib import Path

import wandb

wandb_api = wandb.Api()

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from plot_params import get_plot_params

get_plot_params()


if __name__ == "__main__":
    """Plot the training curves for the best performing coeff matrix model."""

    run = wandb_api.run("sudarshanvj/rudorff_lilienfeld_sn2_dataset_model/65gkbt2n")

    fig, ax = plt.subplots(1, 1, figsize=(2.75, 2), constrained_layout=True)

    df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    __output_dir__ = Path("output")

    for i, row in run.history().iterrows():
        val_loss = row["val_loss"]
        train_loss = row["train_loss"]

        data_to_concat = {
            "epoch": i / 2,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        df = pd.concat([df, pd.DataFrame(data_to_concat, index=[0])], ignore_index=True)

    print(df)

    sns.lineplot(
        data=df,
        x="epoch",
        y="train_loss",
        ax=ax,
        label="Train",
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y="val_loss",
        ax=ax,
        label="Validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig(__output_dir__ / "training_curves.pdf")
    fig.savefig(__output_dir__ / "training_curves.png", dpi=300)
