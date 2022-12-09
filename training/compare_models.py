import matplotlib.pyplot as plt
import seaborn as sns

import wandb

api = wandb.Api()

if __name__ == "__main__":

    run_diffclassifier = api.run("sudarshanvj/diffclassifier_model/2wsxvvd9")
    metrics_diffclassifier = run_diffclassifier.history()

    run_interpolate = api.run("sudarshanvj/interpolate_model/3vv6wjxy")
    metrics_interpolate = run_interpolate.history()

    run_diffinterpolate = api.run("sudarshanvj/interpolate_diff_model/2dv4uhle")
    metrics_diffinterpolate = run_diffinterpolate.history()

    fig, ax = plt.subplots(1, 3, figsize=(6.75, 3.0), constrained_layout=True)

    # Make a seaborn lineplot for train_acc vs _step
    # in blue and val_acc vs _step in red
    sns.lineplot(
        x="_step",
        y="train_acc",
        data=metrics_diffclassifier,
        ax=ax[0],
        color="tab:blue",
        label="Training",
    )
    sns.lineplot(
        x="_step",
        y="val_acc",
        data=metrics_diffclassifier,
        ax=ax[0],
        color="tab:red",
        label="Validation",
    )

    # Make a seaborn lineplot for train_loss and val_loss vs _step
    sns.lineplot(
        x="_step",
        y="train_loss",
        data=metrics_interpolate,
        ax=ax[1],
        color="tab:blue",
    )
    sns.lineplot(
        x="_step", y="val_loss", data=metrics_interpolate, ax=ax[1], color="tab:red"
    )

    sns.lineplot(
        x="_step",
        y="train_loss",
        data=metrics_diffinterpolate,
        ax=ax[2],
        color="tab:blue",
    )
    sns.lineplot(
        x="_step", y="val_loss", data=metrics_diffinterpolate, ax=ax[2], color="tab:red"
    )

    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Step")
    ax[0].set_title("diff-classifier")
    ax[1].set_ylabel("Loss (eV)")
    ax[1].set_xlabel("Step")
    ax[1].set_title("interpolate")
    ax[2].set_ylabel("Loss (eV)")
    ax[2].set_xlabel("Step")
    ax[2].set_title("diff-interpolate")
    # Set limit for y-axis
    ax[1].set_ylim(0, 5)
    ax[2].set_ylim(0, 5)

    # Save the figure
    fig.savefig("output/compare_models.png", dpi=300)
