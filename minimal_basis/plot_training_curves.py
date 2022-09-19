"""Plot training curves from a log file."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from plot_params import get_plot_params

get_plot_params()

if __name__ == "__main__":
    """For a gives training log file, plot the training curves."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("--save", type=str, help="Path to save the plot.")
    args = parser.parse_args()

    if args.save is None:
        logfilename = os.path.basename(args.log_file)
        savefile = os.path.join("plots", logfilename.replace(".log", ".png"))

    # Read the log file.
    log_data = np.loadtxt(args.log_file, delimiter="\t", skiprows=1)

    # The first column is the epoch number, the second is the
    # loss and the third is the accuracy.

    # Plot the loss and accuracy curves.
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, constrained_layout=True)
    ax[0].plot(log_data[:, 0], log_data[:, 1])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("RMSE Loss (eV)")
    ax[1].plot(log_data[:, 0], log_data[:, 2])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Mean absolute error (eV)")
    fig.savefig(savefile, dpi=300)
