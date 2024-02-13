import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from plot_params import get_plot_params

get_plot_params()

from ase import units


def arhenius_rate(T, Ga):
    hbar = 6.582119569e-16  # eV s
    A = units.kB * T / hbar
    return A * np.exp(-Ga / (units.kB * T))


if __name__ == "__main__":
    """Plot the sensitivity of the barrier to the temperature."""

    temperatures = [300, 400, 500, 600, 700, 800, 900, 1000]
    colors = sns.color_palette("coolwarm", len(temperatures))
    Ga_range = np.linspace(0, 1.5, 100)
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), constrained_layout=True)

    for idx, temperature in enumerate(temperatures):

        rate = arhenius_rate(temperature, Ga_range)
        rate_error_positive = arhenius_rate(temperature, Ga_range + 0.1)
        rate_error_negative = arhenius_rate(temperature, Ga_range - 0.1)

        ax.plot(Ga_range, rate, color=colors[idx], label=f"{temperature} K")
        ax.fill_between(
            Ga_range,
            rate_error_positive,
            rate_error_negative,
            color=colors[idx],
            alpha=0.2,
        )

    ax.set_xlabel(r"$G_a$ (eV)")
    ax.set_ylabel(r"Rate (s$^{-1}$)")
    ax.set_yscale("log")
    ax.legend(loc="best", frameon=False)

    plt.savefig("output/sensitivity_of_barrier.png", dpi=300)
