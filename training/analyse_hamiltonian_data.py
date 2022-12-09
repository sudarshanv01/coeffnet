import os
from collections import defaultdict

import numpy as np

import pandas as pd

from monty.serialization import loadfn, dumpfn

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    """Analyse the Hamiltonian dataset."""

    # Load the data
    train_data = loadfn(os.path.join("input_files", "train_QMrxn20.json"))
    validate_data = loadfn(os.path.join("input_files", "validate_QMrxn20.json"))

    # Create a dataframe to store the data
    df = pd.DataFrame(
        columns=["label", "reaction energy", "reaction barrier", "descirptor"]
    )

    for data in train_data:

        reactant_alpha_fock_matrix = np.array(data["reactant_alpha_fock_matrix"])
        product_alpha_fock_matrix = np.array(data["product_alpha_fock_matrix"])
        transition_state_alpha_fock_matrix = np.array(
            data["transition_state_alpha_fock_matrix"]
        )

        # Subtract the transition state from the reactant
        descriptor_matrix = (
            transition_state_alpha_fock_matrix - reactant_alpha_fock_matrix
        )
        # Make every element of the matrix positive
        # descriptor_matrix = np.abs(descriptor_matrix)
        # Find the norm of the descriptor matrix
        # descriptor = np.linalg.norm(descriptor_matrix)
        # Find the minimum value of the descriptor matrix
        descriptor = np.min(descriptor_matrix)

        data_to_store = {
            "label": data["label"],
            "reaction energy": data["reactant_product_energy_difference"],
            "reaction barrier": data["transition_state_reactant_energy_difference"],
            "descriptor": descriptor,
        }

        # Concatenate the data to the dataframe
        df = pd.concat([df, pd.DataFrame(data_to_store, index=[0])], ignore_index=True)

    # Plot the reaction energy vs. reaction barrier where the color of the points
    # is determined by the descriptor
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    sns.scatterplot(
        x="descriptor", y="reaction barrier", hue="reaction energy", data=df, ax=ax
    )

    # Save the figure
    fig.savefig(
        os.path.join("output", "sandbox_reaction_energy_vs_reaction_barrier.png"),
        dpi=300,
    )
