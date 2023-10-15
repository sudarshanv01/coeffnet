
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pymatgen.core.structure import Molecule

import pandas as pd

from instance_mongodb import instance_mongodb_sei


if __name__ == "__main__":
    """Make a potential energy surface plot for the difficult transition state."""

    db = instance_mongodb_sei(project="mlts")

    collection = db.visualize_difficult_ts_calculation

    find_tags = {"tags.group": "visualize_difficult_transition_state"}

    # Create an empty dataframe to store the data.
    df = pd.DataFrame()

    for idx, document in enumerate(collection.find(find_tags)):

        molecule = document['output']['initial_molecule']
        molecule = Molecule.from_dict(molecule)
        energy = document['output']['final_energy']

        positions = [a.coords for a in molecule.sites]

        # Get the distance between the (0,2) and (0,3) atoms.
        distance_02 = np.linalg.norm(np.array(positions[0]) - np.array(positions[3]))
        distance_03 = np.linalg.norm(np.array(positions[0]) - np.array(positions[2]))

        data_to_concat = pd.DataFrame(
            {
                "distance_02": distance_02,
                "distance_03": distance_03,
                "energy": energy,
            },
            index=[idx],
        )

        df = pd.concat([df, data_to_concat])

    # Make a 2D contour plot of the potential energy surface.
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

    # Get the data.
    x = df["distance_02"]
    y = df["distance_03"]
    z = df["energy"]

    # Make an imshow plot
    im = ax.tricontourf(x, y, z, levels=100, cmap="viridis")
    # Add a colorbar.
    fig.colorbar(im, ax=ax, label="Energy (eV)")


    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Distance (Å)")

    fig.savefig("output/visualize_difficult_transition_state.png", dpi=300)
