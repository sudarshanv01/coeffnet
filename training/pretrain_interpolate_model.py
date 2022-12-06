import os
import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import pickle

import numpy as np
import numpy.typing as npt

import scipy
from scipy import special
from scipy import optimize

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from sklearn.kernel_ridge import KernelRidge

from ase import io as ase_io

import torch

from monty.serialization import loadfn, dumpfn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from minimal_basis.predata import GenerateParametersClassifier


def get_cli():
    """Get the hyperparameters from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Hyperparamer for the number of samples to generate",
    )
    parser.add_argument("--restart", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=1.0, help="Hyperparameter alpha")
    parser.add_argument("--mu", type=float, default=0.5, help="Hyperparameter mu")
    parser.add_argument("--sigma", type=float, default=2, help="Hyperparameter sigma")
    return parser.parse_args()


if __name__ == "__main__":
    """Generate parameters to determine the interpolated transition
    state based on the initial and final state complexes."""

    # Individually get the parameters and hyperparameters
    args = get_cli()

    # Read in the training data
    rapter_data = loadfn("input_files/train_RAPTER.json")

    # Structure output folder
    structure_output_folder = os.path.join("output", "interpolated_structures")
    if not os.path.exists(structure_output_folder):
        os.makedirs(structure_output_folder)

    # Store all the data in lists
    data_to_store = defaultdict(list)

    # Initialize the class
    paramclass = GenerateParametersClassifier(num_samples=args.num_samples)

    for datapoint in rapter_data:

        reactant_graph = datapoint["reactant_molecule_graph"]
        product_graph = datapoint["product_molecule_graph"]
        transition_state_graph = datapoint["transition_state_molecule_graph"]

        transition_state_coords = transition_state_graph.molecule.cart_coords
        data_to_store["real_transition_state_coords"].append(transition_state_coords)

        reaction_energy = datapoint["reaction_energy"]

        # Get the interpolated transition states
        logger.info("Generating the interpolated transition states")
        interpolated_transition_states = (
            paramclass.get_interpolated_transition_state_positions(
                is_positions=reactant_graph.molecule.cart_coords,
                fs_positions=product_graph.molecule.cart_coords,
                mu=args.mu,
                sigma=args.sigma,
                alpha=args.alpha,
                deltaG=reaction_energy,
            )
        )

        # Get a list of the atomic numbers
        atomic_numbers = np.array(transition_state_graph.molecule.atomic_numbers)

        # The features to the Kernel ridge model is simply the interpolated
        # transitions state coords and the atom numbers of the species
        # of the transition state
        input_features = np.concatenate(
            (interpolated_transition_states, atomic_numbers.reshape(-1, 1)), axis=1
        )

        # Store the input features
        data_to_store["input_features"].append(input_features)

    # Kernel ridge model to predict the transition state
    logger.info("Training the kernel ridge model")
    kernel_ridge_model = KernelRidge(kernel="rbf")
    kernel_ridge_model.fit(
        np.concatenate(data_to_store["input_features"], axis=0),
        np.concatenate(data_to_store["real_transition_state_coords"], axis=0),
    )

    # Write out the hyperparameters
    with open("output/hyperparameters.json", "w") as f:
        json.dump(
            {
                "alpha": args.alpha,
                "mu": args.mu,
                "sigma": args.sigma,
            },
            f,
        )

    # Save the model as a pickle file
    with open("output/kernel_ridge_model.pickle", "wb") as f:
        pickle.dump(kernel_ridge_model, f)
