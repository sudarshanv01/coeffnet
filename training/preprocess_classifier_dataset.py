import os
import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional
import matplotlib.pyplot as plt
import argparse

import numpy as np
import numpy.typing as npt

import scipy
from scipy import special
from scipy import optimize

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from ase import io as ase_io

import torch

from monty.serialization import loadfn, dumpfn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from minimal_basis.predata import GenerateParametersClassifier


def get_cli():
    """Get the hyperparameters from the command line."""
    args = argparse.ArgumentParser()
    args.add_argument("--num_samples", type=int, default=10)
    return args.parse_args()


if __name__ == "__main__":
    """Generate parameters to determine the interpolated transition
    state based on the initial and final state complexes."""

    # Read in the training data
    rapter_data = loadfn("input_files/train_RAPTER.json")

    # Structure output folder
    structure_output_folder = os.path.join("output", "interpolated_structures")
    if not os.path.exists(structure_output_folder):
        os.makedirs(structure_output_folder)

    # Hyperparameters
    args = get_cli()

    # Extract the positions of the transition states, initial states,
    is_positions = []
    fs_positions = []
    ts_positions = []
    ts_species = []

    # Store the reaction energy as well
    reaction_energy = []

    for datapoint in rapter_data:
        reactant_structure = datapoint["reactant_structure"]
        product_structure = datapoint["product_structure"]
        transition_state_structure = datapoint["transition_state_structure"]

        # Get the cartesian coordinates of the reactant, product, and
        # transition state structures, they are all Molecules from pymatgen
        reactant_cartesian_coords = reactant_structure.cart_coords
        product_cartesian_coords = product_structure.cart_coords
        transition_state_cartesian_coords = transition_state_structure.cart_coords

        # Append the cartesian coordinates of the reactant, product, and
        # transition state structures to the appropriate lists as numpy arrays
        is_positions.append(reactant_cartesian_coords)
        fs_positions.append(product_cartesian_coords)
        ts_positions.append(transition_state_cartesian_coords)

        # Store the species names so that atoms objects can be created
        # from them later
        transition_state_species = transition_state_structure.species
        ts_species.append(transition_state_species)

        # Append the reaction energy to the reaction energy list
        reaction_energy.append(datapoint["reaction_energy"])

    # Convert the reaction energy list to a numpy array
    reaction_energy = np.array(reaction_energy)

    # Initialize the class
    paramclass = GenerateParametersClassifier(
        ts_positions=ts_positions,
        is_positions=is_positions,
        fs_positions=fs_positions,
        deltaG=reaction_energy,
        num_samples=args.num_samples,
    )

    init_mu = 0.5
    init_sigma = 4.0
    init_alpha = 1.0
    # Generate the parameters for the classifier based on minimizing
    # the output of the objective function
    parameters = scipy.optimize.minimize(
        paramclass.objective_function,
        x0=[init_mu, init_sigma, init_alpha],
    )

    # Print the parameters
    print(parameters)

    # Create a dict with all the parameters
    parameters_dict = {
        "mu": parameters.x[0],
        "sigma": parameters.x[1],
    }

    # Write out both the interpolated transition state and the
    # real transition state as xyz files
    for i, species in enumerate(ts_species):
        # Get the interpolated transition state
        interpolated_transition_state = (
            paramclass.get_interpolated_transition_state_positions(
                is_positions=is_positions[i],
                fs_positions=fs_positions[i],
                mu=parameters.x[0],
                sigma=parameters.x[1],
                alpha=parameters.x[2],
            )
        )
        # Get the real transition state
        real_transition_state = ts_positions[i]

        interpolated_ts_molecule = Molecule(species, interpolated_transition_state)
        real_ts_molecule = Molecule(species, real_transition_state)

        # Convert the molecules to ASE atoms objects
        interpolated_ts_atoms = AseAtomsAdaptor.get_atoms(interpolated_ts_molecule)
        real_ts_atoms = AseAtomsAdaptor.get_atoms(real_ts_molecule)
        # Write out the interpolated transition state

        ase_io.write(
            f"{structure_output_folder}/interpolated_transition_state_{i}.xyz",
            [interpolated_ts_atoms, real_ts_atoms],
        )

    #   Save the parameters to a json file
    with open("input_files/classifier_parameters.json", "w") as f:
        json.dump(parameters_dict, f)
