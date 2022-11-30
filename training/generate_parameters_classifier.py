import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import numpy.typing as npt

import scipy
from scipy import special
from scipy import optimize

import torch

from monty.serialization import loadfn, dumpfn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from minimal_basis.predata import GenerateParametersClassifier


if __name__ == "__main__":
    """Generate parameters to determine the interpolated transition
    state based on the initial and final state complexes."""

    # Read in the training data
    rapter_data = loadfn("input_files/train_RAPTER.json")

    # Extract the positions of the transition states, initial states,
    is_positions = []
    fs_positions = []
    ts_positions = []

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
        num_data_points=len(rapter_data),
    )

    # Generate the parameters for the classifier based on minimizing
    # the output of the objective function
    parameters = scipy.optimize.minimize(
        paramclass.objective_function,
        x0=[0.1],
    )

    # Print the parameters
    print(parameters)

    # Create a dict with all the parameters
    parameters_dict = {
        "alpha": parameters.x[0],
        "mu": None,
        "sigma": paramclass.sigma.tolist(),
    }

    #   Save the parameters to a json file
    with open("input_files/classifier_parameters.json", "w") as f:
        json.dump(parameters_dict, f)
