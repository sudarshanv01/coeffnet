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


class GenerateParametersClassifier:
    def __init__(
        self,
        is_positions: List[Union[npt.ArrayLike, torch.Tensor]],
        fs_positions: List[Union[npt.ArrayLike, torch.Tensor]],
        deltaG: Union[npt.ArrayLike, torch.Tensor],
        num_data_points: int,
        ts_positions: List[Union[npt.ArrayLike, torch.Tensor]] = None,
    ):
        """Generate parameters for the classifier.

        Args:
            is_positions: The positions of the initial states.
            fs_positions: The positions of the final states.
            deltaG: The free energy difference between the initial
                and final states.
            num_data_points (int): The number of data points.
            ts_positions: The positions of the transition states.

        """
        self.logger = logging.getLogger(__name__)

        # Decide based on the inputs if we are using numpy or torch
        if isinstance(ts_positions[0], torch.Tensor):
            self.use_torch = True
            self.lib = torch
        else:
            self.use_torch = False
            self.lib = np

        self.ts_positions = ts_positions
        self.num_data_points = num_data_points
        self.deltaG = deltaG
        self.is_positions = is_positions
        self.fs_positions = fs_positions
        self.validate_parameters()

        # Every instance of this class generates
        # a completely random set of sigma parameters
        self.sigma = self.lib.random.randn(self.num_data_points)

    def f(self, x: Union[npt.ArrayLike, torch.Tensor], alpha: float) -> npt.ArrayLike:
        """Return f based on parameter x and hyperparameter alpha."""

        if self.use_torch:
            return 0.5 * (1 + torch.erf(alpha * x / (self.lib.sqrt(2))))
        else:
            return 0.5 * (1 + scipy.special.erf(alpha * x / (self.lib.sqrt(2))))

    def h(
        self, x: Union[npt.ArrayLike, torch.Tensor], mu: float = None, sigma: float = 1
    ) -> npt.ArrayLike:
        """Return a normal distribution."""

        if mu is None:
            # If mu is not provided, use a centered normal distribution
            mu = 0.0

        return self.lib.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
            sigma * self.lib.sqrt(2 * self.lib.pi)
        )

    def get_interpolated_ts_positions(self, alpha: float, mu: float = None):
        """Generate parameters for the classifier."""
        # generate the deciding parameter of "initial-state-like" or
        # "final-state-like" for each data point based on the
        # value of Delta G and the distribution of Delta G
        f = self.f(self.deltaG, alpha)
        h = self.h(self.deltaG, mu=mu, sigma=self.sigma)

        # generate the probability of each data point being
        # "initial-state-like" or "final-state-like"
        p = (f + h) / 2
        assert 0 <= p.all() <= 1, "p must be between 0 and 1"
        p_prime = self.lib.ones_like(p) - p

        # Generate the interpolated transition state positions
        # based on the probability of each data point being
        # "initial-state-like" or "final-state-like"
        int_ts_positions = []
        for i in range(self.num_data_points):
            int_ts_positions.append(
                p[i] * self.is_positions[i] + p_prime[i] * self.fs_positions[i]
            )

        return int_ts_positions

    def objective_function(self, alpha: float, mu: float = None):
        """Generate parameters for the classifier."""

        # Make sure we are using the numpy library
        assert not self.use_torch, "This function is only for numpy"

        int_ts_positions = self.get_interpolated_ts_positions(alpha, mu=mu)

        # Calculate the mean squared error between the
        # interpolated transition state positions and the
        # actual transition state positions
        # Start by generating the norm of the difference
        # between the interpolated transition state positions
        # and the actual transition state positions
        norm = np.linalg.norm(
            np.array(int_ts_positions) - np.array(self.ts_positions), axis=1
        )

        # Calculate the mean squared error
        mse = np.mean(norm**2)
        rmse = np.sqrt(mse)
        self.logger.debug("rmse: %s", rmse)

        return rmse


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
        "sigma": paramclass.sigma,
    }

    #   Save the parameters to a json file
    with open("input_files/classifier_parameters.json", "w") as f:
        json.dump(parameters_dict, f)
