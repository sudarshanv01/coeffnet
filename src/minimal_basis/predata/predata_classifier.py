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

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)

logger = logging.getLogger(__name__)


class GenerateParametersClassifier:
    def __init__(
        self,
        is_positions: List[Union[npt.ArrayLike, torch.Tensor]],
        fs_positions: List[Union[npt.ArrayLike, torch.Tensor]],
        deltaG: Union[npt.ArrayLike, torch.Tensor],
        num_data_points: int,
        ts_positions: List[Union[npt.ArrayLike, torch.Tensor]] = None,
        sigma: List[Union[npt.ArrayLike, torch.Tensor]] = None,
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
        if isinstance(deltaG, torch.Tensor):
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

        # Every instance of this class generates
        # a completely random set of sigma parameters
        # where all sigma parameters are between 0.1 and 1.0
        if sigma is None:
            self.sigma = np.random.uniform(0.1, 1.0, size=self.num_data_points)
            if self.use_torch:
                self.sigma = torch.tensor(self.sigma, dtype=DTYPE)
        else:
            self.sigma = sigma
            # Make self.sigma a numpy array or a torch tensor depending on the input
            if self.use_torch:
                self.sigma = torch.tensor(self.sigma, dtype=DTYPE)
            else:
                self.sigma = np.array(self.sigma)
        self.logger.debug(f"Shape of sigma: {self.sigma.shape}")

    def f(self, x: Union[npt.ArrayLike, torch.Tensor], alpha: float) -> npt.ArrayLike:
        """Return f based on parameter x and hyperparameter alpha."""

        if self.use_torch:
            return 0.5 * (1 + torch.erf(alpha * x / (self.lib.sqrt(torch.tensor(2.0)))))
        else:
            return 0.5 * (1 + scipy.special.erf(alpha * x / (self.lib.sqrt(2))))

    def h(
        self, x: Union[npt.ArrayLike, torch.Tensor], mu: float = None, sigma: float = 1
    ) -> npt.ArrayLike:
        """Return a normal distribution."""

        if mu is None:
            # If mu is not provided, use a centered normal distribution
            mu = 0.0

        return self.lib.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def get_p_and_pprime(
        self, alpha: float, mu: float = None
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        # generate the deciding parameter of "initial-state-like" or
        # "final-state-like" for each data point based on the
        # value of Delta G and the distribution of Delta G
        f = self.f(self.deltaG, alpha)
        self.logger.debug(f"Shape of f: {f.shape}")
        h = self.h(self.deltaG, mu=mu, sigma=self.sigma)
        self.logger.debug(f"Shape of h: {h.shape}")

        # generate the probability of each data point being
        # "initial-state-like" or "final-state-like"
        p = (f + h) / 2
        p_prime = self.lib.ones_like(p) - p
        self.logger.debug(f"Shape of p: {p.shape}")

        return p, p_prime

    def get_interpolated_ts_positions(self, alpha: float, mu: float = None):
        """Generate parameters for the classifier."""

        p, p_prime = self.get_p_and_pprime(alpha, mu=mu)

        # Generate the interpolated transition state positions
        # based on the probability of each data point being
        # "initial-state-like" or "final-state-like"
        int_ts_positions = []
        for i in range(self.num_data_points):
            int_ts_positions.append(
                p[i] * self.is_positions[i] + p_prime[i] * self.fs_positions[i]
            )
        self.logger.debug(f"Shape of int_ts_positions: {len(int_ts_positions)}")

        return int_ts_positions

    def objective_function(self, alpha: float, mu: float = None):
        """Generate parameters for the classifier."""

        # Make sure we are using the numpy library
        assert not self.use_torch, "This function is only for numpy"

        int_ts_positions = self.get_interpolated_ts_positions(alpha, mu=mu)

        # Calculate the difference between the interpolated
        # transition state positions and the actual transition
        square_error = []
        for i in range(self.num_data_points):
            _square_error = np.linalg.norm(
                int_ts_positions[i] - self.ts_positions[i], axis=1
            )
            square_error.append(np.sum(_square_error**2) / len(_square_error))
        square_error = np.array(square_error)
        self.logger.debug(f"Shape of square_error: {square_error.shape}")

        # Calculate the mean squared error
        rmse = np.sqrt(np.mean(square_error))
        self.logger.debug("rmse: %s", rmse)

        return rmse
