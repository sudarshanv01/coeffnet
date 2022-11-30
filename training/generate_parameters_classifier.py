import logging

import numpy as np
import numpy.typing as npt

import scipy

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerateParametersClassifier:
    def __init__(
        self,
        ts_positions: npt.ArrayLike,
        is_positions: npt.ArrayLike,
        fs_positions: npt.ArrayLike,
        deltaG: npt.ArrayLike,
        num_data_points: int,
    ):
        """Generate parameters for the classifier.

        Args:
            num_data_points: Number of data points for which labelled data exists.

        """
        self.logger = logging.getLogger(__name__)
        self.ts_positions = ts_positions
        self.num_data_points = num_data_points
        self.deltaG = deltaG
        self.is_positions = is_positions
        self.fs_positions = fs_positions
        self.validate_parameters()

        # Every instance of this class generates
        # a completely random set of sigma parameters
        self.sigma = np.random.uniform(0.1, 1, self.num_data_points)

    def validate_parameters(self):
        """Validate the input parameters to the class."""
        if self.ts_positions is None:
            raise ValueError("ts_positions cannot be None")
        # Make sure ts_positions is a numpy array of shape
        # (num_data_points, 3)
        assert isinstance(
            self.ts_positions, np.ndarray
        ), "ts_positions must be a numpy array"
        assert self.ts_positions.shape == (
            self.num_data_points,
            3,
        ), "ts_positions must be of shape (num_data_points, 3)"

        # Make sure the deltaG is a numpy array of shape
        # (num_data_points, 1)
        assert isinstance(self.deltaG, np.ndarray), "deltaG must be a numpy array"
        assert self.deltaG.shape == (
            self.num_data_points,
            1,
        ), "deltaG must be of shape (num_data_points, 1)"

        # Make sure the is_positions is a numpy array of shape
        # (num_data_points, 3)
        assert isinstance(
            self.is_positions, np.ndarray
        ), "is_positions must be a numpy array"
        assert self.is_positions.shape == (
            self.num_data_points,
            3,
        ), "is_positions must be of shape (num_data_points, 3)"

        # Make sure the fs_positions is a numpy array of shape
        # (num_data_points, 3)
        assert isinstance(
            self.fs_positions, np.ndarray
        ), "fs_positions must be a numpy array"
        assert self.fs_positions.shape == (
            self.num_data_points,
            3,
        ), "fs_positions must be of shape (num_data_points, 3)"

    def f(self, x: npt.ArrayLike, alpha: float) -> npt.ArrayLike:
        """Return f based on parameter x and hyperparameter alpha."""
        return 0.5 * (1 + scipy.special.erf((alpha * x) / np.sqrt(2)))

    def h(self, x: npt.ArrayLike, mu: float = None, sigma: float = 1) -> npt.ArrayLike:
        """Return a normal distribution."""
        if mu is None:
            # If mu is not provided, use a centered normal distribution
            mu = 0.0
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
            sigma * np.sqrt(2 * np.pi)
        )

    def objective_function(self, alpha: float, mu: float = None):
        """Generate parameters for the classifier."""
        self.logger.debug("alpha: %s", alpha)
        self.logger.debug("mu: %s", mu)

        # generate the deciding parameter of "initial-state-like" or
        # "final-state-like" for each data point based on the
        # value of Delta G and the distribution of Delta G
        f = self.f(self.deltaG, alpha)
        self.logger.debug("f: %s", f)
        h = self.h(self.deltaG, mu=mu, sigma=self.sigma)
        self.logger.debug("h: %s", h)

        # generate the probability of each data point being
        # "initial-state-like" or "final-state-like"
        p = (f + h) / 2
        assert 0 <= p.all() <= 1, "p must be between 0 and 1"

        # Generate the interpolated transition state positions
        # based on the probability of each data point being
        # "initial-state-like" or "final-state-like"
        int_ts_positions = p * self.is_positions + (1 - p) * self.fs_positions

        # Minimum distance between the interpolated transition state
        # positions and the actual transition state positions
        min_dist = np.min(np.linalg.norm(int_ts_positions - self.ts_positions, axis=1))

        # Return the minimum distance between the interpolated transition state
        return min_dist


if __name__ == "__main__":
    """Generate parameters to determine the interpolated transition
    state based on the initial and final state complexes."""
