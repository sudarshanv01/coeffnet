import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import numpy.typing as npt

import scipy
from scipy import special
from scipy import optimize

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

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
logging.basicConfig(level=logging.INFO)


class GenerateParametersInterpolator:
    def __init__(
        self,
        reactant_graphs: List[MoleculeGraph] = None,
        product_graphs: List[MoleculeGraph] = None,
        transition_state_graphs: List[MoleculeGraph] = None,
        deltaG: Union[npt.ArrayLike, torch.Tensor] = None,
        grid_size: int = 1000,
        num_samples: int = 10,
        upper: float = 1.0,
        lower: float = 0.0,
    ):
        """Generate parameters for the classifier."""
        self.logger = logging.getLogger(__name__)

        # Decide based on the inputs if we are using numpy or torch
        if isinstance(deltaG, torch.Tensor):
            self.use_torch = True
            self.lib = torch
        else:
            self.use_torch = False
            self.lib = np
        self.logger.debug("Using %s", self.lib)

        self.reactant_graphs = reactant_graphs
        self.product_graphs = product_graphs
        self.transition_state_graphs = transition_state_graphs

        self.deltaG = deltaG

        self.grid_size = grid_size
        self.num_samples = num_samples
        self.upper = upper
        self.lower = lower

    def normal_distribution(
        self, x: Union[float, torch.tensor, npt.ArrayLike], mu: float, sigma: float
    ):
        """Construct a normal distribution."""
        if self.use_torch:
            return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
                sigma * torch.sqrt(torch.tensor(2.0) * torch.pi)
            )
        else:
            return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
                sigma * np.sqrt(2 * np.pi)
            )

    def skew_normal_distribution(
        self,
        x: Union[float, torch.tensor, npt.ArrayLike],
        mu: float,
        sigma: float,
        alpha: float,
    ):
        """Construct a skewed normal distribution."""
        cdf = self.cumulative_distribution(x, mu, sigma, alpha)
        normal = self.normal_distribution(x, mu, sigma)
        return 2 * normal * cdf

    def cumulative_distribution(
        self,
        x: Union[float, torch.tensor, npt.ArrayLike],
        mu: float,
        sigma: float,
        alpha: float,
    ):
        """Construct a cumulative distribution."""
        if self.use_torch:
            return 0.5 * (
                1 + torch.erf(alpha * (x - mu) / (sigma * torch.sqrt(torch.tensor(2))))
            )
        else:
            return 0.5 * (
                1 + scipy.special.erf(alpha * (x - mu) / (sigma * np.sqrt(2)))
            )

    def truncated_normal_distribution(
        self,
        x: Union[float, torch.tensor, npt.ArrayLike, float],
        mu: float,
        sigma: float,
        lower: float,
        upper: float,
    ):
        """Get a truncated normal distribution."""

        truncated_normal_numerator = self.normal_distribution(x, mu, sigma)
        truncated_normal_denominator = self.cumulative_distribution(
            upper, mu, sigma, alpha=1.0
        ) - self.cumulative_distribution(lower, mu, sigma, alpha=1.0)
        truncated_normal = truncated_normal_numerator / truncated_normal_denominator

        # Set the values outside the range to zero
        truncated_normal[x < lower] = 0
        truncated_normal[x > upper] = 0
        return truncated_normal

    def truncated_skew_normal_distribution(
        self,
        x: Union[float, torch.tensor, npt.ArrayLike, float],
        mu: float,
        sigma: float,
        alpha: float,
        lower: float,
        upper: float,
    ):
        """Get a truncated normal distribution."""

        truncated_normal_numerator = self.skew_normal_distribution(x, mu, sigma, alpha)
        truncated_normal_denominator = self.cumulative_distribution(
            upper, mu, sigma, alpha=1.0
        ) - self.cumulative_distribution(lower, mu, sigma, alpha=1.0)
        truncated_normal = truncated_normal_numerator / truncated_normal_denominator

        # Set the values outside the range to zero
        truncated_normal[x < lower] = 0
        truncated_normal[x > upper] = 0
        return truncated_normal

    def get_sampled_distribution(
        self,
        mu: float,
        sigma: float,
        alpha: float,
        num_samples: int = 10,
        upper: float = 1.0,
        lower: float = 0.0,
    ):
        """Get points sampled from the truncated normal distribution."""

        # Get the grid of points and the truncated normal distribution on
        # the grid
        x_grid = self.lib.linspace(lower, upper, self.grid_size)
        tnd = self.truncated_skew_normal_distribution(
            x_grid, mu, sigma, alpha, lower, upper
        )
        # Get the cumulative distribution of the truncated normal distribution
        if not self.use_torch:
            cdf = self.lib.cumsum(tnd) / self.lib.sum(tnd)
        else:
            cdf = self.lib.cumsum(tnd, dim=0) / self.lib.sum(tnd)

        # Choose a random number between 0 and 1
        if not self.use_torch:
            random_numbers = self.lib.random.rand(num_samples)
        else:
            random_numbers = torch.rand(num_samples)

        # Generate samples from the truncated normal distribution
        if not self.use_torch:
            sample = np.interp(random_numbers, cdf, x_grid)
        else:
            # Use the numpy version of interp for torch
            sample = np.interp(random_numbers, cdf.cpu().numpy(), x_grid.cpu().numpy())
            sample = torch.tensor(sample, dtype=torch.float32)

        return sample

    def get_p_and_pprime(
        self,
        mu: float,
        sigma: float,
        alpha: float,
        upper: float = 1.0,
        lower: float = 0.0,
    ):
        """Get the p and p' values, used to multiply the initial and final state positions
        to get the interpolated transition state positions."""
        # Sample from the truncated skew normal distribution for the interpolated
        # position of the transition state
        sample = self.get_sampled_distribution(
            mu, sigma, alpha, num_samples=self.num_samples, upper=upper, lower=lower
        )
        # Get the interpolated ts positions based on the average of the samples
        p = self.lib.mean(sample) - lower
        p_prime = upper - p
        return p, p_prime

    def get_interpolated_transition_state_positions(
        self,
        is_positions: Union[npt.ArrayLike, torch.tensor],
        fs_positions: Union[npt.ArrayLike, torch.tensor],
        mu: float,
        sigma: float,
        alpha: float,
        deltaG: float,
        upper: float = 1.0,
        lower: float = 0.0,
    ):
        """Get the interpolated transition state positions."""
        # The interpolated TS positions will be a linear combination of the initial and final state positions
        p, p_prime = self.get_p_and_pprime(mu, sigma, alpha * deltaG, upper, lower)
        int_ts_positions = p * fs_positions + p_prime * is_positions
        return int_ts_positions

    def objective_function(
        self, input_params: Union[npt.ArrayLike, torch.Tensor, List[float]]
    ):
        """Generate parameters mu and sigma for the classifier."""

        mu, sigma, alpha = input_params

        # Make sure we are using the numpy library
        assert not self.use_torch, "This function is only for numpy"

        # Calculate the difference between the interpolated
        # transition state positions and the actual transition state positions
        total_error = 0
        for i in range(len(self.deltaG)):

            # Get the interpolated transition state positions
            interp_ts_positions = self.get_interpolated_transition_state_positions(
                self.reactant_graphs[i].molecule.cart_coords,
                self.product_graphs[i].molecule.cart_coords,
                mu=mu,
                sigma=sigma,
                alpha=alpha * self.deltaG[i],
                upper=self.upper,
                lower=self.lower,
            )
            # Get the species from the transition state graphs
            species = self.transition_state_graphs[i].molecule.species

            # Create an interpolated transition state molecule
            interp_ts_molecule = Molecule(
                species=species,
                coords=interp_ts_positions,
                charge=self.transition_state_graphs[i].molecule.charge,
                spin_multiplicity=self.transition_state_graphs[
                    i
                ].molecule.spin_multiplicity,
            )
            # Generate the interpolated transition state graph
            interp_ts_graph = MoleculeGraph.with_local_env_strategy(
                interp_ts_molecule, OpenBabelNN()
            )
            # Add the metal edge extender to the graph
            interp_ts_graph = metal_edge_extender(interp_ts_graph)

            difference_graphs = interp_ts_graph.diff(
                self.transition_state_graphs[i], strict=False
            )
            jaccard_distance = difference_graphs["dist"]
            total_error += jaccard_distance

        # Return the average error
        average_error = total_error / len(self.deltaG)
        logger.info(f"Average error: {average_error:.4f} reaction-1")

        return average_error
