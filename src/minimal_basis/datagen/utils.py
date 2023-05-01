import copy
from typing import List

import numpy as np

from pymatgen.core.structure import Molecule


def perturb_along_eigenmode(
    ts_molecule: Molecule, eigenmode: List[float], scaling: float
) -> Molecule:
    """Perturn the molecule along the eigen modes based on a scaling factor.

    Args:
        ts_molecule: The transition state molecule.
        eigenmode: The eigenmode.
        scaling: The scaling factor to perturb the molecule.
    """

    def validate_eigenmode(eigenmode: List[float]) -> None:
        """Check if the eigenmode is normalised correctly."""
        norm_eigemode = np.linalg.norm(eigenmode)
        is_close = np.isclose(norm_eigemode, 1.0, atol=1e-3)
        if not is_close:
            raise ValueError("The eigenmode is not normalised correctly.")

    eigenmode = np.array(eigenmode)
    validate_eigenmode(eigenmode)
    assert eigenmode.shape == (
        len(ts_molecule),
        3,
    ), "Eigenmode is not the correct shape."

    delta_pos = scaling * eigenmode
    perturbed_molecule_pos = copy.deepcopy(ts_molecule)

    # get positions of atoms
    positions = [a.coords for a in ts_molecule.sites]
    positions = np.array(positions)

    # Get the perturbed positions
    perturbed_pos = positions + delta_pos

    # Set the perturbed positions
    for i, a in enumerate(perturbed_molecule_pos.sites):
        a.coords = perturbed_pos[i]

    return perturbed_molecule_pos
