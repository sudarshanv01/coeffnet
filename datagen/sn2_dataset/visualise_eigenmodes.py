from typing import List

import os

import copy

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import ase.io as ase_io

from instance_mongodb import instance_mongodb_sei


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


if __name__ == "__main__":
    """Visualise the eigenmodes of the transition states."""

    db = instance_mongodb_sei(project="mlts")
    collection = db.rudorff_lilienfeld_data

    scaling = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if not os.path.exists("output/visualise_eigenmodes"):
        os.makedirs("output/visualise_eigenmodes")

    for doc in collection.find({}):
        transition_state_molecule = doc["transition_state_with_one_imaginary_frequency"]
        transition_state_molecule = Molecule.from_dict(transition_state_molecule)
        transition_state_frequency_modes = doc["transition_state_frequency_modes"]
        transition_state_frequency_modes = np.array(transition_state_frequency_modes)
        transition_state_frequencies = doc["transition_state_frequencies"]
        transition_state_frequencies = np.array(transition_state_frequencies)

        rxn_number = doc["rxn_number"]
        reaction_name = doc["reaction_name"]

        perturbed_atoms = []
        for _scaling in scaling:

            idx_imag_freq = np.argwhere(transition_state_frequencies < 0)[0][0]
            transition_state_frequency_mode = transition_state_frequency_modes[
                idx_imag_freq
            ]

            perturbed_molecule = perturb_along_eigenmode(
                transition_state_molecule, transition_state_frequency_mode, _scaling
            )
            _perturbed_atoms = AseAtomsAdaptor.get_atoms(perturbed_molecule)
            perturbed_atoms.append(_perturbed_atoms)

        ase_io.write(
            "output/visualise_eigenmodes/{}_{}.xyz".format(rxn_number, reaction_name),
            perturbed_atoms,
        )
