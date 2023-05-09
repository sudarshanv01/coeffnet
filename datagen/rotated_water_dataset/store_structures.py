from typing import List, Any

import argparse

import numpy as np

from ase import build as ase_build

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from instance_mongodb import instance_mongodb_sei


def rotate_three_dimensions(alpha: float, beta: float, gamma: float):
    """Rotate the molecule by arbitrary angles alpha
    beta and gamma."""
    cos = np.cos
    sin = np.sin

    r_matrix = [
        [
            cos(alpha) * cos(beta),
            cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
            cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
        ],
        [
            sin(alpha) * cos(beta),
            sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
            sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
        ],
        [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)],
    ]

    r_matrix = np.array(r_matrix)

    return r_matrix


def get_command_line_arguments():
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_points",
        type=int,
        default=10,
        help="Number of points to generate.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Create a water structure and rotate it at random angles."""

    args = get_command_line_arguments()

    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_water_initial_structures

    water_atoms = ase_build.molecule("H2O")
    water_molecule = AseAtomsAdaptor.get_molecule(water_atoms)
    collection.insert_one(
        {
            "molecule": water_molecule.as_dict(),
            "angles": [0, 0, 0],
        }
    )

    random_angles = 2 * np.pi * np.random.rand(args.number_points - 1, 3)

    original_positions = water_atoms.get_positions()

    for (alpha, beta, gamma) in random_angles:

        r_matrix = rotate_three_dimensions(alpha, beta, gamma)

        water_rotated = water_atoms.copy()

        new_positions = []
        for pos in original_positions:
            positions = np.dot(r_matrix, pos)
            new_positions.append(positions)

        water_rotated.set_positions(new_positions)

        water_molecule = AseAtomsAdaptor.get_molecule(water_rotated)
        angles = [alpha, beta, gamma]

        collection.insert_one(
            {
                "molecule": water_molecule.as_dict(),
                "angles": angles,
            }
        )
