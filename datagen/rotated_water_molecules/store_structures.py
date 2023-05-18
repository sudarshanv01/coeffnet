from typing import List, Any

import argparse

import numpy as np

from ase import build as ase_build

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from minimal_basis.transforms.rotations import RotationMatrix

from instance_mongodb import instance_mongodb_sei


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
            "idx": 0,
        }
    )

    random_euler_angles = np.random.uniform(
        low=0, high=2 * np.pi, size=(args.number_points, 3)
    )

    original_positions = water_atoms.get_positions()

    for idx, euler_angles in enumerate(random_euler_angles):

        rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)
        r_matrix = rotation_matrix()

        water_rotated = water_atoms.copy()

        new_positions = []
        for pos in original_positions:
            positions = np.dot(r_matrix, pos)
            new_positions.append(positions)

        water_rotated.set_positions(new_positions)

        water_molecule = AseAtomsAdaptor.get_molecule(water_rotated)

        collection.insert_one(
            {
                "molecule": water_molecule.as_dict(),
                "euler_angles": euler_angles,
                "idx": idx + 1,
            }
        )
