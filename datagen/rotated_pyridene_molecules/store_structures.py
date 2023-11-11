from typing import List, Any

import argparse

import numpy as np

from ase import build as ase_build
from ase.data.pubchem import pubchem_atoms_search

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from coeffnet.transforms.rotations import RotationMatrix

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
    """Create a structure and rotate it at random angles."""
    args = get_command_line_arguments()
    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_pyridene_initial_structures
    atoms = pubchem_atoms_search(cid=1049)
    molecule = AseAtomsAdaptor.get_molecule(atoms)
    collection.insert_one(
        {
            "molecule": molecule.as_dict(),
            "euler_angles": [0, 0, 0],
            "idx": 0,
        }
    )
    random_euler_angles = np.random.uniform(0, 2 * np.pi, size=(args.number_points, 3))
    original_positions = atoms.get_positions()
    for idx, euler_angles in enumerate(random_euler_angles):
        rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)
        r_matrix = rotation_matrix()
        rotated_atoms = atoms.copy()
        new_positions = []
        for pos in original_positions:
            positions = np.dot(r_matrix, pos)
            new_positions.append(positions)
        rotated_atoms.set_positions(new_positions)
        molecule = AseAtomsAdaptor.get_molecule(rotated_atoms)
        collection.insert_one(
            {
                "molecule": molecule.as_dict(),
                "euler_angles": euler_angles.tolist(),
                "idx": idx + 1,
            }
        )