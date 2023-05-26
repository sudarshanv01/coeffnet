from typing import List, Any

from pathlib import Path

import os

import argparse

import numpy as np

from ase import build as ase_build
from ase import io as ase_io

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.operations import SymmOp

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
    parser.add_argument(
        "--writeout_xyz",
        action="store_true",
        help="Write out the xyz files.",
        default=False,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run.",
        default=False,
    )
    args = parser.parse_args()
    return args


__output_dir__ = "./output"
__output_dir__ = Path(__output_dir__)
if not os.path.exists(__output_dir__):
    os.makedirs(__output_dir__)

if __name__ == "__main__":
    """Choose a random sn2 reaction and rotate the molecule."""

    args = get_command_line_arguments()

    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_sn2_reaction_initial_structures

    sn2_reaction_data_collection = db.rudorff_lilienfeld_data
    sn2_reaction_data = sn2_reaction_data_collection.find_one(
        {
            "transition_state_molecule": {"$exists": True},
            "perturbed_molecule_0_5_molecule": {"$exists": True},
            "perturbed_molecule_-0_5_molecule": {"$exists": True},
        }
    )

    random_euler_angles = np.random.uniform(
        low=0, high=2 * np.pi, size=(args.number_points, 3)
    )

    original_transition_state_molecule = Molecule.from_dict(
        sn2_reaction_data["transition_state_molecule"]
    )
    original_reactant_molecule = Molecule.from_dict(
        sn2_reaction_data["perturbed_molecule_-0_5_molecule"]
    )
    original_products_molecule = Molecule.from_dict(
        sn2_reaction_data["perturbed_molecule_0_5_molecule"]
    )

    for idx, euler_angles in enumerate(random_euler_angles):

        rotation_matrix = RotationMatrix(angle_type="euler", angles=euler_angles)
        r_matrix = rotation_matrix()
        rotation_matrix = SymmOp.from_rotation_and_translation(
            rotation_matrix=r_matrix, translation_vec=[0, 0, 0]
        )

        transition_state_molecule = original_transition_state_molecule.copy()
        transition_state_molecule.set_charge_and_spin(charge=-1, spin_multiplicity=1)
        transition_state_molecule.apply_operation(rotation_matrix)

        reactant_molecule = original_reactant_molecule.copy()
        reactant_molecule.set_charge_and_spin(charge=-1, spin_multiplicity=1)
        reactant_molecule.apply_operation(rotation_matrix)

        product_molecule = original_products_molecule.copy()
        product_molecule.set_charge_and_spin(charge=-1, spin_multiplicity=1)
        product_molecule.apply_operation(rotation_matrix)

        if args.writeout_xyz:
            label = f"rotated_sn2_reaction_{idx + 1}"
            transition_state_molecule.to(
                fmt="xyz", filename=__output_dir__ / f"ts_{label}.xyz"
            )
            reactant_molecule.to(
                fmt="xyz", filename=__output_dir__ / f"reactant_{label}.xyz"
            )
            product_molecule.to(
                fmt="xyz", filename=__output_dir__ / f"product_{label}.xyz"
            )

        if not args.dryrun:
            collection.insert_one(
                {
                    "transition_state_molecule": transition_state_molecule.as_dict(),
                    "reactant_molecule": reactant_molecule.as_dict(),
                    "product_molecule": product_molecule.as_dict(),
                    "euler_angles": euler_angles.tolist(),
                    "idx": idx + 1,
                }
            )
