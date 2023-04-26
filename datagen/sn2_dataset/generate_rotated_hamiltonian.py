"""Run calculations which generate the Hamiltonian and eigenvalues."""
import json
import yaml
import numpy as np
from ase import io as ase_io
from ase import build as ase_build
from pymatgen.io.ase import AseAtomsAdaptor
from atomate.qchem.fireworks.core import SinglePointFW, ForceFW
from atomate.common.powerups import add_tags
from fireworks import LaunchPad, Workflow


def rotate_three_dimensions(alpha, beta, gamma):
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

    return r_matrix


if __name__ == "__main__":
    """Rotate a water molecule and generate the Hamiltonian and eigenvalues."""

    # Launch the calculation
    lp = LaunchPad.from_file("/Users/sudarshanvijay/fw_config/my_launchpad.yaml")

    # Generate a water molecule
    water = ase_build.molecule("H2O")

    number_points = 10

    # Rotate the water molecule at an arbitrary angle
    np.random.seed(0)
    random_angles = 2 * np.pi * np.random.rand(number_points, 3, 1)

    # Import the simplest possible parameter setup
    with open("config/minimal_parameters.yaml") as f:
        params = yaml.safe_load(f)

    # Create the water structures
    water_structures = []
    angles_list = []
    for (alpha, beta, gamma) in random_angles:
        alpha = alpha[0]
        beta = beta[0]
        gamma = gamma[0]
        # Rotate the water molecules
        r_matrix = rotate_three_dimensions(alpha, beta, gamma)
        water_rotated = water.copy()
        old_positions = water.get_positions()
        new_positions = []
        for pos in old_positions:
            positions = np.dot(r_matrix, pos)
            new_positions.append(positions)
        water_rotated.set_positions(new_positions)
        water_structures.append(water_rotated)

        # Save the angles
        angles_list.append([alpha, beta, gamma])

    ase_io.write("output/water_rotated.xyz", water_structures, format="xyz")

    aaa = AseAtomsAdaptor()
    # Iterate over the elements and create the molecule object
    for index, water in enumerate(water_structures):

        # Create the molecule object
        molecule = aaa.get_molecule(water)

        # Run the simplest workflow available
        firew = SinglePointFW(
            molecule=molecule,
            qchem_input_params=params,
            db_file=">>db_file<<",
            extra_scf_print=True,
        )
        firew_force = ForceFW(
            molecule=molecule,
            qchem_input_params=params,
            db_file=">>db_file<<",
        )
        wf = Workflow([firew, firew_force], name="water_rotated_" + str(index))
        wf = add_tags(
            wf,
            {
                "group": "rotated_water_molecules",
                "angles": angles_list[index],
            },
        )

        lp.add_wf(wf)
