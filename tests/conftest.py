import os

import random

from collections import defaultdict

from itertools import product

import pytest

import numpy as np

from monty.serialization import loadfn, dumpfn
from pymatgen.core.structure import Molecule


@pytest.fixture()
def sn2_reaction_input(tmp_path):
    """Prepare SN2 reaction input."""

    BASIS_FUNCTION_ATOM = {
        "sto-3g": {
            "H": 1,
            "C": 1 + 1 + 3,
            "F": 1 + 1 + 3,
            "Cl": 1 + 1 + 3 + 1 + 3,
            "Br": 1 + 1 + 3 + 1 + 3 + 1 + 3 + 5,
        }
    }

    def _generate_sn2_input_data(basis_set):
        """Create an sn2 reaction data file. Currently implemented
        as a generic A -> B reaction. A and B are both complexes of
        either the initial or final state."""

        attacking_groups = ["H", "F", "Cl", "Br"]

        leaving_groups = ["H", "F", "Cl", "Br"]

        states = ["initial_state", "transition_state", "final_state"]

        datapoints = []

        for attacking_group, leaving_group in product(attacking_groups, leaving_groups):

            # Generate the initial state
            initial_state_structure = Molecule(
                species=["C", "H", "H", attacking_group, leaving_group],
                coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
            )
            final_state_structure = Molecule(
                species=["C", "H", "H", leaving_group, attacking_group],
                coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 1, 1]],
            )
            transition_state_structure = Molecule(
                species=["C", "H", "H", attacking_group, leaving_group],
                coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.75, 1, 1]],
            )
            structures = [
                initial_state_structure,
                transition_state_structure,
                final_state_structure,
            ]

            # Create a list of the final energies for the initial, transition and final states
            final_energy = [random.random() for _ in range(3)]

            num_basis_functions = np.sum(
                [
                    BASIS_FUNCTION_ATOM[basis_set][atom.symbol]
                    for atom in initial_state_structure.species
                ]
            )
            fock_matrices = [
                np.random.rand(2, num_basis_functions, num_basis_functions)
                for _ in range(3)
            ]
            fock_matrices = np.array(fock_matrices)
            fock_matrices = (fock_matrices + fock_matrices.transpose(0, 1, 3, 2)) / 2
            fock_matrices = fock_matrices.tolist()

            eigenvalues = [np.random.rand(num_basis_functions) for _ in range(3)]
            eigenvalues = np.array(eigenvalues)
            eigenvalues = eigenvalues.tolist()

            # Create a list of the overlap matrices for the initial, transition and final states
            overlap_matrices = [
                np.random.rand(2, num_basis_functions, num_basis_functions)
                for _ in range(3)
            ]
            overlap_matrices = np.array(overlap_matrices)
            overlap_matrices = (
                overlap_matrices + overlap_matrices.transpose(0, 1, 3, 2)
            ) / 2
            overlap_matrices = overlap_matrices.tolist()

            datapoint = {
                "fock_matrices": fock_matrices,
                "eigenvalues": eigenvalues,
                "overlap_matrices": overlap_matrices,
                "state": states,
                "final_energy": final_energy,
                "structures": structures,
            }

            datapoints.append(datapoint)

        return datapoints

    input_json_file = os.path.join(tmp_path, "inputs", "sn2_test_data", "input.json")

    os.makedirs(os.path.dirname(input_json_file), exist_ok=True)

    basis_set = "sto-3g"

    input_data = _generate_sn2_input_data(basis_set=basis_set)

    dumpfn(input_data, input_json_file)

    return input_json_file


@pytest.fixture()
def inner_interpolate_input(tmp_path):
    """Generate input for the inner interpolation model."""

    from ase.collections import g2
    from ase.build import molecule as ase_molecule

    from pymatgen.io.ase import AseAtomsAdaptor

    def _inner_interpolate_input():
        # Generate a series of structure to store based
        # on the g2 database of structures
        data_list = []

        for molecule_name in g2.names:
            # Generate the ase steucture for the molecule
            ase_structure = ase_molecule(molecule_name)
            # Convert to pymatgen molecule
            structure = AseAtomsAdaptor.get_molecule(ase_structure)
            # Generate a random float as the energy
            energy = random.random()
            # Generate a random list for the atomic charges
            atomic_charges = np.random.rand(len(structure))

            # Generate a random number for the net charge
            charge = random.random()
            # Generate a random number for the spin multiplicity
            spin_multiplicity = random.random()

            # Create a dict with all the inputs stored
            input_data = {}
            input_data["energy"] = energy
            input_data["structure"] = structure
            input_data["charge"] = charge
            input_data["spin_multiplicity"] = spin_multiplicity
            input_data["atomic_charges"] = atomic_charges

            data_list.append(input_data)

        return data_list

    # The input file for the tests.
    input_json_file = os.path.join(
        tmp_path, "inputs", "inner_interpolate_test_data", "input.json"
    )
    # Create the base directory for the input file.
    os.makedirs(os.path.dirname(input_json_file), exist_ok=True)

    # Generate inputs for the SN2 reaction.
    input_data = _inner_interpolate_input()

    # Write the input data to a json file.
    dumpfn(input_data, input_json_file)

    return input_json_file


def get_basis_file_info(basis_set: str = "sto-3g"):
    """Get the absolute path of the requested basis file."""
    relative_filepath = os.path.join("inputs", f"{basis_set}.json")
    confpath = os.path.dirname(os.path.abspath(__file__))
    absolute_filepath = os.path.join(confpath, relative_filepath)
    return absolute_filepath
