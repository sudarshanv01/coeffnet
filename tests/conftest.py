import os
import random
from collections import defaultdict

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

        # Sequence of attacking groups
        attacking_groups = ["H", "F", "Cl", "Br"]

        # Store all data is a list
        all_data = []

        for attacking_group in attacking_groups:

            input_data = defaultdict(lambda: defaultdict(dict))

            # --- Global output data ---
            # Get some random positive number for the transitition state energy.
            random_ts_energy = random.random()
            input_data["transition_state_energy"] = random_ts_energy
            # Get some random positive number for the reaction energy
            random_reaction_energy = random.random()
            input_data["reaction_energy"] = random_reaction_energy

            # --- Structure related data ---
            species_A = ["C", "C", attacking_group, "H", "H", "H", "H", "H", "H"]
            coords_A = [
                [-0.72558049658904, -0.14174575671651, 0.00208172081967],
                [0.76901052138111, 0.06698557922871, -0.02093370787145],
                [-1.36071750263840, 1.05990731125891, 0.30117556196699],
                [1.27985275772671, -0.87012288772595, -0.25482533354305],
                [-1.09972495621085, -0.47643631328414, -0.96829154886166],
                [-1.01869749371280, -0.86537324884060, 0.76632671534726],
                [1.03764409021868, 0.80730478500607, -0.77564403715648],
                [1.11821307982460, 0.42048053107351, 0.95011062929873],
                [0.0, 0.0, 0.0],
            ]
            molecule_A = Molecule(species_A, coords_A, charge=0, spin_multiplicity=1)

            species_B = ["C", "C", "H", "H", "H", "H", "H", "H", attacking_group]
            coords_B = [
                [-0.72558049658904, -0.14174575671651, 0.00208172081967],
                [0.76901052138111, 0.06698557922871, -0.02093370787145],
                [-1.36071750263840, 1.05990731125891, 0.30117556196699],
                [1.27985275772671, -0.87012288772595, -0.25482533354305],
                [-1.09972495621085, -0.47643631328414, -0.96829154886166],
                [-1.01869749371280, -0.86537324884060, 0.76632671534726],
                [1.03764409021868, 0.80730478500607, -0.77564403715648],
                [1.11821307982460, 0.42048053107351, 0.95011062929873],
                [0.0, 0.0, 0.0],
            ]
            molecule_B = Molecule(species_B, coords_B, charge=0, spin_multiplicity=1)

            # Generate molecule information
            input_data["final_state"]["molecule"] = molecule_A.as_dict()
            input_data["initial_state"]["molecule"] = molecule_B.as_dict()

            # Generate the alpha and beta fock matrix for each molecule
            basis_functions_A = sum(
                [BASIS_FUNCTION_ATOM[basis_set][atom] for atom in species_A]
            )
            basis_functions_B = sum(
                [BASIS_FUNCTION_ATOM[basis_set][atom] for atom in species_B]
            )

            # Generate the alpha and beta fock matrix for each molecule
            # These are random (symmetric) matrices of size (no_basis_functions, no_basis_functions)
            fock_matrix_A = np.random.rand(basis_functions_A, basis_functions_A)
            fock_matrix_A = (fock_matrix_A + fock_matrix_A.T) / 2
            fock_matrix_B = np.random.rand(basis_functions_B, basis_functions_B)
            fock_matrix_B = (fock_matrix_B + fock_matrix_B.T) / 2
            # Create the fock matrix for the transition state
            fock_matrix_X = np.random.rand(basis_functions_A, basis_functions_A)
            fock_matrix_X = (fock_matrix_X + fock_matrix_X.T) / 2

            # Store the fock matrix in the input data.
            input_data["initial_state"]["alpha_fock_matrix"] = fock_matrix_A.tolist()
            input_data["initial_state"]["beta_fock_matrix"] = fock_matrix_A.tolist()

            input_data["final_state"]["alpha_fock_matrix"] = fock_matrix_B.tolist()
            input_data["final_state"]["beta_fock_matrix"] = fock_matrix_B.tolist()

            all_data.append(input_data)

        return all_data

    # The input file for the tests.
    input_json_file = os.path.join(tmp_path, "inputs", "sn2_test_data", "input.json")
    # Create the base directory for the input file.
    os.makedirs(os.path.dirname(input_json_file), exist_ok=True)

    # All inputs for any reaction are stored in a nested dict.
    basis_set = "sto-3g"

    # Generate inputs for the SN2 reaction.
    input_data = _generate_sn2_input_data(basis_set=basis_set)

    # Write the input data to a json file.
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
