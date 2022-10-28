import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

import copy

from collections import defaultdict

import numpy as np

from ase import data as ase_data

from pymatgen.core.structure import Molecule

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

from e3nn import o3

from minimal_basis.data import DataPoint
from minimal_basis.dataset.utils import generate_graphs_by_method
from minimal_basis.data._dtype import DTYPE

logger = logging.getLogger(__name__)


class HamiltonianDataset(InMemoryDataset):
    """Dataset for the Hamiltonian for all species in a reaction."""

    GLOBAL_INFORMATION = ["state_fragment"]
    MOLECULE_INFORMATION = ["positions", "graphs"]
    FEATURE_INFORMATION = ["hamiltonian"]

    # Store the converter to convert the basis functions
    BASIS_CONVERTER = {
        "s": 1,
        "p": 3,
        "d": 5,
        "f": 7,
    }
    # Store the l for each basis function
    l_to_n_basis = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    n_to_l_basis = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}

    def __init__(
        self,
        filename: Union[str, Path],
        basis_file: Union[str, Path],
        root: str,
        transform: str = None,
        pre_transform: bool = None,
        pre_filter: bool = None,
        graph_generation_method: str = "sn2",
    ):
        self.filename = filename
        self.basis_file = basis_file
        self.graph_generation_method = graph_generation_method

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def raw_file_names(self):
        return "input.json"

    @property
    def processed_file_names(self):
        return ["hamiltonian_data.pt", "basis_data.pt"]

    def download(self):
        """Load the json file with the Hamiltonian and basis information."""

        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")

        with open(self.basis_file) as f:
            self.basis_info_raw = json.load(f)
        logger.info("Successfully loaded json file with basis information.")

        logger.info("Parsing basis information.")
        self._parse_basis_data()

    def _parse_basis_data(self):
        """Parse the basis information from data from basissetexchange.org
        json format to a dict containing the number of s, p and d functions
        for each atom. The resulting dictionary, self.basis_info contains the
        total set of basis functions for each atom.
        """

        elemental_data = self.basis_info_raw["elements"]

        # Create a new dict with the basis information
        self.basis_info = {}
        logger.info(f"Parsing basis information from {self.basis_file}")

        for atom_number in elemental_data:
            angular_momentum_all = []
            for basis_index, basis_functions in enumerate(
                elemental_data[atom_number]["electron_shells"]
            ):
                angular_momentum_all.extend(basis_functions["angular_momentum"])
            # Convert each number in the list to a letter corresponding
            # to the orbital in which the basis function is located
            angular_momentum_all = [
                self.n_to_l_basis[element] for element in angular_momentum_all
            ]
            self.basis_info[int(atom_number)] = angular_momentum_all

    def get_indices_of_basis(
        cls, atom_basis_functions: List[List[str]]
    ) -> List[List[int]]:
        """Generate a list containing the start and stop index of the
        basis function corresponding to each orbital."""
        # Create a list of index based on the atom wise basis functions
        # We will generate a list of lists, where the format is:
        # [ [ [start_index, stop_index], [start_index, stop_index], ... ], ... ]
        # - Atom in the molecule
        # - Orbital in the atom
        # - Start and stop index of the basis function
        list_of_indices = []

        # Start labelling index at 0
        overall_index = 0

        # Iterate over the atom_basis_functions and make sure that the
        # basis functions chosen are stored in a separate list for each
        # atom.
        for basis_list in atom_basis_functions:
            start_stop = []
            # Iterate over the basis elements in the basis list
            # And find the start and stop index of the basis function
            for basis_element in basis_list:
                overall_index += basis_element
                start_stop.append([overall_index - basis_element, overall_index])
            list_of_indices.append(start_stop)

        return list_of_indices

    def get_indices_atom_basis(
        cls, indices_of_basis: List[List[int]]
    ) -> List[List[int]]:
        """Get the starting and ending index of each atom. This list
        serves as a sort of `atomic-basis` representation of the
        Hamiltonian matrix."""
        indices_atom_basis = []
        for basis_list in indices_of_basis:
            flattened_basis_list = [item for sublist in basis_list for item in sublist]
            # Assumes that the basis index are in order
            # TODO: Check if there is a better way to do this
            indices_atom_basis.append(
                [flattened_basis_list[0], flattened_basis_list[-1]]
            )
        return indices_atom_basis

    def _string_basis_to_number(self, string_basis_list: List[str]) -> List[int]:
        """Convert a list of lists containing string elements to
        the corresponding number of basis functions."""
        # Convert the string basis to the number of basis functions
        number_basis_list = []
        for string_basis in string_basis_list:
            number_of_elements = [
                self.BASIS_CONVERTER[element] for element in string_basis
            ]
            number_basis_list.append(number_of_elements)
        return number_basis_list

    def process(self) -> List[DataPoint]:
        """Gets data from JSON file and store it in a list of DataPoint objects.
        Conventions:
        -----------
        1. Sign of `molecule_id` is used to check the state of the molecule
           molecules that have a negative `molecule_id` belong to a either
           the reactants or products and vice versa.
        2. `indices_basis` refers to a list of lists containing the
           start and stop index of the basis function corresponding to
           each orbital. That is, it is basis used in the calculations.
        3. `indices_atom_basis` refers to a list of lists containing the
           start and stop index of the basis function corresponding to
           each atom. That is, it is the atomic-basis representation of
           the Hamiltonian matrix.
        """

        # Populate this list with datapoints
        datapoint_list = []

        for reaction_idx, input_data_ in enumerate(self.input_data):

            input_data = copy.deepcopy(input_data_)

            # The label for this reaction
            label = input_data.pop("label")

            # Collect the molecular level information
            molecule_info_collected = defaultdict(dict)

            # Store the molecules separately to make sure
            # that the graph creation process is flexible
            molecules_in_reaction = defaultdict(list)

            # --- Get the output information and store that in the node
            y = input_data.pop("transition_state_energy")

            # --- Get the global information
            global_information = input_data.pop("reaction_energy")

            for molecule_id in input_data:

                state_fragment = input_data[molecule_id]["state_fragments"]

                molecule = input_data[molecule_id]["molecule"]
                if state_fragment == "initial_state":
                    molecules_in_reaction["reactants"].append(molecule)
                    molecules_in_reaction["reactants_index"].append(molecule_id)
                elif state_fragment == "final_state":
                    molecules_in_reaction["products"].append(molecule)
                    molecules_in_reaction["products_index"].append(molecule_id)

                # Get the atom basis functions
                atom_basis_functions = []
                for atom_name in molecule.species:
                    _atom_number = ase_data.atomic_numbers[atom_name.symbol]
                    atom_basis_functions.append(self.basis_info[_atom_number])
                # Get the number of basis functions for each atom
                number_basis_functions = self._string_basis_to_number(
                    atom_basis_functions
                )
                # Sum up all the basis functions, will be used to define
                # all the tensors below.
                total_basis = np.sum([np.sum(a) for a in number_basis_functions])
                logger.debug(f"Total basis functions: {total_basis}")

                # Get the indices of the basis functions
                indices_of_basis = self.get_indices_of_basis(number_basis_functions)
                molecule_info_collected["indices_basis"][molecule_id] = indices_of_basis

                # Get the diagonal elements consisting of each atom
                indices_atom_basis = self.get_indices_atom_basis(indices_of_basis)
                logger.debug(f"Indices of atom basis functions: {indices_atom_basis}")
                molecule_info_collected["indices_atom_basis"][
                    molecule_id
                ] = indices_atom_basis

                # Get the Hamiltonian and partition it into diagonal and off-diagonal
                # elements. Diagonal elements may be blocks of s-s, s-p, ...
                # which consist of the intra-atomic basis functions. The
                # off-diagonal elements are for interactions between
                # the inter-atomic basis functions.
                node_H = torch.zeros(total_basis, total_basis, 2, dtype=DTYPE)
                edge_H = torch.zeros(total_basis, total_basis, 2, dtype=DTYPE)

                for spin_index, spin in enumerate(["alpha", "beta"]):
                    # Get the Hamiltonian for each spin
                    if spin == "beta":
                        if input_data[molecule_id]["beta_fock_matrix"] == None:
                            # There is no computed beta spin, i.e. alpha and beta are the same
                            hamiltonian = input_data[molecule_id]["alpha_fock_matrix"]
                        else:
                            hamiltonian = input_data[molecule_id][spin + "_fock_matrix"]
                    else:
                        # Must always have an alpha spin
                        hamiltonian = input_data[molecule_id][spin + "_fock_matrix"]
                    # Make sure the Hamiltonian is a tensor
                    hamiltonian = torch.tensor(hamiltonian, dtype=DTYPE)
                    logger.debug(f"Hamiltonian shape: {hamiltonian.shape}")

                    # Split the matrix into node and edge features, for each spin separately
                    node_H_spin, edge_H_spin = self.split_matrix_node_edge(
                        hamiltonian, indices_atom_basis
                    )
                    node_H[..., spin_index] = node_H_spin
                    edge_H[..., spin_index] = edge_H_spin

            (
                edge_molecule_mapping,
                edge_internal_mol_mapping,
            ) = generate_graphs_by_method(
                graph_generation_method=self.graph_generation_method,
                molecules_in_reaction=molecules_in_reaction,
                molecule_info_collected=molecule_info_collected,
            )

            datapoint = DataPoint(
                pos=molecule_info_collected["pos"],
                edge_index=molecule_info_collected["edge_index"],
                edge_attr=molecule_info_collected["edge_attr"],
                x=molecule_info_collected["x"],
                y=y,
                indices_of_basis=molecule_info_collected["indices_basis"],
                indices_atom_basis=molecule_info_collected["indices_atom_basis"],
                edge_molecule_mapping=edge_molecule_mapping,
                edge_internal_mol_mapping=edge_internal_mol_mapping,
            )

            logging.debug("Datapoint:")
            logging.debug(datapoint)

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        return datapoint_list

    def split_matrix_node_edge(cls, matrix, indices_of_basis):
        """Split the matrix into node and edge features."""
        # Matrix is a tensor of shape (n,n) where n is the total
        # number of basis functions. We need to separate the diagonal
        # blocks (elements that belong to one atom) and the off-diagonal
        # elements that belong to two-centre interactions.
        elemental_matrix = torch.zeros(matrix.shape[0], matrix.shape[1])
        coupling_matrix = torch.zeros(matrix.shape[0], matrix.shape[1])

        # Populate the elemental matrix first by splicing
        for basis_elements_list in indices_of_basis:
            basis_elements = range(basis_elements_list[0], basis_elements_list[1])
            for y in basis_elements:
                elemental_matrix[basis_elements, y] = matrix[basis_elements, y]

        # Populate the coupling matrix by subtracting the total matrix from the elemental matrix
        coupling_matrix = matrix - elemental_matrix

        return elemental_matrix, coupling_matrix
