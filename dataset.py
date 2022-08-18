"""Generate all the Data objects needed to predict the TS Hamiltonian."""
import os
import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging
import itertools

from collections import defaultdict

from ase import data as ase_data

import numpy as np

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset

from pymatgen.core.structure import Molecule

import networkx as nx
import matplotlib.pyplot as plt

from data import DataPoint


class HamiltonianDataset(InMemoryDataset):
    """Implement simple dataset to read in different
    Hamiltonians for the initial and final state and 
    store them as a Data object."""

    GLOBAL_INFORMATION = ['state_fragment']
    MOLECULE_INFORMATION = ['positions', 'graphs', 'irrep']
    FEATURE_INFORMATION = ['hamiltonian']

    def __init__(self, 
        filename: Union[str, Path],
        basis_file: Dict[str, int],
    ):
        self.filename = filename
        self.basis_file = basis_file 

        # Store the converter to convert the basis functions
        self.BASIS_CONVERTER = {
            "s": 1, 
            "p": 3,
            "d": 5,
            "f": 7,
        }
        # Store the l for each basis function
        self.l_to_n_basis = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
        self.n_to_l_basis = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}

        super().__init__()
    
    def parse_basis_data(self):
        """Parse the basis information from data from basissetexchange.org
        json format to a dict containing the number of s, p and d functions
        for each atom. The resulting dictionary, self.basis_info contains the
        total set of basis functions for each atom.

        TODO: Currently, it is assumed that the number and type of basis functions
        for each atom is fixed. This might not always be the case, and would need
        to be verified based on the calculation.
        """

        elemental_data = self.basis_info_raw['elements']

        # Create a new dict with the basis information
        self.basis_info = {}

        logging.info(f"Parsing basis information from {self.basis_file}")
        logging.info("Basis information (for the first 10 atoms):")

        for atom_number in elemental_data:
            angular_momentum_all = []
            for basis_index, basis_functions in enumerate(elemental_data[atom_number]['electron_shells']):
                angular_momentum_all.extend(basis_functions['angular_momentum'])
            # Convert each number in the list to a letter corresponding
            # to the orbital in which the basis function is located
            angular_momentum_all = [self.n_to_l_basis[element] for element in angular_momentum_all]
            self.basis_info[int(atom_number)] = angular_momentum_all
            # Write out the basis_info for the first 10 atoms so that the level
            # of theory is known (these are also the atoms that are actually used
            # this application).
            if int(atom_number) <= 10:
                logging.info(f'Basis functions for atom {atom_number}: {angular_momentum_all}')
        
        logging.info('')

    def complete_graph(cls, molecule: Molecule, starting_index=0) -> Tuple[np.ndarray, List[int], int]:
        """
        NOTE: Modified from mjwen's eigenn package.
        Build a complete graph, where each node is connected to all other nodes.
        Args:
            N: number of atoms
            starting_index: index of the first graph to start from
        Returns:
            edge index, shape (2, N). For example, for a system with 3 atoms, this is
                [[0,0,1,1,2,2],
                [1,2,0,2,0,1]]
            num_neigh: number of neighbors for each atom
        """
        # Get the number of atoms in the molecule
        N = len(molecule.species)
        # Decide on the start and end index of this particular graph.
        N_start = starting_index
        N_end = N_start + N
        edge_index = np.asarray(list(zip(*itertools.permutations(range(N_start, N_end), r=2))))
        num_neigh = [N - 1 for _ in range(N_start, N_end)]

        # Add the number of atoms to the starting index
        delta_starting_index = N

        return edge_index, num_neigh, delta_starting_index
    
    def load_data(self):
        """Load the json file with the data."""
        with open(self.filename) as f:
            self.input_data = json.load(f)
        logging.info('Successfully loaded json file with data.')
        # Also lead the basis file
        with open(self.basis_file) as f:
            self.basis_info_raw = json.load(f)
        logging.info('Successfully loaded json file with basis information.')
        logging.info('')

    def get_indices_of_basis(cls, atom_basis_functions: List[List[str]]) -> List[List[int]]:
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
                start_stop.append([overall_index-basis_element, overall_index]) 
            list_of_indices.append(start_stop)
        # Log the list of indices if in debug mode
        logging.debug(f'List of indices: {list_of_indices}')
        return list_of_indices
    
    def get_indices_atom_basis(cls, indices_of_basis: List[List[int]]) -> List[List[int]]:
        """Get the starting and ending index of each atom. This list
        serves as a sort of `atomic-basis` representation of the 
        Hamiltonian matrix."""
        indices_atom_basis = []
        for basis_list in indices_of_basis:
            flattened_basis_list = [item for sublist in basis_list for item in sublist]
            # Assumes that the basis index are in order
            # TODO: Check if there is a better way to do this
            indices_atom_basis.append([flattened_basis_list[0], flattened_basis_list[-1]])
        # Logging in debug mode
        logging.debug(f'List of indices for each atom: {indices_atom_basis}')
        return indices_atom_basis 

    def _string_basis_to_number(self, string_basis_list: List[str]) -> List[int]:
        """Convert a list of lists containing string elements to
        the corresponding number of basis functions."""
        # Convert the string basis to the number of basis functions
        number_basis_list = []
        for string_basis in string_basis_list:
            number_of_elements = [self.BASIS_CONVERTER[element] for element in string_basis]
            number_basis_list.append(number_of_elements)
        return number_basis_list

    def get_data(self) -> List[DataPoint]:
        """Gets data from JSON file and store it in a list of DataPoint objects."""

        # This function must return a list of Data objects
        # Each reaction corresponds to a Data object
        datapoint_list = []

        # Iterate over the reaction ID and generate a series
        # of Data objects for each reaction. The Data objects
        # would need the position and edge index, and the features
        # TODO: Currently only implement the Hamiltonian for each
        # species. The features of this Hamiltonian are dictated
        # by the number of elements and the chosen basis set. 

        for reaction_id in self.input_data:
            # The reaction_id is the key for the reactions database
            index_prod = 0 ; index_react = 0

            # Collect the molecular level information
            molecule_info_collected = defaultdict(dict)

            # This index makes sure that a single graph is generated
            # consisting of all reactants and products. The index monitors
            # the total number of atoms, making sure that there is no overlap
            # in terms of the index of the resulting graph.
            starting_index = 0

            for molecule_id in self.input_data[reaction_id]:
                # Prepare the information for each molecule, this forms
                # part of the graph that is making up the DataPoint.

                # --- Get state (global) level information ---
                # Get the state of the molecule, i.e., does it
                # belong to the initial or final state of the reaction?
                logging.info(f"--- Global level information: {self.GLOBAL_INFORMATION}")
                state_fragment = self.input_data[reaction_id][molecule_id]['state_fragments']
                if state_fragment == 'initial_state':
                    index_react -= 1
                    choose_index = index_react
                elif state_fragment == 'final_state':
                    index_prod += 1
                    choose_index = index_prod
                else:
                    raise ValueError('State fragment not recognised.')
                logging.info('State of molecule: {}'.format(state_fragment))

                # --- Get molecule level information ---
                # Get the molecule object
                logging.info(f"--- Molecule level information: {self.MOLECULE_INFORMATION}")
                molecule_dict = self.input_data[reaction_id][molecule_id]['molecule']
                molecule = Molecule.from_dict(molecule_dict)

                # Get the positions of the molecule (these are cartesian coordinates)
                pos = [list(a.coords) for a in molecule]
                molecule_info_collected['pos'][choose_index] = pos

                # Construct a graph from the molecule object, each node
                # of the graph is connected with every other node.
                edge_index, num_neigh, delta_starting_index = self.complete_graph(molecule,
                                                                                  starting_index=starting_index)
                # Move the starting index so that the next molecule comes after
                # this molecule.
                starting_index += delta_starting_index
                molecule_info_collected['edge_index'][choose_index] = edge_index

                # Get the atom basis functions 
                atom_basis_functions = []
                for atom_name in molecule.species:
                    _atom_number = ase_data.atomic_numbers[atom_name.symbol]
                    atom_basis_functions.append(self.basis_info[_atom_number])
                # write out the atom basis functions
                logging.debug(f'Atom basis functions: {atom_basis_functions}')

                # Store an array for the total irrep of the matrix
                irreps_list = [] 
                # Store an array for irrep of each atom, corresponding to the
                # number of atomic basis functions
                irreps_atoms_list = [] 
                # Create a flattened list of the atom basis functions as well
                # with repeats of the number of basis functions for each type of
                # orbital. Useful to map each row of the matrix to a particular
                # basis function, irrespective of the atom. 
                flattened_atom_basis_functions = []
                for atom_basis in atom_basis_functions:
                    tot_number_of_s_atom = atom_basis.count('s')
                    tot_number_of_p_atom = atom_basis.count('p')
                    tot_number_of_d_atom = atom_basis.count('d')
                    tot_number_of_f_atom = atom_basis.count('f')
                    irreps_atoms = f"{tot_number_of_s_atom}x0e + {tot_number_of_p_atom}x1o + {tot_number_of_d_atom}x2e + {tot_number_of_f_atom}x3o"
                    irreps_atoms_list.append(irreps_atoms)
                    for basis_function in atom_basis:
                        irreps_list.append(self.l_to_n_basis[basis_function])
                        # Store the flattened version of the atom basis functions (including repeats)
                        list_flat_rep = [basis_function] * self.BASIS_CONVERTER[basis_function]
                        flattened_atom_basis_functions.extend(list_flat_rep)

                # Store the irreps for each molecule 
                molecule_info_collected['irrep_atoms'][choose_index] = irreps_atoms_list
                # Store the flattened atom basis functions
                molecule_info_collected['atom_basis_functions'][choose_index] = flattened_atom_basis_functions

                # Get the indices of the basis functions
                tot_number_of_s = irreps_list.count(0) 
                tot_number_of_p = irreps_list.count(1)
                tot_number_of_d = irreps_list.count(2)
                tot_number_of_f = irreps_list.count(3)

                logging.info(f'Number of s: {tot_number_of_s}')
                logging.info(f'Number of p: {tot_number_of_p}')
                logging.info(f'Number of d: {tot_number_of_d}')
                logging.info(f'Number of f: {tot_number_of_f}')
                logging.info(f'Total number of computed: {tot_number_of_s + tot_number_of_p + tot_number_of_d + tot_number_of_f}')
                logging.info(f'Irreps required: {len(irreps_list)}')

                # Make sure that the total number of basis functions can be
                # split into just s, p and d basis functions
                assert tot_number_of_s + tot_number_of_p + tot_number_of_d + tot_number_of_f == len(irreps_list)

                # Get the number of basis functions for each atom
                number_basis_functions = self._string_basis_to_number(atom_basis_functions)
                logging.debug(f'Number of basis functions: {number_basis_functions}')

                # Sum up all the basis functions
                total_basis = np.sum([np.sum(a) for a in number_basis_functions])
                logging.info(f'Total basis functions: {total_basis}')

                # Get the indices of the basis functions
                indices_of_basis = self.get_indices_of_basis(number_basis_functions)
                logging.debug(f'Indices of basis functions: {indices_of_basis}')
                molecule_info_collected['indices_of_basis'][choose_index] = indices_of_basis

                # Get the diagonal elements consisting of each atom
                indices_atom_basis = self.get_indices_atom_basis(indices_of_basis)
                logging.debug(f'Indices of atom basis functions: {indices_atom_basis}')

                # Finally, store the irrep
                irrep = f"{tot_number_of_s}x0e + {tot_number_of_p}x1o + {tot_number_of_d}x2e + {tot_number_of_f}x3o"
                molecule_info_collected['irrep'][choose_index] = irrep
                logging.info(f'Irrep: {irrep}')

                # --- Get feauture level information ---
                logging.info(f"--- Feature level information: {self.FEATURE_INFORMATION}")
                # Get the Hamiltonian and partition it into diagonal and off-diagonal
                # elements. Diagonal elements may be blocks of s-s, s-p, ... 
                # which consist of the intra-atomic basis functions. The 
                # off-diagonal elements are for interactions between 
                # the inter-atomic basis functions.
                node_H = torch.zeros(total_basis, total_basis, 2, dtype=torch.float)
                edge_H = torch.zeros(total_basis, total_basis, 2, dtype=torch.float)
                for spin_index, spin in enumerate(['alpha', 'beta']):
                    # Get the Hamiltonian for each spin
                    if spin == 'beta':
                        if self.input_data[reaction_id][molecule_id]['beta_fock_matrix'] == None:
                            # There is no computed beta spin, i.e. alpha and beta are the same
                            hamiltonian = self.input_data[reaction_id][molecule_id]['alpha_fock_matrix']
                        else:
                            hamiltonian = self.input_data[reaction_id][molecule_id][spin + '_fock_matrix']
                    else:
                        # Must always have an alpha spin
                        hamiltonian = self.input_data[reaction_id][molecule_id][spin + '_fock_matrix']

                    # Make sure the Hamiltonian is a tensor
                    hamiltonian = torch.tensor(hamiltonian, dtype=torch.float)
                    logging.info(f'Hamiltonian shape: {hamiltonian.shape}')
                    # Make sure that the total number of basis functions equals the shape of the Hamiltonian
                    assert total_basis == hamiltonian.shape[0] == hamiltonian.shape[1]

                    # Split the matrix into node and edge features, for each spin separately
                    node_H_spin, edge_H_spin = self.split_matrix_node_edge(hamiltonian, indices_atom_basis)
                    node_H[..., spin_index] = node_H_spin
                    edge_H[..., spin_index] = edge_H_spin

                # Store both the node and edge attributes in the dictionary
                # which are essentially just different parts of the Hamiltonian.
                molecule_info_collected['x'][choose_index] = node_H 
                molecule_info_collected['edge_attr'][choose_index] = edge_H

                # --- Get the output information and store that in the node
                y = self.input_data[reaction_id][molecule_id]['barrier_meanref']

            datapoint = DataPoint(
                pos = molecule_info_collected['pos'],
                edge_index = molecule_info_collected['edge_index'],
                edge_attr = molecule_info_collected['edge_attr'],
                x = molecule_info_collected['x'],
                y = y,
                irrep = molecule_info_collected['irrep'],
                irrep_atoms = molecule_info_collected['irrep_atoms'],
                indices_of_basis = molecule_info_collected['indices_of_basis'],
                atom_basis_functions = molecule_info_collected['atom_basis_functions'],
            )
            logging.info('Datapoint:')
            logging.info(datapoint)
            logging.info('-------')

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

if __name__ == '__main__':
    """Test the DataPoint class."""
    JSON_FILE = 'input_files/output_ts_calc_debug.json'
    BASIS_FILE = 'input_files/def2-tzvppd.json'

    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

    data_point = HamiltonianDataset(JSON_FILE, BASIS_FILE)
    data_point.load_data()
    data_point.parse_basis_data()
    datapoint = data_point.get_data()

    # Graph the dataset

    for i, data in enumerate(datapoint):
        # Plot the graph for each datapoint
        plt.figure()
        graph = torch_geometric.utils.to_networkx(data) 
        nx.draw(graph, with_labels=True, )
        plt.savefig(os.path.join('output', f'graph_{i}.png'), dpi=300)
        