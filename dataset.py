"""Generate all the Data objects needed to predict the TS Hamiltonian."""
import json
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset

from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, Union
from pymatgen.core.structure import Molecule

from data import DataPoint

class HamiltonianDataset(InMemoryDataset):
    """Implement simple dataset to read in different
    Hamiltonians for the initial and final state and 
    store them as a Data object."""

    def __init__(self, 
        filename: Union[str, Path],
        basis_info: Dict[str, int],
        basis_set: str,
    ):
        self.filename = filename
        self.basis_file = basis_info
        self.basis_set = basis_set

        # Read in the entire basis_file json 
        with open(self.basis_file) as f:
            self.basis_info = json.load(f)

        # Only use the basis_info for the basis set
        self.basis_info = self.basis_info[self.basis_set]

        # Store the converter to convert the basis functions
        self.BASIS_CONVERTER = {
            "s": 1, 
            "p": 3,
            "d": 5,
            "f": 7,
        }

        super().__init__()
    
    def validate_json_input(self):
        """Validate that the loaded json data has the right structure."""
        # Make sure that 'initial_state', 'transition_state', 'final_state'
        # are all present in the json file for each _id
        # Do this by iterating over all the _id's and checking that
        # all three states are present.
        input_data_copy = self.input_data.copy()
        for _id in input_data_copy:
            for state in ['initial_state', 'transition_state', 'final_state']:
                if state not in self.input_data[_id]:
                    try:
                        raise ValueError(f'{state} not found in json file for _id {_id}')
                    except Exception as e:
                        print(e)
                        # Remove the offending _id from the json file
                        del self.input_data[_id]

    def load_data(self):
        """Load the json file with the data."""
        with open(self.filename) as f:
            self.input_data = json.load(f)

    def get_indices_of_basis(cls, atom_basis_functions): 
        """Generate a list containing the start and stop index of the 
        basis function corresponding to each orbital."""
        # Create a list of index based on the atom wise basis functions
        list_of_indices = []
        # Start labelling index at 0
        overall_index = 0
        for basis_list in atom_basis_functions:
            # Add the index of the atom to the list
            start_stop = []
            for basis_element in basis_list:
                # start_stop.append(overall_index)
                overall_index += basis_element
                start_stop.append([overall_index-basis_element, overall_index]) 
            list_of_indices.append(start_stop)
        return list_of_indices
    
    def get_indices_atom_basis(cls, indices_of_basis):
        """Get the starting and ending index of each atom. This list
        serves as a sort of `atomic-basis` representation of the 
        (most-likely) Hamiltonian matrix."""
        indices_atom_basis = []
        for basis_list in indices_of_basis:
            flattened_basis_list = [item for sublist in basis_list for item in sublist]
            # Assumes that the basis index are in order
            # TODO: Check if there is a better way to do this
            indices_atom_basis.append([flattened_basis_list[0], flattened_basis_list[-1]])
        return indices_atom_basis 

    def _string_basis_to_number(self, string_basis_list):
        """Convert a list of lists containing string elements to
        the corresponding number of basis functions."""
        # Convert the string basis to the number of basis functions
        number_basis_list = []
        for string_basis in string_basis_list:
            number_of_elements = [self.BASIS_CONVERTER[element] for element in string_basis]
            number_basis_list.append(number_of_elements)
        return number_basis_list

    def get_data(self):
        """Loads the data from the json file. Calls
        the relevant methods to split the matrix into
        the relevant structures. That is, the diagonal
        elements are kept as node features and the 
        off-diagonal elements are stored as edge features.
        
        The structure of the data in terms of levels of a
        json-stored dict is:
        1. initial_state, transition_state, final_state
        2. Hamiltonian, Eigenvalues, Overlap Matrix,
        for both alpha and beta spin.
        Mapping of the index of the atom with the index of
        the Hamiltonian, energies referenced
        to the initial state and the structure in pymatgen format.
        """ 

        # Iterate over all input data and store the relevant objects
        all_data = []
        for _id in self.input_data:
            # Perform a separate calculation for each state
            # and the simply concatenate the data to make a single
            # entity comprising of the `initial_state` and `final_state`
            # data. The prediction quantity with the TS energy barrier.
            # Store all of the quantities to pass to the model.
            coords = []
            datapoint_x = {}
            edge_attr = {}
            for state in ['initial_state', 'final_state']:
                # The initial and final state will be inputs to the model
                # and the transitions state will be the output.
                molecule = self.input_data[_id][state]['molecule']
                molecule = Molecule.from_dict(molecule)
                # positions_molecule = molecule.coords
                # Store the positions; just concatenate the lists for now
                # coords.append(positions_molecule)

                # Get the type of basis functions
                # TODO: Implement cases where atom specific basis functions may be used.
                # Determine the atom wise basis functions
                atom_basis_functions = [self.basis_info[atom_name.symbol] for atom_name in molecule.species]

                # Get the number of basis functions for each atom
                number_basis_functions = self._string_basis_to_number(atom_basis_functions)

                # Sum up all the basis functions
                total_basis = np.sum([np.sum(a) for a in number_basis_functions])
                print(f'Total basis functions: {total_basis}')

                # Get the indices of the basis functions
                indices_of_basis = self.get_indices_of_basis(number_basis_functions)

                # Get the diagonal elements consisting of each atom
                indices_atom_basis = self.get_indices_atom_basis(indices_of_basis)

                # Get the Hamiltonian and partition it into diagonal and off-diagonal
                # elements. Diagonal elements may be blocks of s-s, s-p, ... 
                # which consist of the intra-atomic basis functions. The 
                # off-diagonal elements are for interactions between 
                # the inter-atomic basis functions.
                # At the moment we will just average the spin dependence
                # out of the alpha and beta spin components.
                node_H = torch.zeros(total_basis, total_basis)
                edge_H = torch.zeros(total_basis, total_basis)
                for spin in ['alpha', 'beta']:
                    hamiltonian = self.input_data[_id][state][spin + '_H']

                    # Make sure the Hamiltonian is a tensor
                    hamiltonian = torch.tensor(hamiltonian)[0,...]
                    print(f'Hamiltonian shape: {hamiltonian.shape}')

                    # Split the matrix into node and edge features
                    node_H_spin, edge_H_spin = self.split_matrix_node_edge(hamiltonian, indices_atom_basis)
                    node_H += node_H_spin
                    edge_H += edge_H_spin
                # Average the spin contributions
                # TODO: Make this more robust by including different spin contributions
                node_H /= 2
                edge_H /= 2

                # Store the node features as a dictionary
                datapoint_x[state] = node_H
            
            # Store difference between the `transition_state` energy and the 
            # `initial_state` energy as the prediction quantity.
            datapoint_y = self.input_data[_id]['transition_state']['total_energy'] -\
                     self.input_data[_id]['initial_state']['total_energy']
            datapoint_y = torch.tensor(datapoint_y, dtype=torch.float)

            # Store the node and edge features
            datapoint = DataPoint(
                # pos=coords,
                x=datapoint_x,
                y=datapoint_y,
            )

            # Add to all data
            all_data.append(datapoint)

        # Return a list of all the datapoints stored in 
        # DataPoint objects.
        print(all_data[0])
        return all_data

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
    JSON_FILE = 'input_files/predict_data_ML.json'
    BASIS_FILE = 'input_files/basis_info.json'
    BASIS_SET = 'def2-mSVP'

    data_point = HamiltonianDataset(JSON_FILE, BASIS_FILE, BASIS_SET)
    data_point.load_data()
    data_point.validate_json_input()
    data_point.get_data()

