"""Generate all the Data objects needed to predict the TS Hamiltonian."""
import sys
import json
from pathlib import Path
from typing import Dict, Union
import logging

import numpy as np

import torch
from torch_geometric.data import InMemoryDataset

from pymatgen.core.structure import Molecule

from e3nn import o3
from e3nn.util.test import equivariance_error

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
        }
        # Store the l for each basis function
        self.l_basis = {'s': 0, 'p': 1, 'd': 2}

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
                        logging.warning(e)
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
                overall_index += basis_element
                start_stop.append([overall_index-basis_element, overall_index]) 
            list_of_indices.append(start_stop)
        return list_of_indices
    
    def get_indices_atom_basis(cls, indices_of_basis):
        """Get the starting and ending index of each atom. This list
        serves as a sort of `atomic-basis` representation of the 
        Hamiltonian matrix."""
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
                positions_molecule = [a.coords for a in molecule] 
                # Store the positions; just concatenate the lists for now
                coords.append(positions_molecule)

                # Get the type of basis functions
                # TODO: Implement cases where atom specific basis functions may be used.
                # Determine the atom wise basis functions
                atom_basis_functions = [self.basis_info[atom_name.symbol] for atom_name in molecule.species]

                # ---------- Get the irreps for the diagonal elements ----------
                irreps_array = [] 
                # Get the irrep for the diagonal blocks
                for atom_basis in atom_basis_functions:
                    for basis_function in atom_basis:
                        irreps_array.append(self.l_basis[basis_function])

                # Get the indices of the basis functions
                tot_number_of_s = irreps_array.count(0) 
                tot_number_of_p = irreps_array.count(1)
                tot_number_of_d = irreps_array.count(2)
                # tot_number_of_f = irreps_array.count(3)
                # tot_number_of_g = irreps_array.count(4)
                assert tot_number_of_s + tot_number_of_p + tot_number_of_d == len(irreps_array)

                if state == 'initial_state':
                    # Store the initial state data
                    irrep = f"{tot_number_of_s}x0e + {tot_number_of_p}x1o + {tot_number_of_d}x2e"
                elif state == 'final_state':
                    irrep_check = f"{tot_number_of_s}x0e + {tot_number_of_p}x1o + {tot_number_of_d}x2e"
                    assert irrep_check == irrep, f"The initial and final state irrep do not match: {irrep_check} != {irrep}"

                # Get the number of basis functions for each atom
                number_basis_functions = self._string_basis_to_number(atom_basis_functions)

                # Sum up all the basis functions
                total_basis = np.sum([np.sum(a) for a in number_basis_functions])
                logging.info(f'Total basis functions: {total_basis}')

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
                    logging.info(f'Hamiltonian shape: {hamiltonian.shape}')

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
                edge_attr[state] = edge_H
                logging.debug(f'{state} node features shape: {node_H.shape}')
                logging.debug(f'{state} edge features shape: {edge_H.shape}')
            
            # Store difference between the `transition_state` energy and the 
            # `initial_state` energy as the prediction quantity.
            datapoint_y = self.input_data[_id]['transition_state']['total_energy'] -\
                     self.input_data[_id]['initial_state']['total_energy']
            datapoint_y = torch.tensor(datapoint_y, dtype=torch.float)

            # Perform a FullyConnectedTensorProduct between initial and final state
            # to ensure that the output contains irrep of the same dimension
            minimal_basis_H = o3.FullyConnectedTensorProduct(
                irreps_in1=irrep,
                irreps_in2=irrep,
                irreps_out='1x0e + 1x1o + 1x2e',
            )
            logging.info(f"Chosen irreducible representation for the diagonal elements are: {irrep}")

            # Check if to see if everything is okay in terms of equivariance
            equivariance_error(minimal_basis_H, args_in=[datapoint_x['initial_state'], datapoint_x['final_state']],
                                irreps_in=[irrep, irrep], irreps_out='1x0e + 1x1o + 1x2e', )
            # Check the equivariance of the edge attributes
            equivariance_error(minimal_basis_H, args_in=[edge_attr['initial_state'], edge_attr['final_state']],
                                irreps_in=[irrep, irrep], irreps_out='1x0e + 1x1o + 1x2e', )
            logging.debug('Equivariance check passed.')

            # Perform the transformation
            datapoint_x = minimal_basis_H(datapoint_x['initial_state'], datapoint_x['final_state'])
            logging.info(f'Transformed datapoint_x shape: {datapoint_x.shape}')
            edge_attributes = minimal_basis_H(edge_attr['initial_state'], edge_attr['final_state'])
            logging.info(f'Transformed edge_attributes shape: {edge_attributes.shape}')

            # Interpolate between the coords of the initial and final state
            coords_interpolated = self.interpolate_coords(coords)
            coords_interpolated = np.array(coords_interpolated)
            coords_interpolated = torch.tensor(coords_interpolated, dtype=torch.float)

            # Store the node and edge features
            datapoint = DataPoint(
                pos=coords_interpolated,
                x=datapoint_x,
                y=datapoint_y,
                edge_attr=edge_attributes,
            )

            # Add to all data
            all_data.append(datapoint)

        # Return a list of all the datapoints stored in 
        # DataPoint objects.
        return all_data
    
    def interpolate_coords(self, coords):
        """For a list containing two atoms, interpolate between
        the two structures to determine the likely transition state
        structure. This structure will be passed to the DataPoint object.
        """
        initial_coords, final_coords = coords
        # Iterapolate between the two structures
        coords_interpolated = []
        for i in range(len(initial_coords)):
            coords_interpolated.append(
                (initial_coords[i] + final_coords[i]) / 2
            )
        return coords_interpolated

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
    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

    data_point = HamiltonianDataset(JSON_FILE, BASIS_FILE, BASIS_SET)
    data_point.load_data()
    data_point.validate_json_input()
    data_point.get_data()

