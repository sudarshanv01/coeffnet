import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

import copy

from collections import defaultdict

import numpy as np

from ase import data as ase_data

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

import itertools

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

    # minimal_basis_size is the number of basis functions in the minimal basis
    minimal_basis_size = 9

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
        return "hamiltonian_data.pt"

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

    def _get_atomic_number(cls, symbol):
        """Get the atomic number of an element based on the symbol."""
        return ase_data.atomic_numbers[symbol]

    def process(self):
        """Store the following information.
        1. (node features) Flattened list of the Hamiltonian matrix of diagonal-block elements.
        2. (edge features) Flattened list of the Hamiltonian matrix of off-diagonal-block elements.
        3. (mol_index) List of indices of the start and stop of each molecule.
        4. (basis_index) List of indices of the start and stop for each basis function within an atom.
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
            y = torch.tensor([y], dtype=DTYPE)

            # --- Get the global information
            global_information = input_data.pop("reaction_energy")
            global_information = torch.tensor([global_information], dtype=DTYPE)

            # Store the basis index
            all_basis_idx = []

            for molecule_id in input_data:

                state_fragment = input_data[molecule_id]["state_fragments"]

                molecule = input_data[molecule_id]["molecule"]
                if state_fragment == "initial_state":
                    molecules_in_reaction["reactants"].append(molecule)
                    molecules_in_reaction["reactants_index"].append(molecule_id)
                elif state_fragment == "final_state":
                    molecules_in_reaction["products"].append(molecule)
                    molecules_in_reaction["products_index"].append(molecule_id)

                # Get the index of the basis within the atom for each molecule
                # Iterate over the atoms in the molecule and store the indices of the
                # s, p and d basis functions
                basis_s = []
                basis_p = []
                basis_d = []
                # Store the grouping of the basis index of each atom in the molecule
                basis_atom = []

                tot_basis_idx = 0
                for atom_name in molecule.species:
                    # Get the basis functions for this atom
                    atomic_number = self._get_atomic_number(atom_name.symbol)
                    basis_functions = self.basis_info[atomic_number]
                    # Get the atom specific list
                    basis_s_ = []
                    basis_p_ = []
                    basis_d_ = []

                    # Store the initial basis functions
                    counter_tot_basis_idx = tot_basis_idx

                    for basis_function in basis_functions:
                        if basis_function == "s":
                            range_idx = list(range(tot_basis_idx, tot_basis_idx + 1))
                            basis_s_.append(range_idx)
                            tot_basis_idx += 1
                        elif basis_function == "p":
                            range_idx = list(range(tot_basis_idx, tot_basis_idx + 3))
                            basis_p_.append(range_idx)
                            tot_basis_idx += 3
                        elif basis_function == "d":
                            range_idx = list(range(tot_basis_idx, tot_basis_idx + 5))
                            basis_d_.append(range_idx)
                            tot_basis_idx += 5

                    # Store the basis index for this atom
                    basis_atom.append(list(range(counter_tot_basis_idx, tot_basis_idx)))

                    # Store the s,p and d basis functions for this atom
                    basis_s.append(basis_s_)
                    basis_p.append(basis_p_)
                    basis_d.append(basis_d_)

                hamiltonian = np.zeros((tot_basis_idx, tot_basis_idx, 2))
                for spin_index, spin in enumerate(["alpha", "beta"]):
                    if spin == "beta":
                        if input_data[molecule_id]["beta_fock_matrix"] == None:
                            # There is no computed beta spin, i.e. alpha and beta are the same
                            hamiltonian[..., 0] = input_data[molecule_id][
                                "alpha_fock_matrix"
                            ]
                        else:
                            hamiltonian[..., 0] = input_data[molecule_id][
                                spin + "_fock_matrix"
                            ]
                    else:
                        # Must always have an alpha spin
                        hamiltonian[..., 1] = input_data[molecule_id][
                            spin + "_fock_matrix"
                        ]

                # Split the Hamilonian into node and edge features
                all_basis_idx_ = [basis_s, basis_p, basis_d]
                all_basis_idx.append(all_basis_idx_)

                intra_atomic, coupling = self._split_hamiltonian(
                    hamiltonian, *all_basis_idx_
                )

                # Generate minimal basis representations of the intra_atomic
                # matrices by choosing the maximum value for the diagonal value
                # of each element in the matrix
                intra_atomic_mb = self.get_intra_atomic_mb(
                    intra_atomic, basis_atom, *all_basis_idx_
                )
                # Flatten the dimensions of the intra_atomic_mb aside from the first dimension
                intra_atomic_mb = intra_atomic_mb.reshape(intra_atomic_mb.shape[0], -1)

                molecule_info_collected["node_features"][
                    molecule_id
                ] = intra_atomic_mb.tolist()

                # Store the coupling information, which will be processed together
                # with all atoms in the molecule
                molecule_info_collected["edge_features"][molecule_id] = coupling

                # Store the s,p and d basis functions for this atom
                molecule_info_collected["basis_index"][molecule_id] = all_basis_idx_
                # Store the atom basis index grouping
                molecule_info_collected["atom_basis_index"][molecule_id] = basis_atom

            (
                edge_molecule_mapping,
                edge_internal_mol_mapping,
            ) = generate_graphs_by_method(
                graph_generation_method=self.graph_generation_method,
                molecules_in_reaction=molecules_in_reaction,
                molecule_info_collected=molecule_info_collected,
            )

            edge_index = molecule_info_collected["edge_index"]
            datapoint = DataPoint(
                pos=molecule_info_collected["pos"],
                edge_index=edge_index,
                x=molecule_info_collected["node_features"],
                y=y,
                global_attr=global_information,
                all_basis_idx=all_basis_idx,
            )

            # Generate the minimal basis representation of the edge attributes
            edge_features_mb = self.get_edge_features_mb(
                molecule_info_collected["edge_features"],
                datapoint.edge_index.detach().numpy(),
                edge_molecule_mapping,
                edge_internal_mol_mapping,
                molecule_info_collected["basis_index"],
            )

            # Flatten the dimensions of the edge_features_mb aside from the first dimension
            edge_features_mb = edge_features_mb.reshape(edge_features_mb.shape[0], -1)
            # Make edge features a torch tensor from an np array
            datapoint.edge_attr = torch.tensor(edge_features_mb, dtype=torch.float)

            logging.debug("Datapoint:")
            logging.debug(datapoint)

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        self.data = data
        self.slices = slices

        # Save the dataset
        torch.save(self.data, self.processed_paths[0])

    def get_edge_features_mb(
        self,
        edge_features,
        edge_index,
        edge_molecule_mapping,
        edge_internal_mol_mapping,
        basis_index,
    ):

        # Determine the edge index for all molecules participating in the reaction.
        row, col = edge_index

        # The edge attributes will be generated for each edge in the graph
        edge_features_mb = np.zeros(
            (len(row), self.minimal_basis_size, self.minimal_basis_size, 2)
        )

        # Simultaneously loop over all edges
        for vsk_, vrk_ in zip(row, col):

            # Get the molecule index of the edge
            mol_idx_vsk = edge_molecule_mapping[vsk_]
            mol_idx_vrk = edge_molecule_mapping[vrk_]

            # Both the edge indices must be in the same molecule
            # otherwise there is no coupling elements between them
            if mol_idx_vsk != mol_idx_vrk:
                continue

            # Get the edge_feature of the molecule in question
            edge_feature = edge_features[mol_idx_vsk]

            # Get the internal edge index of the edge
            vsk_internal = edge_internal_mol_mapping[vsk_]
            vrk_internal = edge_internal_mol_mapping[vrk_]

            # Split the basis_index into s, p and d contributions
            basis_s, basis_p, basis_d = basis_index[mol_idx_vrk]

            # Get the basis index of the s, p and d atoms
            basis_s_vsk = basis_s[vsk_internal]
            basis_p_vsk = basis_p[vsk_internal]
            basis_d_vsk = basis_d[vsk_internal]

            basis_vsk = [basis_s_vsk, basis_p_vsk, basis_d_vsk]
            basis_vsk_type = ["s", "p", "d"]

            basis_s_vrk = basis_s[vrk_internal]
            basis_p_vrk = basis_p[vrk_internal]
            basis_d_vrk = basis_d[vrk_internal]

            basis_vrk = [basis_s_vrk, basis_p_vrk, basis_d_vrk]
            basis_vrk_type = ["s", "p", "d"]

            # Iterate over every s, p and d basis function
            # between vsk and vrk and populate the minimal basis
            # representation of the edge features
            edge_features_mb_ = np.zeros(
                (self.minimal_basis_size, self.minimal_basis_size, 2)
            )

            # Iterate over the s, p and d basis functions
            for basis_vsk_type_, basis_vrk_type_ in itertools.product(
                basis_vsk_type, basis_vrk_type
            ):
                basis_idx_vsk = basis_vsk_type.index(basis_vsk_type_)
                basis_idx_vrk = basis_vrk_type.index(basis_vrk_type_)

                indices_to_populate = self.get_indices_to_populate(
                    basis_vsk_type[basis_idx_vsk], basis_vrk_type[basis_idx_vrk]
                )

                for basis_idx_vsk_ in basis_vsk[basis_idx_vsk]:
                    for basis_idx_vrk_ in basis_vrk[basis_idx_vrk]:

                        idx_chosen_edge = np.s_[
                            basis_idx_vsk_[0] : basis_idx_vsk_[-1] + 1,
                            basis_idx_vrk_[0] : basis_idx_vrk_[-1] + 1,
                            ...,
                        ]
                        chosen_edge_feature = edge_feature[idx_chosen_edge]

                        index_x, index_y = indices_to_populate
                        idx_chosen_mb = np.s_[
                            index_x[0] : index_x[-1] + 1,
                            index_y[0] : index_y[-1] + 1,
                            ...,
                        ]
                        edge_features_mb_[idx_chosen_mb] = chosen_edge_feature

            # Make edge_features_mb_ symmetric
            edge_features_mb_ = (
                edge_features_mb_ + edge_features_mb_.transpose(1, 0, 2)
            ) / 2
            edge_features_mb[vsk_, :, :, :] = edge_features_mb_

        return edge_features_mb

    def get_indices_to_populate(self, basis_type_vsk, basis_type_vrk):
        """Generate the indices to populate."""
        if basis_type_vsk == "s" and basis_type_vrk == "s":
            indices_to_populate = [[0], [0]]
        elif basis_type_vsk == "s" and basis_type_vrk == "p":
            indices_to_populate = [[0], [1, 2, 3]]
        elif basis_type_vsk == "s" and basis_type_vrk == "d":
            indices_to_populate = [[0], [4, 5, 6, 7, 8]]
        elif basis_type_vsk == "p" and basis_type_vrk == "p":
            indices_to_populate = [[1, 2, 3], [1, 2, 3]]
        elif basis_type_vsk == "p" and basis_type_vrk == "d":
            indices_to_populate = [[1, 2, 3], [4, 5, 6, 7, 8]]
        elif basis_type_vsk == "d" and basis_type_vrk == "d":
            indices_to_populate = [[4, 5, 6, 7, 8], [4, 5, 6, 7, 8]]
        elif basis_type_vsk == "d" and basis_type_vrk == "p":
            indices_to_populate = [[4, 5, 6, 7, 8], [1, 2, 3]]
        elif basis_type_vsk == "p" and basis_type_vrk == "s":
            indices_to_populate = [[1, 2, 3], [0]]
        elif basis_type_vsk == "d" and basis_type_vrk == "s":
            indices_to_populate = [[4, 5, 6, 7, 8], [0]]
        else:
            raise ValueError("Unknown basis type")

        return indices_to_populate

    def get_intra_atomic_mb(self, intra_atomic, basis_atom, basis_s, basis_p, basis_d):
        """Generate the minimal basis representation of the intra_atomic matrix by
        choosing the maximum value for the diagonal value of each element in the
        matrix.

        Parameters
        ----------
        intra_atomic : np.ndarray
            The intra atomic matrix.
        basis_atom : list
            The grouping of the basis index of each atom in the molecule.
        basis_s : list
            The s basis functions for each atom in the molecule.
        basis_p : list
            The p basis functions for each atom in the molecule.
        basis_d : list
            The d basis functions for each atom in the molecule.

        Returns
        -------
        intra_atomic_mb : np.ndarray
            The minimal basis representation of the intra_atomic matrix.
        """

        num_atoms = len(basis_atom)
        intra_atomic_mb = np.zeros(
            (num_atoms, self.minimal_basis_size, self.minimal_basis_size, 2)
        )

        for atom_idx, atom_basis in enumerate(basis_atom):

            # Get the basis functions for this atom
            basis_s_atom_idx = basis_s[atom_idx]
            basis_p_atom_idx = basis_p[atom_idx]
            basis_d_atom_idx = basis_d[atom_idx]

            # Get the minimal_basis s-matrix
            atom_basis_s = np.zeros((1, 1, 2))
            for idx_, basis_s_atom_idx_ in enumerate(basis_s_atom_idx):
                atom_basis_s_ = intra_atomic[basis_s_atom_idx_, :][:, basis_s_atom_idx_]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_s = atom_basis_s_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_s_[..., 0]))
                spin_down_max = np.max(np.diag(atom_basis_s_[..., 1]))

                if spin_up_max > np.max(np.diag(atom_basis_s[..., 0])):
                    atom_basis_s[..., 0] = atom_basis_s_[..., 0]
                if spin_down_max > np.max(np.diag(atom_basis_s[..., 1])):
                    atom_basis_s[..., 1] = atom_basis_s_[..., 1]
            # Add the s-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, 0, 0, :] = atom_basis_s

            # Get the minimal_basis p-matrix
            atom_basis_p = np.zeros((3, 3, 2))
            for idx_, basis_p_atom_idx_ in enumerate(basis_p_atom_idx):
                atom_basis_p_ = intra_atomic[basis_p_atom_idx_, :][:, basis_p_atom_idx_]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_p = atom_basis_p_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_p_[..., 0]))
                spin_down_max = np.max(np.diag(atom_basis_p_[..., 1]))

                if spin_up_max > np.max(np.diag(atom_basis_p[..., 0])):
                    atom_basis_p[..., 0] = atom_basis_p_[..., 0]
                if spin_down_max > np.max(np.diag(atom_basis_p[..., 1])):
                    atom_basis_p[..., 1] = atom_basis_p_[..., 1]
            # Add the p-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, 1:4, 1:4, :] = atom_basis_p

            # Get the minimal_basis d-matrix
            atom_basis_d = np.zeros((5, 5, 2))
            for idx_, basis_d_atom_idx_ in enumerate(basis_d_atom_idx):
                atom_basis_d_ = intra_atomic[basis_d_atom_idx_, :][:, basis_d_atom_idx_]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_d = atom_basis_d_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_d_[..., 0]))
                spin_down_max = np.max(np.diag(atom_basis_d_[..., 1]))

                if spin_up_max > np.max(np.diag(atom_basis_d[..., 0])):
                    atom_basis_d[..., 0] = atom_basis_d_[..., 0]
                if spin_down_max > np.max(np.diag(atom_basis_d[..., 1])):
                    atom_basis_d[..., 1] = atom_basis_d_[..., 1]
            # Add the d-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, 4:9, 4:9, :] = atom_basis_d

        return intra_atomic_mb

    def _split_hamiltonian(cls, matrix_, *args):
        """Split the matrix into node and edge features."""

        h_matrix = np.array(matrix_)

        idx_basis = []
        for arg in args:
            idx_basis += arg

        flatten_basis = []
        for idx_basis_ in idx_basis:
            flatten_basis += idx_basis_

        elemental_matrix = np.zeros(h_matrix.shape)
        coupling_matrix = np.zeros(h_matrix.shape)

        # Populate the elemental matrix first by splicing
        for idx in flatten_basis:
            sp_idx = np.s_[idx[0] : idx[-1] + 1, idx[0] : idx[-1] + 1, ...]
            elemental_matrix[sp_idx] = h_matrix[sp_idx]

        # Populate the coupling matrix by subtracting the total matrix from the elemental matrix
        coupling_matrix = h_matrix - elemental_matrix

        return elemental_matrix, coupling_matrix
