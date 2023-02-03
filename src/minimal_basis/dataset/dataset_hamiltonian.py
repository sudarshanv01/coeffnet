import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

import copy

from collections import defaultdict

import numpy as np

from ase import data as ase_data

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

import itertools

from minimal_basis.data.data_hamiltonian import HamiltonianDataPoint as DataPoint

logger = logging.getLogger(__name__)


class HamiltonianDataset(InMemoryDataset):
    """Dataset for the Hamiltonian for all species in a reaction."""

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
    ):
        self.filename = filename
        self.basis_file = basis_file

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

        # Find the number of global features based on the first data
        self.num_global_features = self.data.num_global_features[0].item()

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

    def get_basis_index(self, molecule):
        """Get the basis index for each atom in the atom in the molecule."""

        # Basis information
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

        # Split the Hamilonian into node and edge features
        all_basis_idx = [basis_s, basis_p, basis_d]

        return all_basis_idx, basis_atom

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

            data_to_store = defaultdict(dict)

            input_data = copy.deepcopy(input_data_)

            fock_matrices = input_data["fock_matrices"]
            overlap_matrices = input_data["overlap_matrices"]
            eigenvalues = input_data["eigenvalues"]
            final_energy = input_data["final_energy"]

            fock_matrices = np.array(fock_matrices)
            overlap_matrices = np.array(overlap_matrices)
            eigenvalues = np.array(eigenvalues)
            final_energy = np.array(final_energy)

            structures = input_data["structures"]
            if isinstance(structures, dict):
                structures = [Molecule.from_dict(structure) for structure in structures]

            # The basis is assumed to be the same across all structures
            all_basis_idx, basis_atom = self.get_basis_index(structures[0])

            for idx_state, state in enumerate(input_data["state"]):

                if state == "transition_state":
                    y = final_energy[idx_state]
                    continue

                logger.debug(f"Processing state {state}")

                data_to_store["global_attr"][state] = np.array(
                    [final_energy[idx_state]]
                )
                data_to_store["num_global_features"][state] = data_to_store[
                    "global_attr"
                ][state].shape[0]

                molecule = structures[idx_state]
                molecule_graph = MoleculeGraph.with_local_env_strategy(
                    molecule, OpenBabelNN()
                )
                positions = molecule_graph.molecule.cart_coords
                data_to_store["pos"][state] = positions

                edges_for_graph = molecule_graph.graph.edges
                edges_for_graph = [list(edge[:-1]) for edge in edges_for_graph]
                edge_index = np.array(edges_for_graph).T
                data_to_store["edge_index"][state] = edge_index

                fock_matrix_state = fock_matrices[idx_state]

                intra_atomic, coupling = self._split_hamiltonian(
                    fock_matrix_state, *all_basis_idx
                )

                intra_atomic_mb = self.get_intra_atomic_mb(
                    intra_atomic, basis_atom, *all_basis_idx
                )

                # Flatten the dimensions of the intra_atomic_mb aside from the first dimension
                intra_atomic_mb = intra_atomic_mb.reshape(intra_atomic_mb.shape[0], -1)

                data_to_store["node_features"][state] = intra_atomic_mb.tolist()

                # Store the s,p and d basis functions for this atom
                data_to_store["basis_index"][state] = all_basis_idx

                # Store the atom basis index grouping
                data_to_store["atom_basis_index"][state] = basis_atom

                # Generate the minimal basis representation of the edge attributes
                edge_features_mb = self.get_edge_features_mb(
                    coupling, edge_index, *all_basis_idx
                )

                # Flatten the dimensions of the edge_features_mb aside from the first dimension
                edge_features_mb = edge_features_mb.reshape(
                    edge_features_mb.shape[0], -1
                )

                data_to_store["edge_features"][state] = edge_features_mb.tolist()

            datapoint = DataPoint(
                pos=data_to_store["pos"],
                edge_index=data_to_store["edge_index"],
                x=data_to_store["node_features"],
                edge_attr=data_to_store["edge_features"],
                global_attr=data_to_store["global_attr"],
                y=y,
                all_basis_idx=all_basis_idx,
                num_global_features=data_to_store["num_global_features"][
                    "initial_state"
                ],
            )

            logging.debug("Datapoint:")
            logging.debug(datapoint)

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_edge_features_mb(
        self,
        edge_features,
        edge_index,
        basis_s,
        basis_p,
        basis_d,
    ):
        # Determine the edge index for all molecules participating in the reaction.
        row, col = edge_index

        # The edge attributes will be generated for each edge in the graph
        edge_features_mb = np.zeros(
            (len(row), 2, self.minimal_basis_size, self.minimal_basis_size)
        )

        # Simultaneously loop over all edges
        for vsk_, vrk_ in zip(row, col):

            # Get the basis index of the s, p and d atoms
            basis_s_vsk = basis_s[vsk_]
            basis_p_vsk = basis_p[vsk_]
            basis_d_vsk = basis_d[vsk_]

            basis_vsk = [basis_s_vsk, basis_p_vsk, basis_d_vsk]
            basis_vsk_type = ["s", "p", "d"]

            basis_s_vrk = basis_s[vrk_]
            basis_p_vrk = basis_p[vrk_]
            basis_d_vrk = basis_d[vrk_]

            basis_vrk = [basis_s_vrk, basis_p_vrk, basis_d_vrk]
            basis_vrk_type = ["s", "p", "d"]

            # Iterate over every s, p and d basis function
            # between vsk and vrk and populate the minimal basis
            # representation of the edge features
            edge_features_mb_ = np.zeros(
                (2, self.minimal_basis_size, self.minimal_basis_size)
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
                            ...,
                            basis_idx_vsk_[0] : basis_idx_vsk_[-1] + 1,
                            basis_idx_vrk_[0] : basis_idx_vrk_[-1] + 1,
                        ]
                        chosen_edge_feature = edge_features[idx_chosen_edge]

                        index_x, index_y = indices_to_populate
                        idx_chosen_mb = np.s_[
                            ...,
                            index_x[0] : index_x[-1] + 1,
                            index_y[0] : index_y[-1] + 1,
                        ]
                        edge_features_mb_[idx_chosen_mb] = chosen_edge_feature

            # Make edge_features_mb_ symmetric
            edge_features_mb_ = (
                edge_features_mb_ + edge_features_mb_.transpose(0, 2, 1)
            ) / 2
            # Find the index of vsk_ in row
            idx_store = np.where(row == vsk_)[0][0]
            edge_features_mb[idx_store, :, :, :] = edge_features_mb_

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
            (num_atoms, 2, self.minimal_basis_size, self.minimal_basis_size)
        )

        for atom_idx, atom_basis in enumerate(basis_atom):

            # Get the basis functions for this atom
            basis_s_atom_idx = basis_s[atom_idx]
            basis_p_atom_idx = basis_p[atom_idx]
            basis_d_atom_idx = basis_d[atom_idx]

            # Get the minimal_basis s-matrix
            atom_basis_s = np.zeros((2, 1, 1))
            for idx_, basis_s_atom_idx_ in enumerate(basis_s_atom_idx):
                atom_basis_s_ = intra_atomic[:, basis_s_atom_idx_, :][
                    :, :, basis_s_atom_idx_
                ]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_s = atom_basis_s_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_s_[0, ...]))
                spin_down_max = np.max(np.diag(atom_basis_s_[1, ...]))

                if spin_up_max > np.max(np.diag(atom_basis_s[0, ...])):
                    atom_basis_s[0, ...] = atom_basis_s_[0, ...]
                if spin_down_max > np.max(np.diag(atom_basis_s[1, ...])):
                    atom_basis_s[1, ...] = atom_basis_s_[1, ...]
            # Add the s-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, :, 0:1, 0:1] = atom_basis_s

            # Get the minimal_basis p-matrix
            atom_basis_p = np.zeros((2, 3, 3))
            for idx_, basis_p_atom_idx_ in enumerate(basis_p_atom_idx):
                atom_basis_p_ = intra_atomic[:, basis_p_atom_idx_, :][
                    :, :, basis_p_atom_idx_
                ]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_p = atom_basis_p_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_p_[0, ...]))
                spin_down_max = np.max(np.diag(atom_basis_p_[1, ...]))

                if spin_up_max > np.max(np.diag(atom_basis_p[0, ...])):
                    atom_basis_p[0, ...] = atom_basis_p_[0, ...]
                if spin_down_max > np.max(np.diag(atom_basis_p[1, ...])):
                    atom_basis_p[1, ...] = atom_basis_p_[1, ...]

            # Add the p-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, :, 1:4, 1:4] = atom_basis_p

            # Get the minimal_basis d-matrix
            atom_basis_d = np.zeros((2, 5, 5))
            for idx_, basis_d_atom_idx_ in enumerate(basis_d_atom_idx):
                atom_basis_d_ = intra_atomic[:, basis_d_atom_idx_, :][
                    :, :, basis_d_atom_idx_
                ]
                if idx_ == 0:
                    # First iteration, set the values
                    atom_basis_d = atom_basis_d_
                    continue

                spin_up_max = np.max(np.diag(atom_basis_d_[0, ...]))
                spin_down_max = np.max(np.diag(atom_basis_d_[1, ...]))

                if spin_up_max > np.max(np.diag(atom_basis_d[0, ...])):
                    atom_basis_d[0, ...] = atom_basis_d_[0, ...]
                if spin_down_max > np.max(np.diag(atom_basis_d[..., 1])):
                    atom_basis_d[1, ...] = atom_basis_d_[1, ...]

            # Add the d-matrix to the minimal basis matrix
            intra_atomic_mb[atom_idx, :, 4:9, 4:9] = atom_basis_d

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
