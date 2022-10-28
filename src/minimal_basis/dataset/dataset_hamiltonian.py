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
                tot_basis_idx = 0
                for atom_name in molecule.species:
                    # Get the basis functions for this atom
                    atomic_number = self._get_atomic_number(atom_name.symbol)
                    basis_functions = self.basis_info[atomic_number]
                    # Get the atom specific list
                    basis_s_ = []
                    basis_p_ = []
                    basis_d_ = []
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

                node_features, edge_features = self._split_hamiltonian(
                    hamiltonian, *all_basis_idx_
                )

                molecule_info_collected["node_features"][molecule_id] = node_features
                molecule_info_collected["edge_features"][molecule_id] = edge_features

            (
                edge_molecule_mapping,
                edge_internal_mol_mapping,
            ) = generate_graphs_by_method(
                graph_generation_method=self.graph_generation_method,
                molecules_in_reaction=molecules_in_reaction,
                molecule_info_collected=molecule_info_collected,
            )

            # Make the node and edge features the same matrix size by padding
            # the smaller matrix with zeros
            node_features = self.pad_features(molecule_info_collected["node_features"])
            edge_features = self.pad_features(molecule_info_collected["edge_features"])

            datapoint = DataPoint(
                pos=molecule_info_collected["pos"],
                edge_index=molecule_info_collected["edge_index"],
                edge_attr=edge_features,
                x=node_features,
                y=y,
                global_attr=global_information,
                all_basis_idx=all_basis_idx,
            )

            logging.debug("Datapoint:")
            logging.debug(datapoint)

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        self.data = data
        self.slices = slices

        self.data = datapoint_list

        # Save the dataset
        torch.save(self.data, self.processed_paths[0])

    def pad_features(cls, features):
        """Make all features the same size by padding the smallest
        matrices with zeros."""

        shape_ = 0
        for molecule_id in features:
            shape = np.array(features[molecule_id]).shape
            if shape[0] > shape_:
                shape_ = shape[0]

        updated_features = {}
        for molecule_id in features:
            shape = np.array(features[molecule_id]).shape
            if shape[0] < shape_:
                pad = shape_ - shape[0]
                original_matrix = np.array(features[molecule_id])
                padded_matrix = np.pad(
                    original_matrix,
                    ((0, pad), (0, pad), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                updated_features[molecule_id] = padded_matrix
            else:
                updated_features[molecule_id] = features[molecule_id]

        return updated_features

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
