import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

import copy

from collections import defaultdict

import numpy as np
import numpy.typing as npt

from scipy.linalg import eigh

from ase import data as ase_data

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import chemcoord as cc

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

from e3nn import o3

from minimal_basis.data.data_reaction import ReactionDataPoint as DataPoint
from minimal_basis.predata.predata_classifier import GenerateParametersClassifier
from minimal_basis.data.data_reaction import ModifiedCoefficientMatrix

logger = logging.getLogger(__name__)


class ReactionDataset(InMemoryDataset):

    mu = 0.5
    sigma = 0.25
    alpha = 1.0

    def __init__(
        self,
        filename: Union[str, Path] = None,
        basis_filename: Union[str, Path] = None,
        root: str = None,
        transform: str = None,
        pre_transform: bool = None,
        pre_filter: bool = None,
        max_s_functions: int = None,
        max_p_functions: int = None,
        max_d_functions: int = None,
        use_minimal_basis_node_features: bool = False,
        idx_eigenvalue: int = 0,
    ):
        """Dataset for the reaction.
        
        Args:
            filename (Union[str, Path], optional): Path to the json file with the data. Defaults to None.
            basis_filename (Union[str, Path], optional): Path to the json file with the basis information.\
                  The format of the file must be the "JSON" option from basissetexchange.com. Defaults to None.
            root (str, optional): Root directory. Defaults to None.
            transform (str, optional): Transform to apply. Defaults to None.
            pre_transform (bool, optional): Pre-transform to apply. Defaults to None.
            pre_filter (bool, optional): Pre-filter to apply. Defaults to None.
            max_s_functions (int, optional): Maximum number of s functions to be used in constructing node\
                features. Defaults to None.
            max_p_functions (int, optional): Maximum number of p functions to be used in constructing node\
                features. Defaults to None.
            max_d_functions (int, optional): Maximum number of d functions to be used in constructing node\
                features. Defaults to None.
            use_minimal_basis_node_features (bool, optional): Whether to use minimal basis node features.\
                If true, then only _max_ coefficient matrices for the s and p functions are used to construct\
                a minimal basis representation consisting of only 1 s and 1 p function. Defaults to False.
            idx_eigenvalue (int, optional): Index of the eigenvalue to be used for the reaction. If set at 0
                then the smallest occupied eigenvalue is used (i.e. smallest negative number). Any positive or
                negative number will be referenced to this zero value.
        """

        self.filename = filename
        self.root = root
        self.basis_filename = basis_filename
        self.max_s_functions = max_s_functions
        self.max_p_functions = max_p_functions
        self.max_d_functions = max_d_functions
        self.use_minimal_basis_node_features = use_minimal_basis_node_features
        self.idx_eigenvalue = idx_eigenvalue

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "reaction_data.json"

    @property
    def processed_file_names(self):
        return "reaction_data.pt"

    def download(self):

        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")

        with open(self.basis_filename) as f:
            self.basis_info_raw = json.load(f)
        logger.info("Successfully loaded json file with basis information.")
        logger.info("Parsing basis information.")

    def process(self):

        datapoint_list = []

        for reaction_idx, input_data_ in enumerate(self.input_data):

            data_to_store = defaultdict(dict)

            input_data = copy.deepcopy(input_data_)

            eigenvalues = input_data["eigenvalues"]
            final_energy = input_data["final_energy"]
            coeff_matrices = input_data["coeff_matrices"]
            states = input_data["state"]

            eigenvalues = np.array(eigenvalues)
            final_energy = np.array(final_energy)
            coeff_matrices = np.array(coeff_matrices)

            structures = input_data["structures"]

            if isinstance(structures[0], dict):
                structures = [Molecule.from_dict(structure) for structure in structures]

            for idx_state, state in enumerate(states):

                logger.debug(f"Processing state {state}")

                data_to_store["total_energies"][state] = final_energy[idx_state]

                molecule = structures[idx_state]
                molecule_graph = MoleculeGraph.with_local_env_strategy(
                    molecule, OpenBabelNN()
                )
                positions = molecule_graph.molecule.cart_coords
                data_to_store["pos"][state] = positions

                species_in_molecule = molecule_graph.molecule.species
                species_in_molecule = [
                    ase_data.chemical_symbols.index(species.symbol)
                    for species in species_in_molecule
                ]
                species_in_molecule = np.array(species_in_molecule)
                species_in_molecule = species_in_molecule.reshape(-1, 1)
                data_to_store["species"][state] = species_in_molecule

                edges_for_graph = molecule_graph.graph.edges
                edges_for_graph = [list(edge[:-1]) for edge in edges_for_graph]
                edge_index = np.array(edges_for_graph).T
                data_to_store["edge_index"][state] = edge_index

                alpha_coeff_matrix = coeff_matrices[idx_state, 0]
                alpha_eigenvalues = eigenvalues[idx_state, 0]
                selected_eigenval = alpha_eigenvalues[alpha_eigenvalues < 0]
                selected_eigenval = np.sort(selected_eigenval)
                selected_eigenval = selected_eigenval[-1]
                selected_eigenval_index = np.where(
                    alpha_eigenvalues == selected_eigenval
                )[0][0]
                selected_eigenval_index = selected_eigenval_index + self.idx_eigenvalue
                logger.debug(
                    f"Selected eigenvalue {selected_eigenval} with index {selected_eigenval_index}"
                )

                coeff_matrix = ModifiedCoefficientMatrix(
                    molecule_graph=molecule_graph,
                    basis_info_raw=self.basis_info_raw,
                    coefficient_matrix=alpha_coeff_matrix,
                    store_idx_only=selected_eigenval_index,
                    set_to_absolute=True,
                    max_s_functions=self.max_s_functions,
                    max_p_functions=self.max_p_functions,
                    max_d_functions=self.max_d_functions,
                    use_minimal_basis_node_features=self.use_minimal_basis_node_features,
                )

                node_features = coeff_matrix.get_node_features()
                node_features = node_features.reshape(node_features.shape[0], -1)
                data_to_store["node_features"][state] = node_features
                basis_mask = coeff_matrix.basis_mask
                data_to_store["basis_mask"][state] = basis_mask

                minimal_basis_irrep = coeff_matrix.minimal_basis_irrep

            initial_states_structure = structures[states.index("initial_state")]
            final_states_structure = structures[states.index("final_state")]

            instance_generate = GenerateParametersClassifier()
            interpolated_transition_state_pos = (
                instance_generate.get_interpolated_transition_state_positions(
                    initial_states_structure.cart_coords,
                    final_states_structure.cart_coords,
                    mu=self.mu,
                    sigma=self.sigma,
                    alpha=self.alpha,
                    deltaG=final_energy[states.index("final_state")]
                    - final_energy[states.index("initial_state")],
                )
            )
            interpolated_transition_state_structure = Molecule(
                initial_states_structure.species,
                interpolated_transition_state_pos,
                charge=initial_states_structure.charge,
                spin_multiplicity=initial_states_structure.spin_multiplicity,
            )

            p, p_prime = instance_generate.get_p_and_pprime(
                mu=self.mu, sigma=self.sigma, alpha=self.alpha
            )

            data_to_store["pos"][
                "interpolated_transition_state"
            ] = interpolated_transition_state_pos

            interpolated_transition_state_structure_graph = (
                MoleculeGraph.with_local_env_strategy(
                    interpolated_transition_state_structure, OpenBabelNN()
                )
            )

            edges_for_graph = interpolated_transition_state_structure_graph.graph.edges
            interpolated_transition_state_structure_edge_index = [
                list(edge[:-1]) for edge in edges_for_graph
            ]

            interpolated_transition_state_structure_edge_index = np.array(
                interpolated_transition_state_structure_edge_index
            ).T
            data_to_store["edge_index"][
                "interpolated_transition_state"
            ] = interpolated_transition_state_structure_edge_index

            datapoint = DataPoint(
                pos=data_to_store["pos"],
                edge_index=data_to_store["edge_index"],
                x=data_to_store["node_features"],
                total_energies=data_to_store["total_energies"],
                minimal_basis_irrep=minimal_basis_irrep,
                species=data_to_store["species"],
                p=p,
                basis_mask=data_to_store["basis_mask"]["initial_state"],
            )

            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])
