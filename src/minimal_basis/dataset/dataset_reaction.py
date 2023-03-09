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
    ):
        """Generic dataset of reaction data."""

        self.filename = filename
        self.root = root
        self.basis_filename = basis_filename

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

                coeff_matrix = ModifiedCoefficientMatrix(
                    molecule_graph=molecule_graph,
                    basis_info_raw=self.basis_info_raw,
                    coefficient_matrix=alpha_coeff_matrix,
                    store_idx_only=selected_eigenval_index,
                    set_to_absolute=True,
                )
                node_features = coeff_matrix.get_minimal_basis_representation()
                # TODO: Currently we are only implementing a single eigenvalue
                # for the alpha coefficient matrix. At some point multiple
                # eigenvalues need to be added and the following flattening
                # will have to be done differently.
                node_features = node_features.reshape(node_features.shape[0], -1)
                data_to_store["node_features"][state] = node_features

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
            )

            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])
