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

import itertools

from e3nn import o3

from minimal_basis.data.data_reaction import ReactionDataPoint as DataPoint

logger = logging.getLogger(__name__)


class ReactionDataset(InMemoryDataset):
    def __init__(
        self,
        filename: Union[str, Path] = None,
        basis_filename: Union[str, Path] = None,
        root: str = None,
        transform: str = None,
        pre_transform: bool = None,
        pre_filter: bool = None,
    ):
        """Dataset for the Hamiltonian for all species in a reaction."""

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.filename = filename
        self.basis_filename = basis_filename

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [self.filename + ".pt"]

    def download(self):
        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")
        with open(self.basis_file) as f:
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

            initial_states_structure = structures[states.index("initial_state")]
            final_states_structure = structures[states.index("final_state")]
            interpolated_transition_state_structure = interpolate_midpoint_zmat(
                initial_states_structure, final_states_structure
            )

            if interpolated_transition_state_structure is None:
                logger.warning(
                    "Could not interpolate the initial and final state structures."
                )
                continue

            interpolated_transition_state_pos = (
                interpolated_transition_state_structure.cart_coords
            )

            # Create a MoleculeGraph object for the interpolated transition state structure
            interpolated_transition_state_structure_graph = (
                MoleculeGraph.with_local_env_strategy(
                    interpolated_transition_state_structure, OpenBabelNN()
                )
            )
            # Get the edge_index for the interpolated transition state structure
            edges_for_graph = interpolated_transition_state_structure_graph.graph.edges
            interpolated_transition_state_structure_edge_index = [
                list(edge[:-1]) for edge in edges_for_graph
            ]
            interpolated_transition_state_structure_edge_index = np.array(
                interpolated_transition_state_structure_edge_index
            ).T

            datapoint = DataPoint(
                pos=data_to_store["pos"],
                edge_index=data_to_store["edge_index"],
                x=data_to_store["node_features"],
                y=y - y_IS,
                all_basis_idx=all_basis_idx,
                edge_index_interpolated_TS=interpolated_transition_state_structure_edge_index,
                pos_interpolated_TS=interpolated_transition_state_pos,
                irreps_node_features=node_irrep,
                global_attr=data_to_store["global_attr"],
                angles=angles,
                irreps_minimal_basis=irreps_minimal_basis,
                minimal_fock_matrix=data_to_store["minimal_fock_matrix"],
            )
