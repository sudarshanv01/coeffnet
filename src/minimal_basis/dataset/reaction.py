from pathlib import Path

from typing import Union

import logging

import copy

from collections import defaultdict

import numpy as np

import pandas as pd

from ase import data as ase_data

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

from minimal_basis.data.reaction import ReactionDataPoint as DataPoint
from minimal_basis.data.reaction import ModifiedCoefficientMatrixSphericalBasis
from minimal_basis.predata.interpolator import GenerateParametersInterpolator

logger = logging.getLogger(__name__)


class ReactionDataset(InMemoryDataset):
    def __init__(
        self,
        filename: Union[str, Path] = None,
        root: str = None,
        transform: str = None,
        spin_channel: str = "alpha",
        pre_transform: bool = None,
        pre_filter: bool = None,
        idx_eigenvalue: int = 0,
        reactant_tag: str = "reactant",
        product_tag: str = "product",
        transition_state_tag: str = "transition_state",
        mu: float = 0.5,
        sigma: float = 0.25,
        alpha: float = 1.0,
    ):
        """Dataset for the reaction.

        Args:
            filename (Union[str, Path], optional): Path to the json file with the data. Defaults to None.
            root (str, optional): Root directory. Defaults to None.
            transform (str, optional): Transform to apply. Defaults to None.
            spin_channel (str, optional): Spin channel to use; either alpha | beta. Defaults to "alpha".
            pre_transform (bool, optional): Pre-transform to apply. Defaults to None.
            pre_filter (bool, optional): Pre-filter to apply. Defaults to None.
            idx_eigenvalue (int, optional): Index of the eigenvalue to be used for the reaction. If set at 0
                then the smallest occupied eigenvalue is used (i.e. smallest negative number). Any positive or
                negative number will be referenced to this zero value.
            reactant_tag (str, optional): Tag to be used for the reactant. Defaults to "reactant".
            product_tag (str, optional): Tag to be used for the product. Defaults to "product".
            transition_state_tag (str, optional): Tag to be used for the transition state. Defaults to "transition_state".
            mu (float, optional): Mu parameter for the interpolation. Defaults to 0.5.
            sigma (float, optional): Sigma parameter for the interpolation. Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the interpolation. Defaults to 1.0.
        """

        self.filename = filename
        self.root = root
        self.spin_channel = spin_channel
        self.idx_eigenvalue = idx_eigenvalue
        self.reactant_tag = reactant_tag
        self.product_tag = product_tag
        self.transition_state_tag = transition_state_tag
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

        if self.spin_channel not in ["alpha", "beta"]:
            raise ValueError("Spin channel must be either alpha or beta.")
        elif self.spin_channel == "alpha":
            self.spin_index = 0
        elif self.spin_channel == "beta":
            self.spin_index = 1

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

    def determine_basis_functions(self):
        """Based on `input_data` determine how many s, p, d, f and g functions
        need to be used."""
        max_s_functions = 0
        max_p_functions = 0
        max_d_functions = 0
        max_f_functions = 0
        max_g_functions = 0
        for reaction_idx, input_data in enumerate(self.input_data):
            orbital_info = input_data["orbital_info"][0]
            orbital_info = pd.DataFrame(orbital_info)
            orbital_info.columns = ["species", "idx", "l", "m"]
            orbital_info = orbital_info.groupby("idx")
            for idx, group in orbital_info:
                max_s_functions = max(
                    max_s_functions, group[group["l"] == "s"].shape[0]
                )
                max_p_functions = max(
                    max_p_functions, group[group["l"] == "p"].shape[0]
                )
                max_d_functions = max(
                    max_d_functions, group[group["l"] == "d"].shape[0]
                )
                max_f_functions = max(
                    max_f_functions, group[group["l"] == "f"].shape[0]
                )
                max_g_functions = max(
                    max_g_functions, group[group["l"] == "g"].shape[0]
                )

        self.max_s_functions = max_s_functions / 1
        self.max_p_functions = max_p_functions / 3
        self.max_d_functions = max_d_functions / 5
        self.max_f_functions = max_f_functions / 7
        self.max_g_functions = max_g_functions / 9
        self.max_s_functions = int(np.ceil(self.max_s_functions))
        self.max_p_functions = int(np.ceil(self.max_p_functions))
        self.max_d_functions = int(np.ceil(self.max_d_functions))
        self.max_f_functions = int(np.ceil(self.max_f_functions))
        self.max_g_functions = int(np.ceil(self.max_g_functions))

        self.irreps_in = f"{self.max_s_functions}x0e"
        if self.max_p_functions > 0:
            self.irreps_in += f" +{self.max_p_functions}x1o"
        if self.max_d_functions > 0:
            self.irreps_in += f" +{self.max_d_functions}x2e"
        if self.max_f_functions > 0:
            self.irreps_in += f" +{self.max_f_functions}x3o"
        if self.max_g_functions > 0:
            self.irreps_in += f" +{self.max_g_functions}x4e"
        self.irreps_out = self.irreps_in

    def download(self):
        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")
        self.determine_basis_functions()
        logger.info(
            f"Determined basis functions,\
                     s: {self.max_s_functions},\
                     p: {self.max_p_functions},\
                     d: {self.max_d_functions},\
                     f: {self.max_f_functions},\
                     g: {self.max_g_functions}"
        )

    def process(self):

        datapoint_list = []

        for reaction_idx, input_data_ in enumerate(self.input_data):
            logger.debug(f"Processing reaction {reaction_idx}")

            data_to_store = defaultdict(dict)

            input_data = copy.deepcopy(input_data_)

            eigenvalues = input_data["eigenvalues"]
            logger.debug(f"Shape of eigenvalues: {eigenvalues.shape}")

            final_energy = input_data["final_energy"]
            logger.debug(f"Shape of final energy: {final_energy.shape}")

            coeff_matrices = input_data["coeff_matrices"]
            logger.debug(f"Shape of coefficient matrices: {coeff_matrices.shape}")

            orthogonalization_matrices = input_data["orthogonalization_matrices"]
            logger.debug(
                f"Shape of orthogonalization matrices: {orthogonalization_matrices.shape}"
            )

            states = input_data["state"]
            logger.debug(f"States considered: {states}")

            structures = input_data["structures"]

            identifier = input_data["identifiers"][0]
            orbital_info = input_data["orbital_info"][0]

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

                coeff_matrix_spin = coeff_matrices[idx_state, self.spin_index]
                eigenvalues_spin = eigenvalues[idx_state, self.spin_index]
                selected_eigenval = eigenvalues_spin[eigenvalues_spin < 0]
                selected_eigenval = np.sort(selected_eigenval)
                selected_eigenval = selected_eigenval[-1]
                selected_eigenval_index = np.where(
                    eigenvalues_spin == selected_eigenval
                )[0][0]
                selected_eigenval_index = selected_eigenval_index + self.idx_eigenvalue
                logger.debug(
                    f"Selected eigenvalue {selected_eigenval} with index {selected_eigenval_index}"
                )
                indices_to_keep = input_data["indices_to_keep"][idx_state]
                data_to_store["indices_to_keep"][idx_state] = indices_to_keep
                orthogonalization_matrices_spin = orthogonalization_matrices[
                    idx_state, self.spin_index
                ]

                coeff_matrix = ModifiedCoefficientMatrixSphericalBasis(
                    molecule_graph=molecule_graph,
                    orbital_info=orbital_info,
                    coefficient_matrix=coeff_matrix_spin,
                    store_idx_only=selected_eigenval_index,
                    max_s_functions=self.max_s_functions,
                    max_p_functions=self.max_p_functions,
                    max_d_functions=self.max_d_functions,
                    max_f_functions=self.max_f_functions,
                    max_g_functions=self.max_g_functions,
                    indices_to_keep=indices_to_keep,
                )

                node_features = coeff_matrix.get_node_features()
                node_features = node_features.reshape(node_features.shape[0], -1)
                data_to_store["node_features"][state] = node_features
                basis_mask = coeff_matrix.basis_mask
                data_to_store["basis_mask"][idx_state] = basis_mask

                data_to_store["orthogonalization_matrix"][
                    state
                ] = orthogonalization_matrices_spin.flatten()

            reactant_idx = np.where(states == self.reactant_tag)[0][0]
            product_idx = np.where(states == self.product_tag)[0][0]

            reactant_structure = structures[reactant_idx]
            product_structure = structures[product_idx]

            instance_generate = GenerateParametersInterpolator()
            interpolated_transition_state_pos = (
                instance_generate.get_interpolated_transition_state_positions(
                    reactant_structure.cart_coords,
                    product_structure.cart_coords,
                    mu=self.mu,
                    sigma=self.sigma,
                    alpha=self.alpha,
                    deltaG=final_energy[product_idx] - final_energy[reactant_idx],
                )
            )
            interpolated_transition_state_structure = Molecule(
                reactant_structure.species,
                interpolated_transition_state_pos,
                charge=reactant_structure.charge,
                spin_multiplicity=reactant_structure.spin_multiplicity,
            )

            p, _ = instance_generate.get_p_and_pprime(
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
                species=data_to_store["species"],
                p=p,
                basis_mask=data_to_store["basis_mask"][reactant_idx],
                reactant_tag=self.reactant_tag,
                product_tag=self.product_tag,
                transition_state_tag=self.transition_state_tag,
                orthogonalization_matrix=data_to_store["orthogonalization_matrix"],
                indices_to_keep=data_to_store["indices_to_keep"][reactant_idx],
                identifier=identifier,
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_out,
            )

            datapoint_list.append(datapoint)

        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])
