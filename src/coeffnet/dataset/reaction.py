import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from ase import data as ase_data
from monty.serialization import loadfn
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from torch_geometric.data import InMemoryDataset

from coeffnet.constants import (
    IRREPS_ORBITALS,
    NUM_ORBITALS,
    ORBITALS,
    electronic_states,
)
from coeffnet.data.reaction import ModifiedCoefficientMatrixSphericalBasis
from coeffnet.data.reaction import ReactionDataPoint as DataPoint
from coeffnet.predata.interpolator import GenerateParametersInterpolator

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
        invert_coordinates: bool = False,
    ):
        """Dataset for the reaction needed for ML for reaction properties.

        Creates the dataset needed to apply different neural-network models.
        This class wraps around the appropriate Data class and returns a list
        of dataset structures.

        Args:
            filename (Union[str, Path], optional): Path to the json file with
                the data. Defaults to None.
            root (str, optional): Root directory. Defaults to None.
            transform (str, optional): Transform to apply. Defaults to None.
            spin_channel (str, optional): Spin channel to use; either alpha or
                beta. Defaults to "alpha".
            pre_transform (bool, optional): Pre-transform to apply. Defaults to
                None.
            pre_filter (bool, optional): Pre-filter to apply. Defaults to None.
            idx_eigenvalue (int, optional): Index of the eigenvalue to be used
                for the reaction. If set at 0 then the smallest occupied
                eigenvalue is used (i.e. smallest negative number). Any
                positive or negative number will be referenced to this zero
                value.
            reactant_tag (str, optional): Tag to be used for the reactant.
                Defaults to "reactant".
            product_tag (str, optional): Tag to be used for the product.
                Defaults to "product".
            transition_state_tag (str, optional): Tag to be used for the
                transition state. Defaults to "transition_state".
            mu (float, optional): Mu parameter for the interpolation.
                Defaults to 0.5.
            sigma (float, optional): Sigma parameter for the interpolation.
                Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the interpolation.
                Defaults to 1.0.
            invert_coordinates (bool, optional): Invert coordinates
                (x,y,z) -> (z,x,y). Defaults to False.
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
        self.invert_coordinates = invert_coordinates

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

    def _determine_max_num_orbital(self, orbital_type, prev_max, group):
        return max(prev_max, group[group["l"] == orbital_type].shape[0])

    def _determine_num_functions_orbitals(self, num_of_functions, number_of_orbitals):
        return int(np.ceil(num_of_functions / number_of_orbitals))

    def determine_basis_functions_and_irreps(self):
        """Based on `input_data` determine how many s, p, d, f and g
        functions need to be used."""
        max_functions = {orb: 0 for orb in ORBITALS}
        for reaction_idx, input_data in enumerate(self.input_data):
            orbital_info = input_data["orbital_info"][0]
            orbital_info = pd.DataFrame(orbital_info)
            orbital_info.columns = ["species", "idx", "l", "m"]
            orbital_info = orbital_info.groupby("idx")
            for idx, group in orbital_info:
                for orbital in ORBITALS:
                    max_functions[orbital] = self._determine_max_num_orbital(
                        orbital_type=orbital,
                        prev_max=max_functions[orbital],
                        group=group,
                    )
        self.irreps_in = ""
        for orbital in ORBITALS:
            num_orbitals = self._determine_num_functions_orbitals(
                num_of_functions=max_functions[orbital],
                number_of_orbitals=NUM_ORBITALS[orbital],
            )
            setattr(self, f"max_{orbital}_functions", num_orbitals)
            max_function = getattr(self, f"max_{orbital}_functions")
            if max_function > 0:
                self.irreps_in += f" +{max_function}x{IRREPS_ORBITALS[orbital]}"
        if not self.irreps_in:
            raise ValueError("There are no orbitals for this datapoint.")
        self.irreps_in = self.irreps_in[2:]
        self.irreps_out = self.irreps_in

    def determine_classes_for_one_hot_encoding(self):
        """Determine the number of classes for one-hot encoding the
        atomic number.

        This method provides all the unique atomic numbers that are
        present in the dataset.
        """

        species = []

        for reaction_idx, input_data in enumerate(self.input_data):
            orbital_info = input_data["orbital_info"][0]
            orbital_info = pd.DataFrame(orbital_info)
            orbital_info.columns = ["species", "idx", "l", "m"]
            _species = orbital_info["species"].unique()
            species.extend(_species)

        self.unique_species_in_dataset = np.unique(species)
        self.unique_atomic_numbers = [
            ase_data.atomic_numbers[species]
            for species in self.unique_species_in_dataset
        ]
        self.unique_atomic_numbers = np.sort(np.unique(self.unique_atomic_numbers))

        occupancy_dict = {}
        for _atomic_number in self.unique_atomic_numbers:
            electronic_config = electronic_states[
                ase_data.chemical_symbols[_atomic_number]
            ]
            electronic_config = electronic_config.split(" ")
            electronic_config = [
                [int(e[0]), str(e[1]), int(e[2])] for e in electronic_config
            ]
            electronic_config = np.array(electronic_config)
            df = pd.DataFrame(electronic_config, columns=["n", "l", "occ"])
            occ_s = df[df["l"] == "s"]["occ"].values[-1]
            if "p" not in df["l"].values:
                occ_p = 0
            else:
                occ_p = df[df["l"] == "p"]["occ"].values[-1]
            occupancy_dict[_atomic_number] = [int(occ_s), int(occ_p)]
        self.occupancy_dict = occupancy_dict

        self.irreps_node_attr = f"{len(self.unique_atomic_numbers)}x0e"

    def download(self):
        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")
        self.determine_basis_functions_and_irreps()
        logger.info(
            f"Determined basis functions,\
                     s: {self.max_s_functions},\
                     p: {self.max_p_functions},\
                     d: {self.max_d_functions},\
                     f: {self.max_f_functions},\
                     g: {self.max_g_functions}"
        )
        self.determine_classes_for_one_hot_encoding()
        logger.info(f"Unique atomic numbers in dataset: {self.unique_atomic_numbers}")

    def process(self):
        """Processes the data in an appropriate format for machine learning.

        The dictionary that passed into the Data class is in a nested dict,
        which has the three states as the first keys.
        """
        datapoint_list = []
        for reaction_idx, input_data in enumerate(self.input_data):
            data_to_store = defaultdict(dict)
            input = InputDataDict(**input_data)
            #
            reactant_idx = np.where(input.state == self.reactant_tag)[0][0]
            product_idx = np.where(input.state == self.product_tag)[0][0]
            reactant_structure = input.structures[reactant_idx]
            product_structure = input.structures[product_idx]
            deltaG = input.final_energy[product_idx] - input.final_energy[reactant_idx]
            #
            for idx_state, state in enumerate(input.state):
                molecule = input.structures[idx_state]
                molecule_graph = create_molecule_graph(molecule)
                coeff_matrix = input.coeff_matrices[idx_state, self.spin_index]
                ortho_matrix = input.orthogonalization_matrices[
                    idx_state, self.spin_index
                ]
                eigenvalues = input.eigenvalues[idx_state, self.spin_index]
                eigenvalue_idx = generate_eigenval_idx(eigenvalues)
                modified_coeff_matrix = ModifiedCoefficientMatrixSphericalBasis(
                    molecule_graph=molecule_graph,
                    orbital_info=input.orbital_info,
                    coefficient_matrix=coeff_matrix,
                    store_idx_only=eigenvalue_idx,
                    max_s_functions=self.max_s_functions,
                    max_p_functions=self.max_p_functions,
                    max_d_functions=self.max_d_functions,
                    max_f_functions=self.max_f_functions,
                    max_g_functions=self.max_g_functions,
                    indices_to_keep=input.indices_to_keep,
                )
                node_features = modified_coeff_matrix.get_node_features()
                node_features = node_features.reshape(node_features.shape[0], -1)
                basis_mask = modified_coeff_matrix.basis_mask
                data_to_store["total_energies"][state] = input.final_energy[idx_state]
                data_to_store["pos"][state] = generate_positions(
                    molecule_graph, self.invert_coordinates
                )
                data_to_store["species"][state] = generate_species(molecule_graph)
                data_to_store["edge_index"][state] = generate_edge_index(molecule_graph)
                data_to_store["indices_to_keep"][idx_state] = input.indices_to_keep
                data_to_store["node_features"][state] = node_features
                data_to_store["basis_mask"][idx_state] = basis_mask
                data_to_store["ortho_matrix"][state] = ortho_matrix.flatten()
            data_to_store["p"], data_to_store["pos"]["interp_ts"] = generate_interp_ts(
                reactant_structure,
                product_structure,
                deltaG,
                self.mu,
                self.sigma,
                self.alpha,
            )
            edge_index_interp_ts = get_interp_index(
                reactant_index=data_to_store["edge_index"][input.state[reactant_idx]],
                product_index=data_to_store["edge_index"][input.state[product_idx]],
            )
            data_to_store["edge_index"]["interp_ts"] = edge_index_interp_ts
            datapoint = DataPoint(
                pos=data_to_store["pos"],
                edge_index=data_to_store["edge_index"],
                node_inputs=data_to_store["node_features"],
                total_energies=data_to_store["total_energies"],
                species=data_to_store["species"],
                unique_atomic_numbers=self.unique_atomic_numbers,
                p=data_to_store["p"],
                basis_mask=data_to_store["basis_mask"][reactant_idx],
                reactant_tag=self.reactant_tag,
                product_tag=self.product_tag,
                transition_state_tag=self.transition_state_tag,
                orthogonalization_matrix=data_to_store["ortho_matrix"],
                indices_to_keep=data_to_store["indices_to_keep"][reactant_idx],
                identifier=input.identifiers,
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_out,
                irreps_node_attr=self.irreps_node_attr,
                occupancy_dict=self.occupancy_dict,
            )
            datapoint_list.append(datapoint)
        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class InputDataDict:
    state: Sequence
    eigenvalues: Sequence
    final_energy: Sequence
    coeff_matrices: Sequence
    orthogonalization_matrices: Sequence
    identifiers: Sequence
    orbital_info: Sequence
    structures: Sequence
    indices_to_keep: Sequence
    irreps: Sequence

    def is_iterable(self, seq):
        if isinstance(seq, Sequence) or isinstance(seq, np.ndarray):
            return True
        else:
            return False

    def __post_init__(self):
        if self.is_iterable(self.identifiers):
            self.identifiers = self.identifiers[0]
        if self.is_iterable(self.orbital_info):
            self.orbital_info = self.orbital_info[0]
        if self.is_iterable(self.structures):
            for idx, structure in enumerate(self.structures):
                self.structures[idx] = Molecule.from_dict(structure)


def create_molecule_graph(molecule: Molecule):
    molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
    return molecule_graph


def generate_species(molecule_graph):
    species_in_molecule = molecule_graph.molecule.species
    species_in_molecule = [
        ase_data.chemical_symbols.index(species.symbol)
        for species in species_in_molecule
    ]
    species_in_molecule = np.array(species_in_molecule).reshape(-1, 1)
    return species_in_molecule


def generate_positions(molecule_graph, invert_coordinates):
    positions = molecule_graph.molecule.cart_coords
    if invert_coordinates:
        positions = np.array(positions)
        positions[:, [0, 1, 2]] = positions[:, [2, 0, 1]]
    return positions


def generate_edge_index(molecule_graph):
    edges_for_graph = molecule_graph.graph.edges
    edges_for_graph = [list(edge[:-1]) for edge in edges_for_graph]
    edge_index = np.array(edges_for_graph).T
    return edge_index


def generate_eigenval_idx(eigenvalues, idx_eigenvalue=0):
    selected_eigenval = eigenvalues[eigenvalues < 0]
    selected_eigenval = np.sort(selected_eigenval)
    selected_eigenval = selected_eigenval[-1]
    selected_eigenval_index = np.where(eigenvalues == selected_eigenval)[0][0]
    selected_eigenval_index = selected_eigenval_index + idx_eigenvalue
    return selected_eigenval_index


def generate_interp_ts(reactant_structure, product_structure, deltaG, mu, sigma, alpha):
    instance_generate = GenerateParametersInterpolator()
    data_to_store = defaultdict(dict)
    interp_ts_pos = instance_generate.get_interpolated_transition_state_positions(
        reactant_structure.cart_coords,
        product_structure.cart_coords,
        mu=mu,
        sigma=sigma,
        alpha=alpha,
        deltaG=deltaG,
    )
    p, _ = instance_generate.get_p_and_pprime(mu, sigma, alpha * deltaG)
    return p, interp_ts_pos


def get_interp_index(reactant_index, product_index):
    combined_index = np.concatenate([reactant_index, product_index], axis=1)
    return np.unique(combined_index, axis=1)
