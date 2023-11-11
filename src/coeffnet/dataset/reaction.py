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
        idx_eigenvalue: Union[int, Sequence] = 0,
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
        if isinstance(idx_eigenvalue, int):
            self.idx_eigenvalue = np.array([idx_eigenvalue])
        elif isinstance(idx_eigenvalue, list):
            self.idx_eigenvalue = np.array(idx_eigenvalue)

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
        """Determine maximum number of basis functions and irreps.

        Based on the input_data dictionary determine how many s, p, d, f and g
        functions needed to fully describe a given dataset. The maximum number of
        orbitals required is given by the maximum number of orbitals for _any_ atom in
        the dataset. Adding heavier elements causes the number of basis functions to
        increase leading to a larger number of orbital functions to describe all atoms.
        """
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
        irreps_in = ""
        for orbital in ORBITALS:
            num_orbitals = self._determine_num_functions_orbitals(
                num_of_functions=max_functions[orbital],
                number_of_orbitals=NUM_ORBITALS[orbital],
            )
            setattr(self, f"max_{orbital}_functions", num_orbitals)
            max_function = getattr(self, f"max_{orbital}_functions")
            if max_function > 0:
                irreps_in += f" +{max_function}x{IRREPS_ORBITALS[orbital]}"
        if not irreps_in:
            raise ValueError("There are no orbitals for this datapoint.")
        irreps_in = " ".join([irreps_in] * self.idx_eigenvalue.shape[0])
        irreps_in = irreps_in[2:]
        self.irreps_in = irreps_in.replace(" ", "")
        self.irreps_out = self.irreps_in

    def determine_classes_for_one_hot_encoding(self):
        """Determine the number of classes for one-hot encoding the atomic number.

        One hot encoding based on the number of species in the dataset. This method
        provides all the unique atomic numbers that are present in the dataset. The
        more species you have in your dataset, the more the number of separate one-hot-
        encoding components that are required.
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
        """Download the dataset.

        The dataset is "downloaded" from json files stored in the directory under the
        filename inputs to this class.
        """
        self.input_data = loadfn(self.filename)
        logger.info("Successfully loaded json file with data.")
        self.determine_basis_functions_and_irreps()
        logger.info(
            f"Determined basis functions per eigenvalue,\
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
                eigenvalue_idx = generate_eigenval_idx(eigenvalues, self.idx_eigenvalue)
                node_features = []
                basis_mask = []
                for _eigenvalue_idx in eigenvalue_idx:
                    modified_coeff_matrix = ModifiedCoefficientMatrixSphericalBasis(
                        molecule_graph=molecule_graph,
                        orbital_info=input.orbital_info,
                        coefficient_matrix=coeff_matrix,
                        store_idx_only=_eigenvalue_idx,
                        max_s_functions=self.max_s_functions,
                        max_p_functions=self.max_p_functions,
                        max_d_functions=self.max_d_functions,
                        max_f_functions=self.max_f_functions,
                        max_g_functions=self.max_g_functions,
                        indices_to_keep=input.indices_to_keep,
                    )
                    _node_features = modified_coeff_matrix.get_node_features()
                    _node_features = _node_features.reshape(_node_features.shape[0], -1)
                    node_features.append(_node_features)
                    basis_mask.append(modified_coeff_matrix.basis_mask)
                node_features = np.concatenate(node_features, axis=1)
                basis_mask = np.concatenate(basis_mask, axis=1)
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
                number_eigenvalues=len(self.idx_eigenvalue),
            )
            datapoint_list.append(datapoint)
        data, slices = self.collate(datapoint_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class InputDataDict:
    state: Sequence
    """The states of the reaction; typically reactant, product and transition_state"""
    eigenvalues: Sequence
    """Eigenvalues of the Hamiltonian matrix"""
    final_energy: Sequence
    """Energies of the reactant product and transition state, in the order of state"""
    coeff_matrices: Sequence
    """Coefficient matrices in an array with dimensions (state, spin, ...)"""
    orthogonalization_matrices: Sequence
    """Orthogonalization matrices in an array wit dimensions (state, spin, ...)"""
    identifiers: Sequence or str
    """String identifier for this datapoint. If sequence if provided it is converted to
    a single string (i.e. only one identifier for a datapoint possible)"""
    orbital_info: Sequence
    """Information about the type and number of orbitals."""
    structures: Sequence
    """Structures for the reactant, product and transition state in the order of state"""
    indices_to_keep: Sequence
    """Indices to keep in the analysis. Useful for contracting the number of basis
    functions required for the learning process."""
    irreps: Sequence
    """Irreducible representations of the coefficient matrices"""

    def is_iterable(self, seq):
        """Checks if sequence is iterable

        Return true if the inputs are a sequence or a numpy array.
        """
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
    """Converts a molecule to a molecule graph

    The molecule graph will always be converted to a graph using OpenBabel
    """
    molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
    return molecule_graph


def generate_species(molecule_graph):
    """Generate the species by the molecule graph

    Create the species based on the molecule graph and the species it contains. ase data
    is used to convert the symbol to an atomic number.

    Args:
        molecule_graph (MoleculeGraph): molecule graph of a species.
    Returns:
        species_in_molecule (np.ndarray): Atom numbers of all species.
    """
    species_in_molecule = molecule_graph.molecule.species
    species_in_molecule = [
        ase_data.chemical_symbols.index(species.symbol)
        for species in species_in_molecule
    ]
    species_in_molecule = np.array(species_in_molecule).reshape(-1, 1)
    return species_in_molecule


def generate_positions(molecule_graph, invert_coordinates):
    """Generates the positions from the molecule graph.

    The cartesian coordinates of the molecule graph are generated as positions. If the
    species were generated from an ab-initio calculation, the the coordinate system
    must be inverted to make sure it is consistent with the spherical harmonics used in
    the equivariant network. This function takes care of this if invert_coordinates is
    set to True.

    Args:
        molecule_graph (MoleculeGraph): molecule graph of a species
        invert_coordinates (bool): Invert the coordinates from xyz -> zxy
    Returns:
        positions (np.ndarray): positions of the atoms
    """
    positions = molecule_graph.molecule.cart_coords
    if invert_coordinates:
        positions = np.array(positions)
        positions[:, [0, 1, 2]] = positions[:, [2, 0, 1]]
    return positions


def generate_edge_index(molecule_graph):
    """Generate edge index for a molecule graph

    Returns the edge indices of the graph used for learning and inference. Makes sure
    that the weights are not returned (weights that may be stored during the relaxation
    process, for example).

    Args:
        molecule_graph (MoleculeGraph): molecule graph of a species
    Returns:
        edge_index (np.ndarray): indices of the edges of the graph
    """
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
    """Generates the interpolated transition state structure.

    Based on the input parameters (mu, sigma, alpha, Delta) and the reactant and product
    structures, generate the interpolated transition state structures.

    Args:
        reactant_structure (Molecule): structure of the reactant
        product_structure (Molecule): structure of the product
        deltaG (float): reaction energy between the product and reactant
        mu (float): mu parameter of the truncated gaussian
        sigma (float): sigma parameter of the truncated gaussian
        alpha (float): alpha parameter of the truncated gaussian
    Returns:
        p (float): p parameter output of the interpolation
        interp_ts_position (np.ndarray): interpolated transition state positions
    """
    instance_generate = GenerateParametersInterpolator()
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
    """Generate the edge index of the interpolated transition state graph.

    The interpolated transition state graph with have the indices as the union of the
    reactant and product graphs.

    Args:
        reactant_index (np.ndarray): index of the edges of the reactant graph
        product_index (np.ndarray): index of the edges of the product graph
    Returns:
        (np.ndarray): union of the reactant and product indices
    """
    combined_index = np.concatenate([reactant_index, product_index], axis=1)
    return np.unique(combined_index, axis=1)
