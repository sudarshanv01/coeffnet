import logging

from monty.serialization import loadfn

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import torch

from torch_geometric.data import InMemoryDataset

from minimal_basis.data import ActivationBarrierDatapoint as Datapoint
from minimal_basis.predata import GenerateParametersClassifier

logger = logging.getLogger(__name__)

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)


class ActivationBarrierDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        filename: str = None,
        filename_classifier_parameters: str = None,
    ):
        """Dataset which stores the data for predicting the raw activation barrier."""

        self.filename = filename
        self.filename_classifier_parameters = filename_classifier_parameters
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
        return ["data_for_ml.json", "classifier_parameters.json"]

    @property
    def processed_file_names(self):
        return "activation_barrier_data.pt"

    def download(self):
        logger.info("Loading data from json file.")
        self.input_data = loadfn(self.filename)
        logger.info("Done loading data from json file.")
        self.classifier_parameters = loadfn(self.filename_classifier_parameters)

    def process(self):
        data_list = []

        for data_ in self.input_data:

            # --- Transition state graph construction ---
            reactant_structure = data_["reactant_structure"]
            product_structure = data_["product_structure"]

            # Generate the transition state structure from the reactant and product structures
            parameters_transition_state = GenerateParametersClassifier(
                deltaG=torch.tensor([data_["reaction_energy"]], dtype=DTYPE),
            )
            transition_state_coords = (
                parameters_transition_state.get_interpolated_transition_state_positions(
                    is_positions=reactant_structure.cart_coords,
                    fs_positions=product_structure.cart_coords,
                    alpha=self.classifier_parameters["alpha"],
                    mu=self.classifier_parameters["mu"],
                    sigma=self.classifier_parameters["sigma"],
                )
            )
            logger.debug(
                f"Shape of transition state coords: {transition_state_coords.shape}"
            )

            # Generate a molecule object for the transition state coords
            transition_state_structure = Molecule(
                species=reactant_structure.species,
                coords=transition_state_coords,
            )
            # Make a graph out of the transition state structure
            transition_state_graph = MoleculeGraph.with_local_env_strategy(
                transition_state_structure, OpenBabelNN()
            )

            # Get the proportion of the reactant and product structures
            p, p_prime = parameters_transition_state.get_p_and_pprime(
                alpha=self.classifier_parameters["alpha"],
                mu=self.classifier_parameters["mu"],
                sigma=self.classifier_parameters["sigma"],
            )

            # --- Global features ---
            reaction_energy = data_["reaction_energy"]
            charge = data_["charge"]
            spin_multiplicity = data_["spin_multiplicity"]
            global_features = [
                reaction_energy,
                charge,
                spin_multiplicity,
            ]
            num_global_features = len(global_features)
            global_features = torch.tensor(global_features, dtype=torch.float)

            # --- Node features ---
            # Get the reactant and product quantties
            reactant_partial_charges = torch.tensor(
                data_["reactant_partial_charges"], dtype=DTYPE
            )
            product_partial_charges = torch.tensor(
                data_["product_partial_charges"], dtype=DTYPE
            )
            reactant_core_elctrons = torch.tensor(
                data_["reactant_core_electrons"], dtype=DTYPE
            )
            product_core_electrons = torch.tensor(
                data_["product_core_electrons"], dtype=DTYPE
            )
            reactant_valence_electrons = torch.tensor(
                data_["reactant_valence_electrons"], dtype=DTYPE
            )
            product_valence_electrons = torch.tensor(
                data_["product_valence_electrons"], dtype=DTYPE
            )
            reactant_rydber_electrons = torch.tensor(
                data_["reactant_rydberg_electrons"], dtype=DTYPE
            )
            product_rydberg_electrons = torch.tensor(
                data_["product_rydberg_electrons"], dtype=DTYPE
            )

            # The quantities will be partitioned the same way as the structures
            nbo_charges = (
                p_prime * reactant_partial_charges + p * product_partial_charges
            )
            core_electrons = (
                p_prime * reactant_core_elctrons + p * product_core_electrons
            )
            valence_electrons = (
                p_prime * reactant_valence_electrons + p * product_valence_electrons
            )
            rydberg_electrons = (
                p_prime * reactant_rydber_electrons + p * product_rydberg_electrons
            )

            node_features = torch.cat(
                [
                    nbo_charges.unsqueeze(1),
                    core_electrons.unsqueeze(1),
                    valence_electrons.unsqueeze(1),
                    rydberg_electrons.unsqueeze(1),
                ],
                dim=1,
            )
            logger.debug("node_features: {}".format(node_features))
            logger.debug("node_features.shape: {}".format(node_features.shape))

            # --- Edge features ---
            edges_for_graph = transition_state_graph.graph.edges
            edges_for_graph = [list(edge[:-1]) for edge in edges_for_graph]
            logger.debug("edges_for_graph: {}".format(edges_for_graph))
            edge_index = torch.tensor(edges_for_graph, dtype=torch.long)
            logger.debug("edge_index: {}".format(edge_index))
            edge_index = edge_index.t().contiguous()

            # Collect the edge features, current the bond length
            # between two atom centres will be used as the edge feature
            row, col = edge_index
            atom_positions = transition_state_structure.cart_coords
            atom_positions = torch.tensor(atom_positions, dtype=torch.float)
            edge_attributes = torch.linalg.norm(
                atom_positions[row] - atom_positions[col], dim=1
            )
            edge_attributes = edge_attributes.view(-1, 1)

            # Collect 'y' which is determines if the reaction is feasible or not
            y = data_["transition_state_energy"]
            logging.debug("y: {}".format(y))

            # Create the datapoint
            data = Datapoint(
                x=node_features,
                edge_index=edge_index,
                y=y,
                global_attr=global_features,
                edge_attr=edge_attributes,
                num_global_features=num_global_features,
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
