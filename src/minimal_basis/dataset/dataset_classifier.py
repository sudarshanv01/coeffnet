import logging

from monty.serialization import loadfn

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import torch

from torch_geometric.data import InMemoryDataset

from minimal_basis.data import DatapointClassifier as Datapoint

logger = logging.getLogger(__name__)


class SimpleTSDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        filename: str = None,
    ):

        self.filename = filename
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data_for_ml.json"

    @property
    def processed_file_names(self):
        return "data_for_ml.pt"

    def download(self):
        logger.info("Loading data from json file.")
        self.input_data = loadfn(self.filename)
        logger.info("Done loading data from json file.")

    def process(self):
        data_list = []

        for data_ in self.input_data:

            reactant_structure = data_["reactant_structure"]
            product_structure = data_["product_structure"]

            # Get the average of all the quantities based on the reactant and product structures
            nbo_charges = np.array(data_["reactant_partial_charges"]) + np.array(
                data_["product_partial_charges"]
            )
            nbo_charges = nbo_charges / 2
            core_electrons = np.array(data_["reactant_core_electrons"]) + np.array(
                data_["product_core_electrons"]
            )
            core_electrons = core_electrons / 2
            valence_electrons = np.array(
                data_["reactant_valence_electrons"]
            ) + np.array(data_["product_valence_electrons"])
            valence_electrons = valence_electrons / 2
            rydberg_electrons = np.array(
                data_["reactant_rydberg_electrons"]
            ) + np.array(data_["product_rydberg_electrons"])
            rydberg_electrons = rydberg_electrons / 2

            # Choose the reaction energy as the global feature
            reaction_energy = data_["reaction_energy"]
            charge = data_["charge"]
            spin_multiplicity = data_["spin_multiplicity"]
            global_features = [
                reaction_energy,
                charge,
                spin_multiplicity,
            ]
            global_features = torch.tensor(global_features, dtype=torch.float)

            # Make a graph out of the transition state structure
            transition_state_graph = MoleculeGraph.with_local_env_strategy(
                transition_state_structure, OpenBabelNN()
            )

            # Collect the node features
            node_features = []
            for idx, node in enumerate(transition_state_graph.molecule):
                node_feature = [
                    node.specie.Z,
                    nbo_charges[idx],
                    core_electrons[idx],
                    valence_electrons[idx],
                    rydberg_electrons[idx],
                ]
                node_features.append(node_feature)
            node_features = torch.tensor(node_features, dtype=torch.float)
            logger.debug("node_features: {}".format(node_features))
            logger.debug("node_features.shape: {}".format(node_features.shape))

            # Collect the edge index
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
            y = data_["feasiblity"]
            logging.debug("y: {}".format(y))

            # Create the datapoint
            data = Datapoint(
                x=node_features,
                edge_index=edge_index,
                y=y,
                u=global_features,
                edge_attr=edge_attributes,
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
