import logging

from monty.serialization import loadfn

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

import torch

from torch_geometric.data import InMemoryDataset

from minimal_basis.data import DiffClassifierDatapoint as Datapoint
from minimal_basis.predata import GenerateParametersClassifier


from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
)


class DiffClassifierDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        filename: str = None,
        pretrain_params_json: str = None,
        feasibility_threshold: float = 0.65,
        **kwargs,
    ):
        """Dataset which stores the data for predicting the raw activation barrier."""

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        if kwargs.get("debug", False):
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Debug mode is on.")

        self.filename = filename
        self.pretrain_params_json = pretrain_params_json

        self.feasibility_threshold = feasibility_threshold

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
        return "diffclassifier_data.pt"

    def download(self):
        self.logger.info("Loading data from json file.")
        self.input_data = loadfn(self.filename)
        self.logger.info("Done loading data from json file.")
        self.pretrain_params = loadfn(self.pretrain_params_json)

    def process(self):
        data_list = []

        for data_ in self.input_data:

            # --- Transition state graph construction ---
            reactant_structure = data_["reactant_structure"]
            product_structure = data_["product_structure"]
            reaction_energy = data_["reaction_energy"]

            # Generate the transition state structure from the reactant and product structures
            parameters_transition_state = GenerateParametersClassifier(
                deltaG=torch.tensor([data_["reaction_energy"]], dtype=DTYPE),
            )
            transition_state_coords = (
                parameters_transition_state.get_interpolated_transition_state_positions(
                    is_positions=reactant_structure.cart_coords,
                    fs_positions=product_structure.cart_coords,
                    alpha=self.pretrain_params["alpha"],
                    mu=self.pretrain_params["mu"],
                    sigma=self.pretrain_params["sigma"],
                    deltaG=torch.tensor([data_["reaction_energy"]], dtype=DTYPE),
                )
            )
            self.logger.debug(
                f"Shape of transition state coords: {transition_state_coords.shape}"
            )

            # Generate a molecule object for the transition state coords
            transition_state_structure = Molecule(
                species=reactant_structure.species,
                coords=transition_state_coords,
            )
            # Store the species of the transition state structure
            transition_state_species = transition_state_structure.species
            # Make a graph out of the transition state structure
            transition_state_graph = MoleculeGraph.with_local_env_strategy(
                transition_state_structure, OpenBabelNN()
            )
            # Add the metal edge extender to the graph
            transition_state_graph = metal_edge_extender(transition_state_graph)

            # Store the reactant graph as well
            reactant_graph = data_["reactant_molecule_graph"]

            # Get the proportion of the reactant and product structures
            p, p_prime = parameters_transition_state.get_p_and_pprime(
                alpha=self.pretrain_params["alpha"],
                mu=self.pretrain_params["mu"],
                sigma=self.pretrain_params["sigma"],
            )

            # --- Global features ---
            charge = data_["charge"]
            spin_multiplicity = data_["spin_multiplicity"]
            reactant_energy = data_["reactant_energy"]
            product_energy = data_["product_energy"]
            mean_reactant_product_energy = (reactant_energy + product_energy) / 2
            global_features_transition_state = [
                charge,
                spin_multiplicity,
                mean_reactant_product_energy,
            ]
            num_global_features = len(global_features_transition_state)
            global_features_reactant = [
                charge,
                spin_multiplicity,
                reactant_energy,
            ]
            # Convert to tensors
            global_features_transition_state = torch.tensor(
                global_features_transition_state, dtype=DTYPE
            )
            global_features_reactant = torch.tensor(
                global_features_reactant, dtype=DTYPE
            )
            global_features = {
                "transition_state": global_features_transition_state,
                "reactant": global_features_reactant,
            }

            # --- Node features ---
            # Get the atomic numbers
            atomic_numbers = torch.tensor(
                [species.Z for species in transition_state_species], dtype=DTYPE_INT
            )
            # Get the reactant and product quantites
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
            reactant_rydberg_electrons = torch.tensor(
                data_["reactant_rydberg_electrons"], dtype=DTYPE
            )
            product_rydberg_electrons = torch.tensor(
                data_["product_rydberg_electrons"], dtype=DTYPE
            )

            # The quantities will be partitioned the same way as the structures
            nbo_charges_transition_state = (
                p_prime * reactant_partial_charges + p * product_partial_charges
            )
            core_electrons_transition_state = (
                p_prime * reactant_core_elctrons + p * product_core_electrons
            )
            valence_electrons_transition_state = (
                p_prime * reactant_valence_electrons + p * product_valence_electrons
            )
            rydberg_electrons_transition_state = (
                p_prime * reactant_rydberg_electrons + p * product_rydberg_electrons
            )

            node_features_transition_state = torch.cat(
                [
                    atomic_numbers.unsqueeze(1),
                    nbo_charges_transition_state.unsqueeze(1),
                    core_electrons_transition_state.unsqueeze(1),
                    valence_electrons_transition_state.unsqueeze(1),
                    rydberg_electrons_transition_state.unsqueeze(1),
                ],
                dim=1,
            )
            node_features_reactant = torch.cat(
                [
                    atomic_numbers.unsqueeze(1),
                    reactant_partial_charges.unsqueeze(1),
                    reactant_core_elctrons.unsqueeze(1),
                    reactant_valence_electrons.unsqueeze(1),
                    reactant_rydberg_electrons.unsqueeze(1),
                ],
                dim=1,
            )

            self.logger.debug(
                "node_features: {}".format(node_features_transition_state)
            )
            self.logger.debug(
                "node_features.shape: {}".format(node_features_transition_state.shape)
            )

            node_features = {
                "reactant": node_features_reactant,
                "transition_state": node_features_transition_state,
            }

            # --- Edge features ---
            edges_for_graph_transition_state = transition_state_graph.graph.edges
            edges_for_graph_transition_state = [
                list(edge[:-1]) for edge in edges_for_graph_transition_state
            ]
            self.logger.debug(
                "edges_for_graph (TS): {}".format(edges_for_graph_transition_state)
            )
            edge_index_transition_state = torch.tensor(
                edges_for_graph_transition_state, dtype=torch.long
            )
            self.logger.debug("edge_index (TS): {}".format(edge_index_transition_state))
            edge_index_transition_state = edge_index_transition_state.t().contiguous()

            edges_for_graph_reactant = reactant_graph.graph.edges
            edges_for_graph_reactant = [
                list(edge[:-1]) for edge in edges_for_graph_reactant
            ]
            self.logger.debug(
                "edges_for_graph (Reactant): {}".format(edges_for_graph_reactant)
            )
            edge_index_reactant = torch.tensor(
                edges_for_graph_reactant, dtype=torch.long
            )
            self.logger.debug("edge_index (Reactant): {}".format(edge_index_reactant))
            edge_index_reactant = edge_index_reactant.t().contiguous()

            # Remove structures where either the transition state and the reactants
            # have no edges
            if len(edge_index_transition_state) == 0 or len(edge_index_reactant) == 0:
                self.logger.warning(
                    "No edges found for either the transition state or the reactant"
                )
                continue

            # Collect the edge features, current the bond length
            # between two atom centres will be used as the edge feature
            row_ts, col_ts = edge_index_transition_state
            atom_positions_transition_state = transition_state_structure.cart_coords
            atom_positions_transition_state = torch.tensor(
                atom_positions_transition_state, dtype=torch.float
            )
            edge_attributes_transition_state = torch.linalg.norm(
                atom_positions_transition_state[row_ts]
                - atom_positions_transition_state[col_ts],
                dim=1,
            )
            edge_attributes_transition_state = edge_attributes_transition_state.view(
                -1, 1
            )

            row_react, col_react = edge_index_reactant
            atom_positions_reactant = reactant_structure.cart_coords
            atom_positions_reactant = torch.tensor(
                atom_positions_reactant, dtype=torch.float
            )
            edge_attributes_reactant = torch.linalg.norm(
                atom_positions_reactant[row_react] - atom_positions_reactant[col_react],
                dim=1,
            )
            edge_attributes_reactant = edge_attributes_reactant.view(-1, 1)

            edge_index = {
                "reactant": edge_index_reactant,
                "transition_state": edge_index_transition_state,
            }
            edge_atributes = {
                "reactant": edge_attributes_reactant,
                "transition_state": edge_attributes_transition_state,
            }
            pos = {
                "reactant": atom_positions_reactant,
                "transition_state": atom_positions_transition_state,
            }

            # Collect 'y' which is determines if the reaction is feasible or not
            y_barrier = data_["reaction_barrier"]

            if y_barrier > self.feasibility_threshold:
                y = torch.tensor([1])
            else:
                y = torch.tensor([0])

            logging.debug("y: {}".format(y))

            data_to_return = dict(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_atributes,
                pos=pos,
                y=y,
                num_global_features=num_global_features,
                global_attr=global_features,
            )
            self.logger.debug("data_to_return: {}".format(data_to_return))
            # Create the data object
            data = Datapoint(**data_to_return)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
