import json
from typing import Dict, Union, List, Tuple
import logging

from collections import defaultdict

from ase import data as ase_data

import numpy as np

from torch_geometric.data import InMemoryDataset

from pymatgen.core.structure import Molecule

import networkx as nx
import matplotlib.pyplot as plt

from minimal_basis.data import DataPoint

from minimal_basis.utils import separate_graph, sn2_graph, sn2_positions


class ChargeDataset(InMemoryDataset):
    def __init__(self, filename: str, graph_generation_method: str = "sn2"):
        """Create a dataset consisting of just NBO data as the
        node features."""
        self.filename = filename
        self.graph_generation_method = graph_generation_method
        self.logging = logging.getLogger(__name__)

        super().__init__()

    def load_data(self):
        """Load data from json file."""
        with open(self.filename) as f:
            self.input_data = json.load(f)
        self.logging.info("Successfully loaded json file with data.")

    def len(self):
        """Get the length of the dataset."""
        return len(self.input_data)

    def get_data(self):
        """Get the data from the json file."""

        # This function must return a list of Data objects
        # Each reaction corresponds to a Data object
        datapoint_list = []

        for reaction_id in self.input_data:
            # The reaction_id is the key for the reactions database
            index_prod = 0
            index_react = 0

            # Collect the molecular level information
            molecule_info_collected = defaultdict(dict)

            # Store the molecules separately to make sure
            # that the graph creation process is flexible
            molecules_in_reaction = defaultdict(list)

            for molecule_id in self.input_data[reaction_id]:
                # Prepare the information for each molecule, this forms
                # part of the graph that is making up the DataPoint.

                # --- Get state (global) level information ---
                # Get the state of the molecule, i.e., does it
                # belong to the initial or final state of the reaction?
                self.logging.info(
                    f"--- Global level information: {self.GLOBAL_INFORMATION}"
                )
                state_fragment = self.input_data[reaction_id][molecule_id][
                    "state_fragments"
                ]
                if state_fragment == "initial_state":
                    index_react -= 1
                    choose_index = index_react
                elif state_fragment == "final_state":
                    index_prod += 1
                    choose_index = index_prod
                else:
                    raise ValueError("State fragment not recognised.")
                self.logging.info("State of molecule: {}".format(state_fragment))

                # --- Get molecule level information ---
                # Get the molecule object
                self.logging.info(
                    f"--- Molecule level information: {self.MOLECULE_INFORMATION}"
                )
                molecule_dict = self.input_data[reaction_id][molecule_id]["molecule"]
                if state_fragment == "initial_state":
                    molecules_in_reaction["reactants"].append(molecule_dict)
                    molecules_in_reaction["reactants_index"].append(choose_index)
                elif state_fragment == "final_state":
                    molecules_in_reaction["products"].append(molecule_dict)
                    molecules_in_reaction["products_index"].append(choose_index)
                molecule = Molecule.from_dict(molecule_dict)

                # Get the NBO charges (which will be used as node features)
                nbo_charge = self.input_data[reaction_id][molecule_id]["nbo_charge"]
                molecule_info_collected["x"][choose_index] = nbo_charge

                # --- Get the output information and store that in the node
                y = self.input_data[reaction_id][molecule_id]["transition_state_energy"]

            # Generate a separate graph for each reactant and product.
            # There are several possibilities to generate such a graph
            # so depending on the option selected, the relative ordering
            # of the dict may change, this is expected behaviour.
            if self.graph_generation_method == "separate":
                # Generate an internally fully connected graph between
                # each atom in a specific molecule. Separate molecules
                # are not linked to each other.

                # Keep a tab on the index of the molecule
                starting_index = 0

                # There is no separation between reactants and products
                # with this method, but since they are stored in separate
                # entries in the dictionary, this is not a problem.
                for states in ["reactants", "products"]:
                    molecules_list = molecules_in_reaction[states]
                    choose_indices = molecules_in_reaction[states + "_index"]

                    for k, choose_index in enumerate(choose_indices):
                        # Choose the corresponding molecule
                        molecule = Molecule.from_dict(molecules_list[k])

                        # Construct a graph from the molecule object, each node
                        # of the graph is connected with every other node.
                        edge_index, _, delta_starting_index = self.separate_graph(
                            molecule,
                            starting_index=starting_index,
                        )

                        # Get the positions of the molecule (these are cartesian coordinates)
                        pos = [list(a.coords) for a in molecule]

                        molecule_info_collected["pos"][choose_index] = pos

                        # if edge_index is empty, then it is monoatomic and hence
                        # the edges must be connected to themselves.
                        if len(edge_index) == 0:
                            edge_index = [[starting_index], [starting_index]]
                        # Move the starting index so that the next molecule comes after
                        # this molecule.
                        starting_index += delta_starting_index
                        molecule_info_collected["edge_index"][choose_index] = edge_index

            elif self.graph_generation_method == "sn2":
                # In this generation scheme, the reactant and product are
                # have separate graphs which are connected to each other
                # the positions of each fragments have to altered

                # Keep a tab on the index of the molecule
                starting_index = 0

                # This dictionary creates a mapping between the edges
                # list, which is a cumulative list of all the edges
                # and the molecule from which they came from.
                edge_molecule_mapping = {}
                # This dictionary creates a mapping between the edges
                # list and the internal index of the molecule.
                edge_internal_mol_mapping = {}

                # There is no separation between reactants and products
                # with this method, but since they are stored in separate
                # entries in the dictionary, this is not a problem.
                for states in ["reactants", "products"]:
                    molecules_list = molecules_in_reaction[states]
                    # Convert each molecule_list to a molecule
                    molecules_list = [
                        Molecule.from_dict(molecule) for molecule in molecules_list
                    ]
                    choose_indices = molecules_in_reaction[states + "_index"]
                    # Create a dict between molecule_list and choose_indices
                    molecule_dict = {}
                    for k, choose_index in enumerate(choose_indices):
                        molecule_dict[choose_index] = molecules_list[k]

                    # Reorder the molecular positions
                    molecules_list = self.sn2_positions(molecules_list)

                    # Generate the graph based on the sn2 method
                    (
                        edge_index,
                        _,
                        delta_starting_index,
                        edge_molecule_mapping_,
                        edge_internal_mol_mapping_,
                    ) = self.sn2_graph(
                        molecule_dict,
                        starting_index=starting_index,
                    )
                    edge_molecule_mapping.update(edge_molecule_mapping_)
                    edge_internal_mol_mapping.update(edge_internal_mol_mapping_)

                    starting_index += delta_starting_index

                    for k, choose_index in enumerate(choose_indices):

                        # Choose the corresponding molecule
                        molecule = molecules_list[k]

                        # Get the positions of the molecule (these are cartesian coordinates)
                        pos = [list(a.coords) for a in molecule]
                        molecule_info_collected["pos"][choose_index] = pos

                        # Store the edge_index for only one of the molecules
                        # because they will be the same
                        if k == 0:
                            molecule_info_collected["edge_index"][
                                choose_index
                            ] = edge_index
                        else:
                            molecule_info_collected["edge_index"][choose_index] = None

            # Store the datapoint object
            datapoint = DataPoint(
                pos=molecule_info_collected["pos"],
                edge_index=molecule_info_collected["edge_index"],
                x=molecule_info_collected["x"],
                y=y,
            )

            logging.info("Datapoint:")
            logging.info(datapoint)
            logging.info("-------")

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        return datapoint_list
