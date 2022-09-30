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
from minimal_basis.dataset.utils import generate_graphs_by_method


class ChargeDataset(InMemoryDataset):

    GLOBAL_INFORMATION = ["state_fragment"]
    MOLECULE_INFORMATION = ["positions", "graphs"]
    FEATURE_INFORMATION = ["charges"]

    def __init__(self, filename: str, graph_generation_method: str = "sn2"):
        """Dataset for charges (a single float) as node features."""
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
                self.logging.info("State of molecule: {}".format(state_fragment))

                # --- Get molecule level information ---
                # Get the molecule object
                self.logging.info(
                    f"--- Molecule level information: {self.MOLECULE_INFORMATION}"
                )
                molecule_dict = self.input_data[reaction_id][molecule_id]["molecule"]
                if state_fragment == "initial_state":
                    molecules_in_reaction["reactants"].append(molecule_dict)
                    molecules_in_reaction["reactants_index"].append(molecule_id)
                elif state_fragment == "final_state":
                    molecules_in_reaction["products"].append(molecule_dict)
                    molecules_in_reaction["products_index"].append(molecule_id)
                molecule = Molecule.from_dict(molecule_dict)

                # Get the NBO charges (which will be used as node features)
                atom_charge = self.input_data[reaction_id][molecule_id]["atom_charge"]
                molecule_info_collected["x"][molecule_id] = atom_charge

                # --- Get the output information and store that in the node
                y = self.input_data[reaction_id][molecule_id]["transition_state_energy"]

            # Store information about the graph based on the generation method.
            generate_graphs_by_method(
                graph_generation_method=self.graph_generation_method,
                molecules_in_reaction=molecules_in_reaction,
                molecule_info_collected=molecule_info_collected,
            )

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
