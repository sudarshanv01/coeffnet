import copy
from typing import Dict, Union, List, Tuple
import logging

from collections import defaultdict

from ase import data as ase_data

import numpy as np

from pymatgen.core.structure import Molecule

import torch

from torch_geometric.data import InMemoryDataset

from monty.serialization import loadfn

from minimal_basis.data import DataPoint
from minimal_basis.dataset.utils import generate_graphs_by_method

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)

logger = logging.getLogger(__name__)


class ChargeDataset(InMemoryDataset):

    GLOBAL_INFORMATION = ["state_fragment"]
    MOLECULE_INFORMATION = ["positions", "graphs"]
    FEATURE_INFORMATION = ["charges"]

    def __init__(
        self,
        root: str,
        transform: str = None,
        pre_transform: bool = None,
        pre_filter: bool = None,
        filename: str = None,
        graph_generation_method: str = "sn2",
    ):
        """Dataset for charges (a single float) as node features."""

        self.filename = filename
        self.graph_generation_method = graph_generation_method
        self.logging = logging.getLogger(__name__)

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def raw_file_names(self):
        return "input.json"

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        """Load data from json file."""
        logger.info("Loading data from json file.")
        self.input_data = loadfn(self.filename)
        self.logging.info("Successfully loaded json file with data.")

    def process(self):
        """Get the data from the json file."""

        # This function must return a list of Data objects
        # Each reaction corresponds to a Data object
        datapoint_list = []

        for reaction_idx, input_data_ in enumerate(self.input_data):

            input_data = copy.deepcopy(input_data_)

            # Collect the molecular level information
            molecule_info_collected = defaultdict(dict)

            # Store the molecules separately to make sure
            # that the graph creation process is flexible
            molecules_in_reaction = defaultdict(list)

            # The label for this reaction
            label = input_data.pop("label")

            # --- Get the output information and store that in the node
            y = input_data.pop("transition_state_energy")
            y = torch.tensor([y], dtype=DTYPE)

            # --- Get the global information
            global_information = input_data.pop("reaction_energy")
            global_information = torch.tensor([global_information], dtype=DTYPE)

            for molecule_id in input_data:
                # Prepare the information for each molecule, this forms
                # part of the graph that is making up the DataPoint.

                # --- Get state (global) level information ---
                # Get the state of the molecule, i.e., does it
                # belong to the initial or final state of the reaction?
                self.logging.debug(
                    f"--- Global level information: {self.GLOBAL_INFORMATION}"
                )
                state_fragment = input_data[molecule_id]["state_fragments"]
                self.logging.debug("State of molecule: {}".format(state_fragment))

                # --- Get molecule level information ---
                # Get the molecule object
                self.logging.debug(
                    f"--- Molecule level information: {self.MOLECULE_INFORMATION}"
                )
                molecule = input_data[molecule_id]["molecule"]
                if state_fragment == "initial_state":
                    molecules_in_reaction["reactants"].append(molecule)
                    molecules_in_reaction["reactants_index"].append(molecule_id)
                elif state_fragment == "final_state":
                    molecules_in_reaction["products"].append(molecule)
                    molecules_in_reaction["products_index"].append(molecule_id)

                # Get the NBO charges (which will be used as node features)
                atom_charge_dict = input_data[molecule_id]["atom_charge"]
                # Make atom charges a list
                atom_charge = [
                    atom_charge_dict[atom_idx] for atom_idx in atom_charge_dict
                ]
                # Get the atomic number of the atom
                atomic_numbers = [
                    ase_data.atomic_numbers[str(atom)] for atom in molecule.species
                ]
                # Make everything a float
                atom_charge = [float(charge) for charge in atom_charge]
                atomic_numbers = [float(number) for number in atomic_numbers]

                # Input features will be a list containing the atomic number and the
                # NBO charge as a list.
                input_node_features = [
                    [atomic_number, charge]
                    for atomic_number, charge in zip(atomic_numbers, atom_charge)
                ]
                molecule_info_collected["x"][molecule_id] = input_node_features

            # Store information about the graph based on the generation method.
            generate_graphs_by_method(
                graph_generation_method=self.graph_generation_method,
                molecules_in_reaction=molecules_in_reaction,
                molecule_info_collected=molecule_info_collected,
            )

            edge_index = molecule_info_collected["edge_index"]
            tensor_edge_index = [[], []]
            for state_index in sorted(edge_index):
                if edge_index[state_index] is not None:
                    edge_scr, edge_dest = edge_index[state_index]
                    tensor_edge_index[0].extend(edge_scr)
                    tensor_edge_index[1].extend(edge_dest)

            # Convert tensor_edge_index to a tensor
            tensor_edge_index = torch.as_tensor(tensor_edge_index, dtype=DTYPE_INT)
            rows, cols = tensor_edge_index

            # Convert the positions to a tensor.
            tensor_pos = []
            for state_index in sorted(molecule_info_collected["pos"]):
                if molecule_info_collected["pos"][state_index] is not None:
                    tensor_pos.extend(molecule_info_collected["pos"][state_index])
            tensor_pos = torch.as_tensor(tensor_pos, dtype=DTYPE)

            # Make a list of distances between the atoms that are in edge index
            edge_attr = torch.linalg.norm(tensor_pos[rows] - tensor_pos[cols], dim=1)

            # Store the datapoint object
            datapoint = DataPoint(
                pos=tensor_pos,
                edge_index=molecule_info_collected["edge_index"],
                x=molecule_info_collected["x"],
                y=y,
                global_attr=global_information,
                edge_attr=edge_attr,
            )

            logging.info("Datapoint:")
            logging.info(datapoint)
            logging.info("-------")

            # Store the datapoint in a list
            datapoint_list.append(datapoint)

        # Store the list of datapoints in the dataset
        data, slices = self.collate(datapoint_list)
        self.data = data
        self.slices = slices

        # Save the dataset
        torch.save((data, slices), self.processed_paths[0])
