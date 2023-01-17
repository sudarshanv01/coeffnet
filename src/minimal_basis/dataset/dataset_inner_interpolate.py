import logging

from monty.serialization import loadfn

import numpy as np

from ase import data as ase_data

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

import torch

from torch_geometric.data import InMemoryDataset

from minimal_basis.data import InnerInterpolateDatapoint as Datapoint

logger = logging.getLogger(__name__)

from minimal_basis.data._dtype import (
    DTYPE,
    DTYPE_INT,
    DTYPE_BOOL,
    TORCH_FLOATS,
    TORCH_INTS,
)


class InnerInterpolateDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        filename: str = None,
    ):
        """Dataset for the Inner interpolate model."""
        self.filename = filename

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
        return "inner_interpolate_data.pt"

    def download(self):
        logger.info("Loading data from json file.")
        self.input_data = loadfn(self.filename)

    def process(self):
        data_list = []

        for data_ in self.input_data:
            # This data point should contain the atomic structure
            # The details of what this structure is doesnt really matter
            # as we are doing a structure -> energy mapping
            structure = data_["structure"]
            if isinstance(structure, dict):
                structure = Molecule.from_dict(structure)

            # We need the energy which is the quantity to predict
            energy = data_["energy"]

            # --- Global features ---
            charge = data_["charge"]
            spin_multiplicity = data_["spin_multiplicity"]
            global_features = [
                charge,
                spin_multiplicity,
            ]
            num_global_features = len(global_features)
            global_features = torch.tensor(global_features, dtype=TORCH_FLOATS[1])

            # --- Node features ---
            atomic_numbers = [
                ase_data.atomic_numbers[species.species_string] for species in structure
            ]
            atomic_charges = data_["atomic_charges"]
            # Make a tensor of the atomic numbers and charges
            atomic_numbers = torch.tensor(atomic_numbers, dtype=DTYPE_INT)
            atomic_charges = torch.tensor(atomic_charges, dtype=TORCH_FLOATS[1])
            node_features = torch.cat(
                [
                    atomic_numbers.unsqueeze(1),
                    atomic_charges.unsqueeze(1),
                ],
                dim=1,
            )

            # --- Edge features ---
            # Make a pymatgen MoleculeGraph
            structure_graph = MoleculeGraph.with_local_env_strategy(
                structure, OpenBabelNN()
            )
            structure_graph = metal_edge_extender(structure_graph)

            edges_for_graph = structure_graph.graph.edges
            edges_for_graph = [list(edge[:-1]) for edge in edges_for_graph]
            logger.debug("edges_for_graph: {}".format(edges_for_graph))
            edge_index = torch.tensor(edges_for_graph, dtype=torch.long)
            logger.debug("edge_index: {}".format(edge_index))
            edge_index = edge_index.t().contiguous()

            # Collect the edge features, current the bond length
            # between two atom centres will be used as the edge feature

            # if `edge_index` is an empty tensor, then create empty
            # tensors for row and col
            if edge_index.shape[0] == 0:
                row = torch.tensor([], dtype=torch.long)
                col = torch.tensor([], dtype=torch.long)
            else:
                row, col = edge_index

            atom_positions = structure.cart_coords
            atom_positions = torch.tensor(atom_positions, dtype=torch.float)
            edge_attributes = torch.linalg.norm(
                atom_positions[row] - atom_positions[col], dim=1
            )
            edge_attributes = edge_attributes.view(-1, 1)

            # Create the datapoint
            data = Datapoint(
                x=node_features,
                edge_index=edge_index,
                y=energy,
                global_attr=global_features,
                edge_attr=edge_attributes,
                num_global_features=num_global_features,
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
