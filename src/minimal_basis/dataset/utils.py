from typing import Dict, List, Union

from pymatgen.core.structure import Molecule

from collections import defaultdict

from minimal_basis.utils import separate_graph, sn2_graph, sn2_positions


def generate_graphs_by_method(
    graph_generation_method: str = "separate",
    molecules_in_reaction: Dict[str, dict] = {},
    states: List[str] = ["reactants", "products"],
    molecule_info_collected: Dict[str, Dict[int, List[list]]] = defaultdict(dict),
) -> Union[None, Dict]:

    if graph_generation_method == "separate":
        # Generate an internally fully connected graph between
        # each atom in a specific molecule. Separate molecules
        # are not linked to each other.

        # Keep a tab on the index of the molecule
        starting_index = 0

        # There is no separation between reactants and products
        # with this method, but since they are stored in separate
        # entries in the dictionary, this is not a problem.
        for state in states:
            molecules_list = molecules_in_reaction[state]
            choose_indices = molecules_in_reaction[state + "_index"]

            for k, choose_index in enumerate(choose_indices):
                # Choose the corresponding molecule
                molecule = Molecule.from_dict(molecules_list[k])

                # Construct a graph from the molecule object, each node
                # of the graph is connected with every other node.
                edge_index, _, delta_starting_index = separate_graph(
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

        return None, None

    elif graph_generation_method == "sn2":
        # In this graph generation scheme, the reactants and
        # products have their own (separate) graphs. The reactants
        # and products have similar structure - the backbone is connected
        # by the different bonds of the molecule and the attacking (or leaving)
        # group is fully connected to all atoms of the backbone.

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
        for state in states:
            molecules_list = molecules_in_reaction[state]
            choose_indices = molecules_in_reaction[state + "_index"]
            # Create a dict between molecule_list and choose_indices
            molecule_dict = {}
            for k, choose_index in enumerate(choose_indices):
                molecule_dict[choose_index] = molecules_list[k]

            # Reorder the molecular positions
            molecules_list = sn2_positions(molecules_list)

            # Generate the graph based on the sn2 method
            (
                edge_index,
                _,
                delta_starting_index,
                edge_molecule_mapping_,
                edge_internal_mol_mapping_,
            ) = sn2_graph(
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
                    molecule_info_collected["edge_index"][choose_index] = edge_index
                else:
                    molecule_info_collected["edge_index"][choose_index] = None

        return edge_molecule_mapping, edge_internal_mol_mapping
