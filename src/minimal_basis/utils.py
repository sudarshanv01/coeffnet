import os
import itertools
from typing import Dict, Union, List, Tuple
import glob

import numpy as np

import matplotlib.pyplot as plt

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN


def separate_graph(
    molecule: Molecule, starting_index=0
) -> Tuple[np.ndarray, List[int], int]:
    """
    NOTE: Modified from mjwen's eigenn package.
    Build a complete graph, where each node is connected to all other nodes.
    Args:
        N: number of atoms
        starting_index: index of the first graph to start from
    Returns:
        edge index, shape (2, N). For example, for a system with 3 atoms, this is
            [[0,0,1,1,2,2],
            [1,2,0,2,0,1]]
        num_neigh: number of neighbors for each atom
    """
    # Get the number of atoms in the molecule
    N = len(molecule.species)
    # Decide on the start and end index of this particular graph.
    N_start = starting_index
    N_end = N_start + N
    edge_index = np.asarray(
        list(zip(*itertools.permutations(range(N_start, N_end), r=2)))
    )
    num_neigh = [N - 1 for _ in range(N_start, N_end)]

    # Add the number of atoms to the starting index
    delta_starting_index = N

    return edge_index, num_neigh, delta_starting_index


def sn2_positions(
    molecule_list: List[Molecule], distance_to_translate=10
) -> List[Molecule]:
    """Generate the positions which will be put into the SN2 graph.
    The idea is to put the incoming (or outgoing) species suitably far
    away from the backbone."""

    assert len(molecule_list) == 2, "Only two molecules are allowed in the SN2 graph."

    # Determine which species is the incoming/outgoing one and which is
    # the backbone. The incoming/outgoing species is the one with the
    # fewer atoms.
    if len(molecule_list[0].species) < len(molecule_list[1].species):
        attacking = molecule_list[0]
        backbone = molecule_list[1]
    else:
        attacking = molecule_list[1]
        backbone = molecule_list[0]

    # Center the backbone and attacking molecule
    backbone_centered = backbone.get_centered_molecule()
    attacking_centered = attacking.get_centered_molecule()

    # Move the attacking_centered molecule 10A away from the backbone
    attacking_centered.translate_sites(vector=[distance_to_translate, 0, 0])

    return [backbone_centered, attacking_centered]


def sn2_graph(
    molecule_dict: Dict[int, Molecule], starting_index=0
) -> Tuple[np.ndarray, List[int], int, Dict[int, int]]:
    """Generate the SN2 graph where the each molecule in
    `molecule_list` is connected via the bonds, but the
    atoms between molecules are fully connected."""

    # Get the total number of atoms in the molecule list
    num_atoms_index = []
    counter_atoms = starting_index
    all_edges = []
    # Keep track of which edge belongs to which molecule
    # through mapping each edge with the choose_index
    edge_choose_index_mapping = {}
    # Keep track of the index of the atom in a molecule
    # by mapping it to the edge index
    edge_internal_mol_mapping = {}

    for k, (choose_index, molecule) in enumerate(molecule_dict.items()):
        # In this loop, a graph consisting of the atoms of a particular
        # molecule are generated. This is an _intra-molecular_ graph.
        # The index counter_atoms makes sure that the atoms are numbered
        # correctly in the globals scheme of things wrt reactants and products.

        mol_graph = MoleculeGraph.with_local_env_strategy(
            molecule, OpenBabelNN(order=True)
        )
        edges_mol = [
            (counter_atoms + i, counter_atoms + j)
            for i, j, attr in mol_graph.graph.edges.data()
        ]
        all_edges.extend(edges_mol)

        # Add the number of atoms to the total number of atoms
        # This is used to keep track of the total number of atoms
        # in the system, including all the seperate molecules in
        # either the reactants or the products.
        # `num_atoms_index` is supposed to function as the starting
        # index of a new molecule.
        num_atoms_index.append(counter_atoms)

        # Add the mapping between the edges and the choose_index
        if mol_graph.graph.edges.data():
            for edge in edges_mol:
                edge_choose_index_mapping[edge[0]] = choose_index
                edge_choose_index_mapping[edge[1]] = choose_index
                # Map the internal index of the atom in the molecule
                # to the edge index
                edge_internal_mol_mapping[edge[0]] = edge[0] - counter_atoms
                edge_internal_mol_mapping[edge[1]] = edge[1] - counter_atoms
        else:
            # There is no edge in the graph, so the molecule is a single atom
            # so include its index
            edge_choose_index_mapping[counter_atoms] = choose_index
            # Map the internal index of the atom in the molecule
            # to the edge index
            edge_internal_mol_mapping[counter_atoms] = 0

        # Add the number of atoms in the molecule to the counter
        counter_atoms += len(molecule.species)

    for i, n_i in enumerate(num_atoms_index):
        # In this loop, the _inter-molecular_ edges are added.
        # Each atom from molecule i is connected to each atom from molecule j
        # for i != j.

        if i == len(num_atoms_index) - 1:
            index_species_i = range(n_i, counter_atoms)
        else:
            index_species_i = range(n_i, num_atoms_index[i + 1])

        for j, n_j in enumerate(num_atoms_index):

            if i != j:

                # Only perform fully connected graphs between
                # molecules that are not the same.
                if j == len(num_atoms_index) - 1:
                    index_species_j = range(n_j, counter_atoms)
                else:
                    index_species_j = range(n_j, num_atoms_index[j + 1])

                # Get all combinations of the atoms between the two molecules
                # with index_species_i and index_species_j
                edges = list(zip(*itertools.product(index_species_i, index_species_j)))
                edges = np.array(edges).T.tolist()

                all_edges.extend(edges)

    all_edges.sort(key=lambda pair: pair[0])
    all_edges = np.array(all_edges).T.tolist()

    return (
        all_edges,
        None,
        counter_atoms,
        edge_choose_index_mapping,
        edge_internal_mol_mapping,
    )


def get_plot_params():
    """Create the plot parameters used in the plotting
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    COLOR = "k"
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["axes.labelcolor"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR


def avail_checkpoint(CHECKPOINT_DIR):
    """Check if a checkpoint file is available, and if it is
    return the name of the largest checkpoint file."""
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt"))
    if len(checkpoint_files) == 0:
        return None
    else:
        return max(checkpoint_files, key=os.path.getctime)


def visualize_results(
    predicted_train,
    calculated_train,
    predicted_validate,
    calculated_validate,
    plot_folder,
    epoch=None,
    loss=None,
):
    """Plot the DFT calculated vs the fit results."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)

    # Convert to numpy
    predicted_train = predicted_train.detach().cpu().numpy()
    calculated_train = calculated_train.detach().cpu().numpy()
    predicted_validate = predicted_validate.detach().cpu().numpy()
    calculated_validate = calculated_validate.detach().cpu().numpy()

    ax.scatter(calculated_train, predicted_train, cmap="Set2", label="Train")
    ax.scatter(calculated_validate, predicted_validate, cmap="Set3", label="Validate")
    if epoch is not None and loss is not None:
        ax.set_xlabel(f"Epoch: {epoch}, Loss: {loss.item():.3f}")
    ax.set_xlabel("DFT calculated (eV)")
    ax.set_ylabel("Fit results (eV)")
    ax.legend(loc="best")
    fig.savefig(f"{plot_folder}/step_{epoch:03d}.png", dpi=300)
    plt.close(fig)
