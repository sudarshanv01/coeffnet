"""Export the transition states as xyz files with the perturbed vectors."""
import os
from typing import List, Tuple
import copy
import logging
import argparse

import numpy as np

from ase import io as ase_io

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from instance_mongodb import instance_mongodb_sei

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MATCHER = {
    "X": {
        "A": "F",
        "B": "Cl",
        "C": "Br",
    },
    "Y": {
        "A": "H",
        "B": "F",
        "C": "Cl",
        "D": "Br",
    },
}


def perturb_along_eigenmode(
    ts_molecule: Molecule, eigenmode: List[float], scaling: float
) -> Molecule:
    """Perturn the molecule along the eigen modes based on a scaling factor.

    Args:
        ts_molecule: The transition state molecule.
        eigenmode: The eigenmode.
        scaling: The scaling factor to perturb the molecule.
    """

    def validate_eigenmode(eigenmode: List[float]) -> None:
        """Check if the eigenmode is normalised correctly."""
        norm_eigemode = np.linalg.norm(eigenmode)
        is_close = np.isclose(norm_eigemode, 1.0, atol=1e-3)
        if not is_close:
            raise ValueError("The eigenmode is not normalised correctly.")

    eigenmode = np.array(eigenmode)
    validate_eigenmode(eigenmode)
    assert eigenmode.shape == (
        len(ts_molecule),
        3,
    ), "Eigenmode is not the correct shape."

    delta_pos = scaling * eigenmode
    perturbed_molecule_pos = copy.deepcopy(ts_molecule)

    # get positions of atoms
    positions = [a.coords for a in ts_molecule.sites]
    positions = np.array(positions)

    # Get the perturbed positions
    perturbed_pos = positions + delta_pos

    # Set the perturbed positions
    for i, a in enumerate(perturbed_molecule_pos.sites):
        a.coords = perturbed_pos[i]

    return perturbed_molecule_pos


def expected_transition_state(
    ts_molecule: Molecule,
    ts_molecule_graph: MoleculeGraph,
    eigenmode: List[float],
    eigenvalue: List[float],
    idx_carbon_node: int,
    label: str,
) -> Molecule:
    """Based on the rules outlined in this function, decide if the transition state
    is the one we are looking for.

    Args:
        ts_molecule: The transition state molecule.
        ts_molecule_graph: The transition state molecule graph.
        eigenmode: The eigenmode.
        eigenvalue: The eigenvalue.
        idx_carbon_node: The index of the carbon node in the molecule graph.
        label: The label of the transition state.
    """
    R1, R2, R3, R4, X, Y = label.split("_")
    logger.info(f"X: {MATCHER['X'][X]} Y: {MATCHER['Y'][Y]}")

    imag_eigval_idx = np.where(eigenvalue < 0)
    # There must be exactly one negative eigenvalue
    # if len(imag_eigval_idx[0]) != 1:
    #     return False

    # Based on the transition state graph find the connected fragments
    # of the molecule at the index of the carbon node
    coordination_carbon = ts_molecule_graph.get_coordination_of_site(idx_carbon_node)
    disconnected_fragments = ts_molecule_graph.get_disconnected_fragments()

    # Get the indices of atom with chemical formula MATHCER['X'][X]
    # and MATHCER['Y'][Y]
    if MATCHER["X"][X] != MATCHER["Y"][Y]:
        idx_X = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["X"][X]
        ]
        idx_Y = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["Y"][Y]
        ]

        if len(idx_X) != 1 or len(idx_Y) != 1:
            return False

        idx_X = idx_X[0]
        idx_Y = idx_Y[0]
    else:
        idx_X = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["X"][X]
        ]
        idx_Y = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["Y"][Y]
        ]
        idx_X = idx_X[0]
        idx_Y = idx_Y[1]

    # Perturb the transition state along the eigenmode in the positive direction
    perturbed_molecule_pos = perturb_along_eigenmode(ts_molecule, eigenmode, 0.5)
    perturbed_molecule_neg = perturb_along_eigenmode(ts_molecule, eigenmode, -0.5)

    # Get the difference between the C-X and C-Y distances for the positive and negative
    # perturbed molecules
    dist_pos = perturbed_molecule_pos.get_distance(
        idx_carbon_node, idx_X
    ) - perturbed_molecule_pos.get_distance(idx_carbon_node, idx_Y)
    dist_neg = perturbed_molecule_neg.get_distance(
        idx_carbon_node, idx_X
    ) - perturbed_molecule_neg.get_distance(idx_carbon_node, idx_Y)
    logger.info(f"dist_pos: {dist_pos} dist_neg: {dist_neg}")

    # dist_pos and dist_neg must have opposite signs
    if np.sign(dist_pos) == np.sign(dist_neg):
        return False

    return True


def get_cli():

    args = argparse.ArgumentParser()
    args.add_argument(
        "--store_endstates",
        action="store_true",
        help="Store the endstates in the database.",
        default=False,
    )

    return args.parse_args()


if __name__ == "__main__":

    db = instance_mongodb_sei(project="mlts")

    # Choose the minimum basis calculations
    collection = db.minimal_basis
    initial_structure_collection = db.minimal_basis_initial_structures

    # Get the length of the collection
    n_docs = collection.count_documents({})
    logger.info(f"Number of documents in the collection: {n_docs}")

    # Get the args
    args = get_cli()

    # Perturb the structure along the eigenmode by some amount
    perturb_eigemode_range = np.linspace(-0.5, 0.5, 11)

    # Make sure the output directory exists
    output_dir = "sn2_transition_states"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Index of the carbon node
    idx_carbon_node = 0

    # Collection name for the endstates structures
    groupname_endstates = "endstate_structures_from_ts"

    scaling_chosen = 0.6

    # Write out the transition states for all the structures in the collection
    for doc in collection.find({"tags.group": "sn2_transition_states"}):

        # Store the computational method
        method = doc["orig"]["rem"]["method"]
        basis = doc["orig"]["rem"]["basis"]
        # Replace any dashes with underscores and any * with star
        basis = basis.replace("-", "_")
        basis = basis.replace("*", "star")

        molecule_dict = doc["output"]["optimized_molecule"]
        molecule = Molecule.from_dict(molecule_dict)
        atoms = AseAtomsAdaptor.get_atoms(molecule)
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        # Get the fragments connected to the transition state center
        connected_fragments = molecule_graph.get_connected_sites(idx_carbon_node)

        eigenvectors = doc["output"]["frequency_modes"]
        eigenvectors = np.array(eigenvectors)
        eigenvalues = doc["output"]["frequencies"]
        eigenvalues = np.array(eigenvalues)

        label = doc["tags"]["label"]
        _id_ts = str(doc["_id"])

        # Get the imaginary frequency, which is the index
        # of the negative mode in eigenvales
        # If there are no negative frequencies, then skip
        if np.all(eigenvalues > 0):
            logger.info("No negative frequencies, skipping.")
            continue

        imag_freq_idx = np.where(eigenvalues < 0)[0][0]
        imag_freq = eigenvalues[imag_freq_idx]
        imag_eigenmode = eigenvectors[imag_freq_idx]

        # Check if the transition state is the one we are looking for
        if not expected_transition_state(
            molecule, molecule_graph, imag_eigenmode, imag_freq, idx_carbon_node, label
        ):
            logger.info("Transition state is not the one we are looking for, skipping.")
            continue

        # Perturb the molecule along the eigenmode
        transition_state_eigenmode = []
        for scaling in perturb_eigemode_range:
            perturbed_molecule = perturb_along_eigenmode(
                molecule, imag_eigenmode, scaling_chosen
            )
            transition_state_eigenmode.append(
                AseAtomsAdaptor.get_atoms(perturbed_molecule)
            )

        perturbed_molecule_positive = perturb_along_eigenmode(
            molecule, imag_eigenmode, scaling_chosen
        )
        perturbed_molecule_negative = perturb_along_eigenmode(
            molecule, imag_eigenmode, -scaling_chosen
        )

        # Store the endstates in the database
        perturbed_molecule_positive_dict = perturbed_molecule_positive.as_dict()
        perturbed_molecule_negative_dict = perturbed_molecule_negative.as_dict()
        perturbed_molecule_dicts = [
            perturbed_molecule_positive_dict,
            perturbed_molecule_negative_dict,
        ]

        # Store the endstates in the database with the tags of the negative and positive endstates
        # Also store the scaling factor and the transition state molecule, as well as the label
        # of the transition state as tags
        tags = {
            "scaling": scaling,
            "label": label,
            "ts_idkey": _id_ts,
            "group": groupname_endstates,
        }
        entries = {}
        entries["endstates"] = perturbed_molecule_dicts
        entries["tags"] = tags

        if args.store_endstates:

            # Check if the tags already exist in the database
            # If they do, then don't store them
            # If they don't, then store them
            if initial_structure_collection.count_documents({"tags": tags}) == 0:
                logger.info("Storing the endstate in the database.")
                initial_structure_collection.insert_one(entries)
            else:
                logger.info("The positive endstate is already in the database.")

        perturbed_molecule_positive_ase = AseAtomsAdaptor.get_atoms(
            perturbed_molecule_positive
        )
        perturbed_molecule_negative_ase = AseAtomsAdaptor.get_atoms(
            perturbed_molecule_negative
        )

        # Write out the transition states
        logger.info(
            f"Writing out transition states for {label} computed with {method}/{basis}"
        )
        ase_io.write(
            f"{output_dir}/{method}_{basis}_{label}_eigenmode.xyz",
            transition_state_eigenmode,
        )

        # Write out the endstates with the transition state in the middle
        logger.info(f"Writing out endstates for {label} computed with {method}/{basis}")
        ase_io.write(
            f"{output_dir}/{method}_{basis}_{label}_endstates.xyz",
            [perturbed_molecule_positive_ase, atoms, perturbed_molecule_negative_ase],
        )
