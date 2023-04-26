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


def expected_endstate(
    reactant_complex: Molecule,
    transition_state: Molecule,
    product_complex: Molecule,
    label: str,
    idx_carbon_node: int,
    ts_energy: float,
) -> Tuple[bool, str]:

    # The transition state energy referenced to the mean of the
    # reactant and product complex energies cannot be negative
    if ts_energy < 0:
        logger.warning("TS energy is negative, skipping.")
        return False

    # Get the index of the nucleophile and the leaving group
    R1, R2, R3, R4, X, Y = label.split("_")
    logger.info(f"X: {MATCHER['X'][X]} Y: {MATCHER['Y'][Y]}")

    # Get the index of X and Y in all the complexes
    idx_X = [
        i for i, a in enumerate(transition_state) if a.specie.symbol == MATCHER["X"][X]
    ]
    idx_Y = [
        i for i, a in enumerate(transition_state) if a.specie.symbol == MATCHER["Y"][Y]
    ]

    if len(idx_X) != 1 or len(idx_Y) != 1:
        return False

    idx_X = idx_X[0]
    idx_Y = idx_Y[0]

    # Get the difference between the C-X and C-Y distances for the positive and negative
    # perturbed molecules
    dist_pos = reactant_complex.get_distance(
        idx_carbon_node, idx_X
    ) - reactant_complex.get_distance(idx_carbon_node, idx_Y)
    dist_neg = product_complex.get_distance(
        idx_carbon_node, idx_X
    ) - product_complex.get_distance(idx_carbon_node, idx_Y)
    logger.info(f"dist_pos: {dist_pos} dist_neg: {dist_neg}")

    # dist_pos and dist_neg must have opposite signs
    if np.sign(dist_pos) == np.sign(dist_neg):
        return False

    # Make a graph of the reactant and product complexes
    reactant_graph = MoleculeGraph.with_local_env_strategy(
        reactant_complex, OpenBabelNN()
    )
    product_graph = MoleculeGraph.with_local_env_strategy(
        product_complex, OpenBabelNN()
    )

    # Make sure that the coordination of the carbon site is exactly 4
    # and the nucleophile has a coordination number of 0 in the reactant
    # and 1 in the product and the leaving group has a coordination number
    # of 1 in the reactant and 0 in the product
    reactant_carbon_coordination_number = reactant_graph.get_coordination_of_site(
        idx_carbon_node
    )
    product_carbon_coordination_number = product_graph.get_coordination_of_site(
        idx_carbon_node
    )
    if (
        reactant_carbon_coordination_number != 4
        or product_carbon_coordination_number != 4
    ):
        logger.warning(
            f"Carbon coordination number is not 4, not an endstate, skipping."
        )
        return False

    reactant_nucleophile_coordination_number = reactant_graph.get_coordination_of_site(
        idx_X
    )
    product_nucleophile_coordination_number = product_graph.get_coordination_of_site(
        idx_X
    )
    # If reactant_nucleophile is 0 then product_nucleophile must be 1 and vice versa
    if (
        reactant_nucleophile_coordination_number == 0
        and product_nucleophile_coordination_number == 1
    ):
        pass
    elif (
        reactant_nucleophile_coordination_number == 1
        and product_nucleophile_coordination_number == 0
    ):
        pass
    else:
        logger.warning(
            f"Nucleophile coordination number is not 1, not an endstate, skipping."
        )
        return False

    reactant_leaving_group_coordination_number = (
        reactant_graph.get_coordination_of_site(idx_Y)
    )
    product_leaving_group_coordination_number = product_graph.get_coordination_of_site(
        idx_Y
    )
    # If reactant_leaving_group is 1 then product_leaving_group must be 0 and vice versa
    if (
        reactant_leaving_group_coordination_number == 1
        and product_leaving_group_coordination_number == 0
    ):
        pass
    elif (
        reactant_leaving_group_coordination_number == 0
        and product_leaving_group_coordination_number == 1
    ):
        pass
    else:
        logger.warning(
            f"Leaving group coordination number is not 0, not an endstate, skipping."
        )
        return False

    # If it passes all tests return that is is a valid pair of endstates
    return True


if __name__ == "__main__":
    """Check if the endstate structures are the right structures for the TS."""

    db = instance_mongodb_sei(project="mlts")

    # Choose the minimum basis calculations
    collection = db.minimal_basis
    initial_structure_collection = db.minimal_basis_initial_structures

    # Get the transition states from the database and determine the
    # endstates from the same database.
    ts_prompt = collection.find({"tags.group": "sn2_transition_states"})

    # Index of the carbon node
    idx_carbon_node = 0

    output_dir = "sn2_endstates"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for ts_doc in ts_prompt:
        # Get the transition state _id
        _id = str(ts_doc["_id"])

        # Store the computational method
        method = ts_doc["orig"]["rem"]["method"]
        basis = ts_doc["orig"]["rem"]["basis"]
        label = ts_doc["tags"]["label"]

        # Get the transition state molecule
        ts_structure = ts_doc["output"]["optimized_molecule"]
        ts_structure = Molecule.from_dict(ts_structure)

        # endstate prompt
        endstate_prompt = {
            "tags.ts_idkey": _id,
            "tags.group": "endstate_structures_from_ts",
            "orig.rem.method": method,
            "orig.rem.basis": basis,
        }

        # Use the transition state _id to get the
        # reactant and product docs
        endstate_structures = []
        endstate_energies = []
        ids_endstates = []
        for endstates_doc in collection.find(endstate_prompt):
            endstate_structures.append(
                Molecule.from_dict(endstates_doc["output"]["optimized_molecule"])
            )
            endstate_energies.append(endstates_doc["output"]["final_energy"])
            ids_endstates.append(str(endstates_doc["_id"]))
        logger.info(f"Found {len(endstate_structures)} endstate structures for {label}")
        if len(endstate_structures) != 2:
            logger.warning(
                f"Found {len(endstate_structures)} endstate structures for {label}"
            )
            continue

        ts_atoms = AseAtomsAdaptor.get_atoms(ts_structure)
        reactant_atoms = AseAtomsAdaptor.get_atoms(endstate_structures[0])
        product_atoms = AseAtomsAdaptor.get_atoms(endstate_structures[1])

        # Get the transition state energy referenced to the mean of the reactant
        # and product complex energies
        ts_energy = ts_doc["output"]["final_energy"]
        # Take the mean of the reactant and product energies
        mean_energy = np.mean(endstate_energies)
        # Subtract the mean energy from the transition state energy
        ts_energy = ts_energy - mean_energy
        logger.info(f"TS energy (ref to mean complex energy): {ts_energy}")

        # Verify that the endstates are the correct endstates
        # for the transition state
        if expected_endstate(
            endstate_structures[0],
            ts_structure,
            endstate_structures[1],
            label,
            idx_carbon_node,
            ts_energy,
        ):
            logger.info(f"Endstates are correct for {label}")
        else:
            logger.warning(f"Endstates are incorrect for {label}")
            continue

        # Store the transition state, endstate and final state structures
        # along with the relevant _ids to link back to the relaxation calculations
        tags = {
            "group": "sn2_reactions",
            "label": label,
            "ts_idkey": str(ts_doc["_id"]),
            "reactant_idkey": ids_endstates[0],
            "product_idkey": ids_endstates[1],
            "basis": basis,
            "method": method,
        }
        molecules = {
            "transition_state": ts_structure.as_dict(),
            "reactant": endstate_structures[0].as_dict(),
            "product": endstate_structures[1].as_dict(),
        }
        data_to_initial_structuresdb = {
            "tags": tags,
            "molecules": molecules,
        }
        # Store the data in the database
        initial_structure_collection.insert_one(data_to_initial_structuresdb)

        # Replace any dashes with underscores and any * with star
        # and write out the endstate structures
        basis = basis.replace("-", "_")
        basis = basis.replace("*", "star")
        ase_io.write(
            f"{output_dir}/{method}_{basis}_{label}_endstates.xyz",
            [reactant_atoms, ts_atoms, product_atoms],
        )
