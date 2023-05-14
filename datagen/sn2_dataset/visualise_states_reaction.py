from typing import List

import logging

import os

import copy

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import ase.io as ase_io
from ase.neb import NEB


from instance_mongodb import instance_mongodb_sei

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    """Visualize the initial, transition and final states of the reaction."""

    if not os.path.exists("output/visualise_states"):
        os.makedirs("output/visualise_states")

    db = instance_mongodb_sei(project="mlts")
    collection = db.rudorff_lilienfeld_data

    for doc in collection.find({}):
        transition_state_molecule = doc["transition_state_molecule"]
        transition_state_molecule = Molecule.from_dict(transition_state_molecule)

        rxn_number = doc["rxn_number"]
        reaction_name = doc["reaction_name"]

        perturbed_molecule_keys = [
            key
            for key in doc.keys()
            if "perturbed_molecule" in key and "_energy" not in key
        ]
        conditional_accept = (
            len(perturbed_molecule_keys) > 0
            and len([key for key in perturbed_molecule_keys if "-" in key]) > 0
            and len([key for key in perturbed_molecule_keys if "-" not in key]) > 0
        )
        if not conditional_accept:
            continue

        positive_perturbed_molecules = [
            doc[key] for key in perturbed_molecule_keys if "-" not in key
        ]

        negative_perturbed_molecules = [
            doc[key] for key in perturbed_molecule_keys if "-" in key
        ]

        initial_state_molecule = positive_perturbed_molecules[0]
        initial_state_molecule = Molecule.from_dict(initial_state_molecule)

        final_state_molecule = negative_perturbed_molecules[0]
        final_state_molecule = Molecule.from_dict(final_state_molecule)

        initial_state_atoms = AseAtomsAdaptor.get_atoms(initial_state_molecule)
        final_state_atoms = AseAtomsAdaptor.get_atoms(final_state_molecule)
        transition_state_atoms = AseAtomsAdaptor.get_atoms(transition_state_molecule)

        initial_to_transition = NEB(
            [initial_state_atoms.copy() for i in range(5)]
            + [transition_state_atoms.copy()],
        )
        transition_to_final = NEB(
            [transition_state_atoms.copy() for i in range(5)]
            + [final_state_atoms.copy()],
        )
        initial_to_transition.interpolate()
        transition_to_final.interpolate()

        write_list = []
        for atoms in initial_to_transition.images:
            write_list.append(atoms)
        for atoms in transition_to_final.images:
            write_list.append(atoms)

        ase_io.write(
            "output/visualise_states/{}_{}.xyz".format(rxn_number, reaction_name),
            write_list,
        )
