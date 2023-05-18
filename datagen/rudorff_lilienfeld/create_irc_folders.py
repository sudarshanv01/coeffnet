from typing import List

from monty.serialization import loadfn, dumpfn

import logging

import os

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.qchem.inputs import QCInput

import ase.io as ase_io
from ase.neb import NEB


from instance_mongodb import instance_mongodb_sei

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    """Visualize the initial, transition and final states of the reaction."""

    db = instance_mongodb_sei(project="mlts")
    collection = db.rudorff_lilienfeld_data

    rem_frequency = {
        "job_type": "freq",
        "basis": "6-31+g*",
        "max_scf_cycles": 100,
        "gen_scfman": True,
        "xc_grid": 3,
        "thresh": 14,
        "s2thresh": 16,
        "scf_algorithm": "diis",
        "resp_charges": False,
        "symmetry": False,
        "sym_ignore": True,
        "method": "b3lyp",
        "unrestricted": True,
    }

    rem_irc_positive = {
        "job_type": "rpath",
        "basis": "6-31+g*",
        "max_scf_cycles": 100,
        "gen_scfman": True,
        "xc_grid": 3,
        "thresh": 14,
        "s2thresh": 16,
        "scf_algorithm": "diis",
        "resp_charges": False,
        "symmetry": False,
        "sym_ignore": True,
        "method": "b3lyp",
        "unrestricted": True,
        "rpath_direction": 1,
    }

    rem_irc_negative = {
        "job_type": "rpath",
        "basis": "6-31+g*",
        "max_scf_cycles": 100,
        "gen_scfman": True,
        "xc_grid": 3,
        "thresh": 14,
        "s2thresh": 16,
        "scf_algorithm": "diis",
        "resp_charges": False,
        "symmetry": False,
        "sym_ignore": True,
        "method": "b3lyp",
        "unrestricted": True,
        "rpath_direction": -1,
    }

    for doc in collection.find({}):
        transition_state_molecule = doc["transition_state_molecule"]
        transition_state_molecule = Molecule.from_dict(transition_state_molecule)

        folder_name_positive = (
            f"irc_calculations/{doc['reaction_name']}/{doc['rxn_number']}/positive"
        )
        folder_name_negative = (
            f"irc_calculations/{doc['reaction_name']}/{doc['rxn_number']}/negative"
        )
        if not os.path.exists(folder_name_positive):
            os.makedirs(folder_name_positive)
        if not os.path.exists(folder_name_negative):
            os.makedirs(folder_name_negative)

        tags = {
            "rxn_number": doc["rxn_number"],
            "reaction_name": doc["reaction_name"],
        }

        dumpfn(tags, f"{folder_name_positive}/tags.json")
        dumpfn(tags, f"{folder_name_negative}/tags.json")

        qcinput_frequency = QCInput(
            molecule=transition_state_molecule,
            rem=rem_frequency,
        )

        qcinput_irc_positive = QCInput(
            molecule="read",
            rem=rem_irc_positive,
        )

        qcinput_irc_negative = QCInput(
            molecule="read",
            rem=rem_irc_negative,
        )

        qcinput_frequency.write_multi_job_file(
            job_list=[qcinput_frequency, qcinput_irc_positive],
            filename=f"{folder_name_positive}/input.in",
        )

        qcinput_frequency.write_multi_job_file(
            job_list=[qcinput_frequency, qcinput_irc_negative],
            filename=f"{folder_name_negative}/input.in",
        )
