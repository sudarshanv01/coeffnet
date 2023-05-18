import logging

from typing import List, Dict, Any

import os

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.ase import AseAtomsAdaptor

from ase import io as ase_io

from instance_mongodb import instance_mongodb_sei


def expected_endstate(endstate_molecule: Molecule, initial_molecule: Molecule) -> bool:
    """Check if the endstate is the one we are looking for."""

    initial_molecule_graph = MoleculeGraph.with_local_env_strategy(
        initial_molecule, OpenBabelNN()
    )
    molecule_graph = MoleculeGraph.with_local_env_strategy(
        endstate_molecule, OpenBabelNN()
    )

    if initial_molecule_graph.isomorphic_to(molecule_graph):
        return True
    else:
        return False


if __name__ == "__main__":
    """Looking at the calculation, store endstate data."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    os.makedirs("output/visualise_endstates", exist_ok=True)

    db = instance_mongodb_sei(project="mlts")
    data_collection = db.rudorff_lilienfeld_data
    calculation_collection = db.rudorff_lilienfeld_calculation

    cursor = calculation_collection.find(
        {"task_label": "structure optimization"},
        {
            "calcs_reversed": 1,
            "tags": 1,
            "output": 1,
            "structure_change": 1,
        },
    )

    for doc in cursor:
        state = doc["tags"]["state"]
        rxn_number = doc["tags"]["rxn_number"]
        reaction_name = doc["tags"]["reaction_name"]

        molecule = doc["output"]["optimized_molecule"]
        molecule = Molecule.from_dict(molecule)

        final_energy = doc["output"]["final_energy"]

        initial_molecule = doc["calcs_reversed"][-1]["initial_molecule"]
        initial_molecule = Molecule.from_dict(initial_molecule)

        # if not expected_endstate(molecule, initial_molecule):
        #     logger.info(
        #         "Not expected endstate for {} {} {}".format(
        #             state, rxn_number, reaction_name
        #         )
        #     )
        #     continue
        # if doc["structure_change"][0] != "no_change":
        #     logger.info(
        #         "Structure change for {} {} {}".format(state, rxn_number, reaction_name)
        #     )
        #     continue

        ase_atoms = [AseAtomsAdaptor.get_atoms(molecule)]
        ase_atoms += [AseAtomsAdaptor.get_atoms(initial_molecule)]

        ase_io.write(
            "output/visualise_endstates/{}_{}_{}.xyz".format(
                state, rxn_number, reaction_name
            ),
            ase_atoms,
        )

        data_collection.update_one(
            {"rxn_number": rxn_number, "reaction_name": reaction_name},
            {
                "$set": {
                    f"{state}_molecule": molecule.as_dict(),
                    f"{state}_energy": final_energy,
                }
            },
            upsert=True,
        )
        logger.info("Updated {} {} {}".format(state, rxn_number, reaction_name))
