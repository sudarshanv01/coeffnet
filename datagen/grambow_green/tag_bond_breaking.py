from pathlib import Path

import argparse

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from tqdm import tqdm

from instance_mongodb import instance_mongodb_sei


def identify_number_broken_bonds(
    molecule_graph1: MoleculeGraph, molecule_graph2: MoleculeGraph
):
    """Identify the number of bonds broken in the reaction."""
    return len(molecule_graph1.graph.edges) - len(molecule_graph2.graph.edges)


def get_unique_identifiers(collection, query: dict = None):
    """Get the unique identifiers in the collection."""
    return collection.distinct("tags.rxn_number", query)


def get_all_single_point_calculation(
    collection,
    rxn_number: str,
    state="reactant",
    basis_set: str = "def2-svp",
):
    """Get all single point calculations for a given reaction number."""
    return collection.find(
        {
            "tags.rxn_number": rxn_number,
            "task_label": "single point",
            "tags.state": state,
            "orig.rem.basis": basis_set,
            "tags.number_of_bond_changes": {"$exists": False},
        },
        {
            "output.initial_molecule": 1,
            "tags.state": 1,
        },
    )


def get_command_line_arguments():

    parser = argparse.ArgumentParser(
        description="Tag calculations in the `grambow_green_calculation` collection "
        "with the number of bonds breaking in the reaction."
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run the script.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Tag calculations in the `grambow_green_calculation` collection
    with the number of bonds breaking in the reaction."""

    args = get_command_line_arguments()

    db = instance_mongodb_sei(project="mlts")
    collection = db["grambow_green_calculation"]

    rxn_numbers = get_unique_identifiers(
        collection,
        query={
            "orig.rem.basis": "def2-svp",
            "tags.number_of_bond_changes": {"$exists": False},
        },
    )

    for rxn_number in tqdm(rxn_numbers):

        state_documents = get_all_single_point_calculation(
            collection,
            rxn_number,
            basis_set="def2-svp",
            state={"$in": ["reactant", "product", "transition_state"]},
        )

        state_documents = list(state_documents)

        reactant_document = list(
            filter(lambda x: x["tags"]["state"] == "reactant", state_documents)
        )
        product_document = list(
            filter(lambda x: x["tags"]["state"] == "product", state_documents)
        )
        transition_state_document = list(
            filter(lambda x: x["tags"]["state"] == "transition_state", state_documents)
        )

        if len(reactant_document) == 0 or len(product_document) == 0:
            print(f"Skipping rxn_number = {rxn_number}")
            continue
        if len(reactant_document) > 1 or len(product_document) > 1:
            print(f"Skipping rxn_number = {rxn_number}")
            continue

        reactant_document = reactant_document[0]
        product_document = product_document[0]

        reactant_molecule = reactant_document["output"]["initial_molecule"]
        reactant_molecule = Molecule.from_dict(reactant_molecule)
        product_molecule = product_document["output"]["initial_molecule"]
        product_molecule = Molecule.from_dict(product_molecule)

        reactant_molecule_graph = MoleculeGraph.with_local_env_strategy(
            reactant_molecule, OpenBabelNN()
        )
        product_molecule_graph = MoleculeGraph.with_local_env_strategy(
            product_molecule, OpenBabelNN()
        )

        number_broken_bonds = identify_number_broken_bonds(
            product_molecule_graph, reactant_molecule_graph
        )

        if not args.dryrun:
            collection.update_one(
                {"_id": reactant_document["_id"]},
                {"$set": {"tags.number_of_bond_changes": number_broken_bonds}},
            )
            collection.update_one(
                {"_id": product_document["_id"]},
                {"$set": {"tags.number_of_bond_changes": number_broken_bonds}},
            )
            collection.update_one(
                {"_id": transition_state_document[0]["_id"]},
                {"$set": {"tags.number_of_bond_changes": number_broken_bonds}},
            )
