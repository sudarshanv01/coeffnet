from pathlib import Path

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

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
    collection, rxn_number, state="reactant", basis_set: str = "def2-svp"
):
    """Get all single point calculations for a given reaction number."""
    return collection.find(
        {
            "tags.rxn_number": rxn_number,
            "task_label": "single point",
            "tags.state": state,
            "orig.rem.basis": basis_set,
        },
        {
            "output.initial_molecule": 1,
        },
    )


if __name__ == "__main__":
    """Tag calculations in the `grambow_green_calculation` collection
    with the number of bonds breaking in the reaction."""

    db = instance_mongodb_sei(project="mlts")
    collection = db["grambow_green_calculation"]

    rxn_numbers = get_unique_identifiers(
        collection, query={"orig.rem.basis": "def2-svp"}
    )
    print(f"Number of unique rxn_numbers = {len(rxn_numbers)}")

    for rxn_number in rxn_numbers:

        reactant_document = get_all_single_point_calculation(
            collection, rxn_number, state="reactant", basis_set="def2-svp"
        )
        product_document = get_all_single_point_calculation(
            collection, rxn_number, state="product", basis_set="def2-svp"
        )
        transition_state_document = get_all_single_point_calculation(
            collection, rxn_number, state="transition_state", basis_set="def2-svp"
        )

        reactant_document = list(reactant_document)
        product_document = list(product_document)

        if len(reactant_document) == 0:
            print(f"rxn_number = {rxn_number}, reactant_document = {reactant_document}")
            continue
        if len(product_document) == 0:
            print(f"rxn_number = {rxn_number}, product_document = {product_document}")
            continue

        assert (
            len(reactant_document) == 1
        ), f"len(reactant_document) = {len(reactant_document)}"
        assert (
            len(product_document) == 1
        ), f"len(product_document) = {len(product_document)}"

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
        print(
            f"rxn_number = {rxn_number}, number of broken bonds = {number_broken_bonds}"
        )

        __output_folder__ = Path("output") / "structures" / f"rxn_number_{rxn_number}"
        __output_folder__.mkdir(parents=True, exist_ok=True)
        reactant_xyz_string = __output_folder__ / f"reactant_{number_broken_bonds}.xyz"
        product_xyz_string = __output_folder__ / f"product_{number_broken_bonds}.xyz"
        reactant_molecule.to(reactant_xyz_string.as_posix())
        product_molecule.to(product_xyz_string.as_posix())

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
