import glob

from ase import io as ase_io

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.qchem.outputs import QCOutput

from instance_mongodb import instance_mongodb_sei

if __name__ == "__main__":
    """Store all data in representative xyz files for easy visualisation."""

    db = instance_mongodb_sei("mlts")
    collection = db.grambow_green_initial_structures

    rxn_folders = glob.glob("data/b97d3/rxn*")

    for rxn_folder in rxn_folders:
        rxn_number = rxn_folder.split("/")[-1].replace("rxn", "")

        molecules = []
        for state in ["r", "ts", "p"]:
            try:
                qc_output = QCOutput.multiple_outputs_from_file(
                    QCOutput,
                    f"{rxn_folder}/{state}{rxn_number}.log",
                    keep_sub_files=False,
                )
                if state in ["r", "ts"]:
                    idx = 1
                elif state == "p":
                    idx = 0
                atoms = qc_output[idx].data["molecule_from_optimized_geometry"]

                # ase_atoms = AseAtomsAdaptor.get_atoms(atoms)
                molecules.append(atoms)
            except (KeyError, IndexError):
                print(f"No {state} for {rxn_number}")
                continue

        if len(molecules) == 3:
            ase_molecules = [
                AseAtomsAdaptor.get_atoms(molecule) for molecule in molecules
            ]
            ase_io.write(f"data/b97d3/xyz/{rxn_number}.xyz", ase_molecules)

            collection.insert_one(
                {
                    "rxn_number": rxn_number,
                    "reactant": molecules[0].as_dict(),
                    "transition_state": molecules[1].as_dict(),
                    "product": molecules[2].as_dict(),
                    "functional": "b97d3",
                }
            )

            print(f"Stored xyz file and data in MongoDB for {rxn_number}")
