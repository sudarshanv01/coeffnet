import glob

from ase import io as ase_io

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.core.structure import Molecule

from instance_mongodb import instance_mongodb_sei

if __name__ == "__main__":
    """Store all data in representative xyz files for easy visualisation."""

    db = instance_mongodb_sei("mlts")
    collection = db.rudorff_lilienfeld_initial_structures

    for reaction in ["sn2", "e2"]:

        rxn_folders = glob.glob(
            f"input/from_QMrxn20/transition-states/{reaction}/*.xyz"
        )

        for rxn_folder in rxn_folders:

            rxn_number = rxn_folder.split("/")[-1].split(".")[0]

            molecule = Molecule.from_file(rxn_folder)
            molecule.set_charge_and_spin(charge=-1, spin_multiplicity=1)

            collection.insert_one(
                {
                    "rxn_number": rxn_number,
                    "transition_state": molecule.as_dict(),
                    "reaction_name": reaction,
                }
            )

            print(f"Stored xyz file and data in MongoDB for {rxn_number}")
