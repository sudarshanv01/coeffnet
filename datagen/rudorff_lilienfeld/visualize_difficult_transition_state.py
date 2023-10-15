from pathlib import Path

import numpy as np

from pymatgen.core.structure import Molecule

from instance_mongodb import instance_mongodb_sei


def generate_potential_energy_scan(
        molecule: Molecule,
        index_fixed: int,
        index_move: int,
        distance: float,
):
    """Generate a potential energy scan along a bond."""

    positions = [a.coords for a in molecule.sites]
    positions = np.array(positions)

    initial_distance = np.linalg.norm(positions[index_fixed] - positions[index_move])
    distance = initial_distance + distance

    vector = positions[index_move] - positions[index_fixed]
    vector = vector / np.linalg.norm(vector)

    new_positions = positions.copy()
    new_positions[index_move] = positions[index_fixed] + distance * vector

    new_molecule = Molecule(
        molecule.species,
        new_positions,
        charge=molecule.charge,
        spin_multiplicity=molecule.spin_multiplicity,
    )

    # Set new positions
    for i, a in enumerate(new_molecule.sites):
        a.coords = new_positions[i]
    
    return new_molecule

if __name__ == "__main__":
    """Get the structure of a difficult to compute transition state."""

    __output_folder__ = Path("output") / "trajectory"
    __output_folder__.mkdir(exist_ok=True)

    db = instance_mongodb_sei(project="mlts")
    data_collection = db.rudorff_lilienfeld_data
    calculation_collection = db.rudorff_lilienfeld_calculation
    collection_structures = db.visualize_difficult_transition_state

    grid_spacing = 30
    scalings = np.linspace(-1, 2, grid_spacing)

    doc_ffopt = calculation_collection.find(
        {
            "task_label": "frequency flattening transition state optimization",
        },
    )

    for doc in doc_ffopt:
        if len(doc['opt_trajectory']) > 200:
            opt_trajectory = doc['opt_trajectory']
            opt_trajectory = [Molecule.from_dict(mol['molecule']) for mol in opt_trajectory]
            for i, mol in enumerate(opt_trajectory):
                filename = __output_folder__ / f"{doc['tags']['reaction_name']}_ts_{i}.xyz"
                mol.to(str(filename))
            
            ts_molecule = opt_trajectory[-1]

            for idx_scaling_1, scaling_1 in enumerate(scalings):

                for idx_scaling_2, scaling_2 in enumerate(scalings):

                    new_molecule = generate_potential_energy_scan(
                        ts_molecule,
                        index_fixed=0,
                        index_move=2,
                        distance=scaling_1,
                    )
                    new_molecule = generate_potential_energy_scan(
                        new_molecule,
                        index_fixed=0,
                        index_move=3,
                        distance=scaling_2,
                    )

                    filename = __output_folder__ / f"scaling_{scaling_1}_scaling_{scaling_2}.xyz"
                    new_molecule.to(str(filename))

                    data_to_store = {
                        "reaction_name": doc['tags']['reaction_name'],
                        "scaling": [scaling_1, scaling_2],
                        "molecule": new_molecule.as_dict(),
                    }

                    collection_structures.insert_one(data_to_store)

            break