import logging
import argparse

from bson.objectid import ObjectId

import numpy as np

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW, ForceFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

lp = LaunchPad.from_file("/Users/sudarshanvijay/fw_config/my_launchpad_mlts.yaml")


def get_cli():
    args = argparse.ArgumentParser()
    args.add_argument("--dryrun", action="store_true", default=False)
    args.add_argument("--basis", type=str, default="6-31g*")
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.rotated_water_calculations
    initial_structure_collection = db.rotated_water_initial_structures

    params = {
        "dft_rung": 1,
        "overwrite_inputs": {
            "rem": {
                "basis": args.basis,
                "purecart": "1111",
            }
        },
    }

    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    count_structures = 0

    for document in initial_structure_collection.find({}):

        tags = {
            "euler_angles": document["euler_angles"],
            "idx": document["idx"],
            "inverted_coordinates": True,
            "basis_are_spherical": True,
        }

        molecule = Molecule.from_dict(document["molecule"])

        # Invert the coordinates of the molecule.
        coordinates = np.array(molecule.cart_coords)
        # Create permutted coordinates
        _coordinates = coordinates.copy()
        _coordinates[:, [0, 1, 2]] = _coordinates[:, [2, 0, 1]]

        # Create a new molecule with the inverted coordinates.
        molecule = Molecule(
            species=molecule.species,
            coords=_coordinates,
            charge=molecule.charge,
            spin_multiplicity=molecule.spin_multiplicity,
        )

        count_structures += 1

        firew = SinglePointFW(
            molecule=molecule,
            qchem_input_params=params,
            extra_scf_print=True,
            db_file=">>db_file<<",
        )

        wf = Workflow([firew])
        wf = add_tags(wf, tags)

        if not args.dryrun:
            lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
