import logging
import yaml
import copy
import argparse

from bson.objectid import ObjectId

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import SinglePointFW, ForceFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

lp = LaunchPad.from_file("/global/u1/s/svijay/fw_config/my_launchpad_mlts.yaml")


def get_cli():
    args = argparse.ArgumentParser()
    args.add_argument("--dryrun", action="store_true", default=False)
    return args.parse_args()


if __name__ == "__main__":
    """Run calculations for perturbed structures."""

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # The sei database, where all info is stored
    db = instance_mongodb_sei(project="mlts")

    # Get the command line arguments
    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    INITIAL_STRUCTURE_COLLECTION_NAME = "minimal_basis_initial_structures"
    COLLECTION_NAME = "minimal_basis"
    GROUPNAME_STRUCTURES = "sn2_interpolated_from_transition_states"

    collection = db[COLLECTION_NAME]
    initial_structure_collection = db[INITIAL_STRUCTURE_COLLECTION_NAME]

    # Parameters to import
    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)
    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    count_structures = 0
    for molecule_info in initial_structure_collection.find(
        {"tags.group": GROUPNAME_STRUCTURES, 
        "tags.scaling": {"$in": [-0.5, 0.0, 0.5]},
        },
        # no_cursor_timeout=True,
    ):

        # Get the tags of the molecule
        tags = copy.deepcopy(molecule_info["tags"])
        tags["quantity"] = "forces"
        print(tags)

        # If the collection already includes this `tags` then skip
        if collection.count_documents({"tags": tags}) > 0:
            logger.info(f"Skipping {tags['label']}")
            continue

        # Get the endstate molecules and the transition state
        molecule = molecule_info["structure"]
        # Convert dict to a Molecule
        molecule = Molecule.from_dict(molecule)

        count_structures += 1

        # Run the simplest workflow available
        firew = ForceFW(
            molecule=molecule,
            qchem_input_params=params,
            # extra_scf_print=True,
            db_file=">>db_file<<",
            name=tags["label"],
            # spec={"_dupefinder": DupeFinderExact()},
        )

        # Create a workflow for just one simple firework
        wf = Workflow([firew], name=tags["label"])

        # Label the set appropriately
        wf = add_tags(wf, tags)

        if not args.dryrun:

            # Add set to launchpad
            lp.add_wf(wf)

    logger.info(f"{count_structures} structures processed for calculation.")
