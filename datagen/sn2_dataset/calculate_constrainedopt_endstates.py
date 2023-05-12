import logging
import yaml
import copy
import argparse

from bson.objectid import ObjectId

from pymatgen.core import Molecule

from atomate.qchem.fireworks.core import OptimizeFW
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

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    db = instance_mongodb_sei(project="mlts")

    args = get_cli()
    logger.info("Command line arguments: {}".format(args))

    collection = db.rudorff_lilienfeld_calculation
    initial_structure_collection = db.rudorff_lilienfeld_initial_structures
    find_tags = {}

    with open("config/reproduce_paper_parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    count_structures = 0
    for document in initial_structure_collection.find(find_tags):

        for _state in ["perturbed_molecule_0", "perturbed_molecule_-0"]:

            if _state not in document:
                continue

            molecule_dict = document[_state]
            for perturbation, molecule in molecule_dict.items():
                state = _state + "_" + perturbation
                molecule = Molecule.from_dict(molecule)
                tags = {
                    "state": state,
                    "quantities": ["optimization"],
                    "rxn_number": document["rxn_number"],
                    "reaction_name": document["reaction_name"],
                }

                _params = copy.deepcopy(params)
                _params["overwrite_inputs"]["opt"] = {}
                _params["overwrite_inputs"]["opt"]["FIXED"] = []
                for i, site in enumerate(molecule):
                    _fixed_string = f"{i+1} XYZ"
                    if document["reaction_name"] == "sn2":
                        if i == 0 or i == 4:
                            continue
                    elif document["reaction_name"] == "e2":
                        if i == 0 or i == 1:
                            continue
                    _params["overwrite_inputs"]["opt"]["FIXED"].append(_fixed_string)

                print(_params)

                if collection.count_documents({"tags": tags}) > 0:
                    logger.info(f"Skipping {tags}")
                    continue
                else:
                    logger.info(f"Processing {tags}")

                count_structures += 1

                firew = OptimizeFW(
                    molecule=molecule,
                    qchem_input_params=_params,
                    db_file=">>db_file<<",
                )

                wf = Workflow([firew])
                wf = add_tags(wf, tags)

                if not args.dryrun:
                    lp.add_wf(wf)

        logger.info(f"{count_structures} structures processed for calculation.")
