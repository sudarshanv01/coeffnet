import time

import json

import pickle

from minimal_basis.predata.predata_qchem import TaskdocsToData

from instance_mongodb import instance_mongodb_sei


if __name__ == "__main__":
    """Generate a list of dicts that contain data for the Grambow-Green dataset."""

    db = instance_mongodb_sei(project="mlts")
    collection = db.grambow_green_calculation

    tastdocs_to_data = TaskdocsToData(
        collection=collection,
        filter_collection={"task_label": "single point"},
        identifier="rxn_number",
        state_identifier="state",
        reactant_tag="reactant",
        product_tag="product",
        transition_state_tag="transition_state",
    )
    data = tastdocs_to_data.get_data(debug=True)
