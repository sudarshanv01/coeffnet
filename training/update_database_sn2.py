from typing import List, Dict, Tuple
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
from numpy import typing as npt

import matplotlib.pyplot as plt

import pandas as pd

from ase import io as ase_io

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import plotly.express as px

from instance_mongodb import instance_mongodb_sei

if __name__ == "__main__":
    """Visualise the matrices through a reaction."""

    db = instance_mongodb_sei(project="mlts")

    collections_data = db.minimal_basis

    collections_data_new = db.interp_sn2_calculation
    groupname = "sn2_interpolated_from_transition_states"

    reaction_labels = collections_data.distinct("tags.label")
    print(reaction_labels)

    for reaction_label in reaction_labels:
        print(reaction_label)
        cursor = (
            collections_data.find(
                {
                    "tags.label": reaction_label,
                    "tags.group": groupname,
                    "tags.scaling": {"$in": [-0.5, 0.0, 0.5]},
                }
            )
            .sort("tags.scaling", 1)
            .limit(3)
        )

        for document in cursor:
            collections_data_new.insert_one(document)
