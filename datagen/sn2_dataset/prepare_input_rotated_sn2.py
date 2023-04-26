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

from minimal_basis.predata.predata_qchem import BaseQuantitiesQChem

import plotly.express as px

from instance_mongodb import instance_mongodb_sei


if __name__ == "__main__":
    """Visualise the matrices through a reaction."""

    db = instance_mongodb_sei(project="mlts")

    collections_data = db.minimal_basis

    # Create a new collection to store the data in
    collections_data_new = db.rotated_sn2_reaction
    groupname = "rotated_sn2_reaction"

    # Find all unique reaction labels
    reaction_labels = collections_data.distinct("tags.idx")

    for reaction_label in reaction_labels:

        # Generate a cursor that sorts the documents by `tags.type_structure`
        # first the `initial_state`, then the `transition_state` and finally
        # the `final_state`
        cursor = collections_data.find(
            {
                "tags.idx": reaction_label,
                "tags.group": groupname,
            }
        )

        # Store all the fock matrices, eigenvalues and coeff_matrices
        # for the reaction in a list
        fock_matrices = []
        eigenvalues = []
        coeff_matrices = []
        overlap_matrices = []
        final_energy = []
        state = []
        structures = []

        # Get the alpha and beta fock matrices, the alpha and beta
        # coefficient matrices and the eigenvalues. Using these quantities
        # determine the overlap matrix.
        for document in cursor:

            # Get the structure
            structure = document["output"]["initial_molecule"]
            structures.append(structure)

            # Get the alpha and beta fock matrices
            alpha_fock_matrix = document["calcs_reversed"][0]["alpha_fock_matrix"]
            if "beta_fock_matrix" in document["calcs_reversed"][0]:
                beta_fock_matrix = document["calcs_reversed"][0]["beta_fock_matrix"]
            else:
                beta_fock_matrix = alpha_fock_matrix

            # Get the alpha and beta coefficient matrices
            alpha_coeff_matrix = document["calcs_reversed"][0]["alpha_coeff_matrix"]
            if "beta_coeff_matrix" in document["calcs_reversed"][0]:
                beta_coeff_matrix = document["calcs_reversed"][0]["beta_coeff_matrix"]
            else:
                beta_coeff_matrix = alpha_coeff_matrix

            # Get the eigenvalues
            alpha_eigenvalues = document["calcs_reversed"][0]["alpha_eigenvalues"]
            if "beta_eigenvalues" in document["calcs_reversed"][0]:
                beta_eigenvalues = document["calcs_reversed"][0]["beta_eigenvalues"]
            else:
                beta_eigenvalues = alpha_eigenvalues

            alpha_base_quantities = BaseQuantitiesQChem(
                fock_matrix=alpha_fock_matrix,
                coeff_matrix=alpha_coeff_matrix,
                eigenvalues=alpha_eigenvalues,
            )
            alpha_ortho_coeff_matrix = alpha_base_quantities.get_ortho_coeff_matrix()
            alpha_overlap_matrix = alpha_base_quantities.get_overlap_matrix()

            if "beta_fock_matrix" in document["calcs_reversed"][0]:
                beta_base_quantities = BaseQuantitiesQChem(
                    fock_matrix=beta_fock_matrix,
                    coeff_matrix=beta_coeff_matrix,
                    eigenvalues=beta_eigenvalues,
                )
                beta_ortho_coeff_matrix = beta_base_quantities.get_ortho_coeff_matrix()
                beta_overlap_matrix = beta_base_quantities.get_overlap_matrix()
            else:
                beta_ortho_coeff_matrix = alpha_ortho_coeff_matrix
                beta_overlap_matrix = alpha_overlap_matrix

            # Store the information in the dataframe
            fock_matrices.append([alpha_fock_matrix, beta_fock_matrix])
            eigenvalues.append([alpha_eigenvalues, beta_eigenvalues])
            coeff_matrices.append([alpha_ortho_coeff_matrix, beta_ortho_coeff_matrix])
            overlap_matrices.append([alpha_overlap_matrix, beta_overlap_matrix])

            final_energy.append(document["output"]["final_energy"])

            state.append(document["tags"]["type_structure"])

        # Make all the quantities a numpy array before performing operations on them
        fock_matrices = np.array(fock_matrices)
        eigenvalues = np.array(eigenvalues)
        coeff_matrices = np.array(coeff_matrices)
        overlap_matrices = np.array(overlap_matrices)
        final_energy = np.array(final_energy)

        if len(final_energy) != 3:
            print(f"Reaction {reaction_label} does not have 3 states")
            continue

        data_to_store = {}
        data_to_store = {
            "fock_matrices": fock_matrices.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "overlap_matrices": overlap_matrices.tolist(),
            "coeff_matrices": coeff_matrices.tolist(),
            "state": state,
            "final_energy": final_energy.tolist(),
            "structures": structures,
            "angles": document["tags"]["angles"],
        }

        if collections_data_new.count_documents({"tags.label": reaction_label}) == 0:
            print("Storing data...")
            collections_data_new.insert_one(data_to_store)
