import logging

from typing import List

import copy

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.ase import AseAtomsAdaptor

from instance_mongodb import instance_mongodb_sei

from minimal_basis.predata.predata_qchem import BaseQuantitiesQChem

MATCHER = {
    "X": {
        "A": "F",
        "B": "Cl",
        "C": "Br",
    },
    "Y": {
        "A": "H",
        "B": "F",
        "C": "Cl",
        "D": "Br",
    },
}


def perturb_along_eigenmode(
    ts_molecule: Molecule, eigenmode: List[float], scaling: float
) -> Molecule:
    """Perturn the molecule along the eigen modes based on a scaling factor.

    Args:
        ts_molecule: The transition state molecule.
        eigenmode: The eigenmode.
        scaling: The scaling factor to perturb the molecule.
    """

    def validate_eigenmode(eigenmode: List[float]) -> None:
        """Check if the eigenmode is normalised correctly."""
        norm_eigemode = np.linalg.norm(eigenmode)
        is_close = np.isclose(norm_eigemode, 1.0, atol=1e-3)
        if not is_close:
            raise ValueError("The eigenmode is not normalised correctly.")

    eigenmode = np.array(eigenmode)
    validate_eigenmode(eigenmode)
    assert eigenmode.shape == (
        len(ts_molecule),
        3,
    ), "Eigenmode is not the correct shape."

    delta_pos = scaling * eigenmode
    perturbed_molecule_pos = copy.deepcopy(ts_molecule)

    positions = [a.coords for a in ts_molecule.sites]
    positions = np.array(positions)

    perturbed_pos = positions + delta_pos

    for i, a in enumerate(perturbed_molecule_pos.sites):
        a.coords = perturbed_pos[i]

    return perturbed_molecule_pos


def expected_transition_state(
    ts_molecule: Molecule,
    ts_molecule_graph: MoleculeGraph,
    eigenmode: List[float],
    eigenvalue: List[float],
    idx_carbon_node: int,
    label: str,
) -> Molecule:
    """Based on the rules outlined in this function, decide if the transition state
    is the one we are looking for.

    Args:
        ts_molecule: The transition state molecule.
        ts_molecule_graph: The transition state molecule graph.
        eigenmode: The eigenmode.
        eigenvalue: The eigenvalue.
        idx_carbon_node: The index of the carbon node in the molecule graph.
        label: The label of the transition state.
    """
    R1, R2, R3, R4, X, Y = label.split("_")

    imag_eigval_idx = np.where(eigenvalue < 0)
    if len(imag_eigval_idx[0]) != 1:
        tags_failure += "+imag_freq"
        return False

    coordination_carbon = ts_molecule_graph.get_coordination_of_site(idx_carbon_node)
    disconnected_fragments = ts_molecule_graph.get_disconnected_fragments()

    if MATCHER["X"][X] != MATCHER["Y"][Y]:
        idx_X = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["X"][X]
        ]
        idx_Y = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["Y"][Y]
        ]

        assert len(idx_X) == 1, "More than one X atom."
        idx_X = idx_X[0]

        dists = [
            ts_molecule.get_distance(idx_carbon_node, idx_Y[i])
            for i in range(len(idx_Y))
        ]
        idx_Y = idx_Y[np.argmin(dists)]

    else:
        idx_X = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["X"][X]
        ]
        idx_Y = [
            i for i, a in enumerate(ts_molecule) if a.specie.symbol == MATCHER["Y"][Y]
        ]
        idx_X = idx_X[0]
        idx_Y = idx_Y[1]

    perturbed_molecule_pos = perturb_along_eigenmode(ts_molecule, eigenmode, 0.5)
    perturbed_molecule_neg = perturb_along_eigenmode(ts_molecule, eigenmode, -0.5)

    dist_pos = perturbed_molecule_pos.get_distance(
        idx_carbon_node, idx_X
    ) - perturbed_molecule_pos.get_distance(idx_carbon_node, idx_Y)
    dist_neg = perturbed_molecule_neg.get_distance(
        idx_carbon_node, idx_X
    ) - perturbed_molecule_neg.get_distance(idx_carbon_node, idx_Y)

    if np.sign(dist_pos) == np.sign(dist_neg):
        return False

    return True


def expected_interpolation(energies, states):
    """Check if the interpolated energies are in the right order."""

    if len(energies) != 3:
        assert len(energies) == 3, "The number of energies is not 3."

    idx_initial_state = states.index("initial_state")
    idx_final_state = states.index("final_state")
    idx_transition_state = states.index("transition_state")

    is_to_ts_check = energies[idx_transition_state] > energies[idx_initial_state]
    fs_to_ts_check = energies[idx_transition_state] > energies[idx_final_state]

    if is_to_ts_check and fs_to_ts_check:
        return True
    else:
        return False


if __name__ == "__main__":
    """Generate a dataset based on the interpolation of SN2 reactions."""

    db = instance_mongodb_sei(project="mlts")

    collection_data = db.minimal_basis
    collection_data_new = db.sn2_reaction_dataset

    GROUPNAME_TS = "sn2_transition_states"
    GROUPNAME_INTERP = "sn2_interpolated_from_transition_states"
    find_TS_keys = {
        "tags.group": GROUPNAME_TS,
        "orig.rem.basis": "6-31g*",
        "orig.rem.method": "b3lyp",
    }
    idx_carbon_node = 0

    num_accepted_ts = 0
    for doc in collection_data.find(find_TS_keys):

        tags_failure = ""

        molecule_dict = doc["output"]["optimized_molecule"]
        molecule = Molecule.from_dict(molecule_dict)
        atoms = AseAtomsAdaptor.get_atoms(molecule)
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        connected_fragments = molecule_graph.get_connected_sites(idx_carbon_node)

        eigenvectors = doc["output"]["frequency_modes"]
        eigenvectors = np.array(eigenvectors)
        eigenvalues = doc["output"]["frequencies"]
        eigenvalues = np.array(eigenvalues)

        label = doc["tags"]["label"]

        if np.all(eigenvalues > 0):
            continue

        imag_freq_idx = np.where(eigenvalues < 0)[0][0]
        imag_freq = eigenvalues[imag_freq_idx]
        imag_eigenmode = eigenvectors[imag_freq_idx]

        if not expected_transition_state(
            molecule,
            molecule_graph,
            imag_eigenmode,
            imag_freq,
            idx_carbon_node,
            label,
        ):
            tags_failure += "+ts_failure"
            print("Transition state is not accepted, storing for training.")

        cursor = (
            collection_data.find(
                {
                    "tags.label": label,
                    "tags.group": GROUPNAME_INTERP,
                    "tags.scaling": {"$in": [-0.5, 0.0, 0.5]},
                }
            )
            .sort("tags.scaling", 1)
            .limit(3)
        )

        final_energies = []
        interp_structures = []
        coeff_matrices = []
        ortho_coeff_matrices = []
        orthogonalisation_matrices = []
        state = []
        interp_eigenvalues = []

        for document in cursor:

            structure = document["output"]["initial_molecule"]
            interp_structures.append(structure)
            final_energies.append(document["output"]["final_energy"])

            if document["tags"]["scaling"] == -0.5:
                state.append("initial_state")
            elif document["tags"]["scaling"] == 0.0:
                state.append("transition_state")
            elif document["tags"]["scaling"] == 0.5:
                state.append("final_state")
            else:
                raise ValueError("Scaling value not recognised")

            alpha_fock_matrix = document["calcs_reversed"][0]["alpha_fock_matrix"]
            beta_fock_matrix = document["calcs_reversed"][0]["beta_fock_matrix"]

            alpha_fock_matrix = np.array(alpha_fock_matrix)
            beta_fock_matrix = np.array(beta_fock_matrix)

            alpha_coeff_matrix = document["calcs_reversed"][0]["alpha_coeff_matrix"]
            beta_coeff_matrix = document["calcs_reversed"][0]["beta_coeff_matrix"]

            alpha_coeff_matrix = np.array(alpha_coeff_matrix)
            beta_coeff_matrix = np.array(beta_coeff_matrix)

            alpha_eigenvalues = document["calcs_reversed"][0]["alpha_eigenvalues"]
            beta_eigenvalues = document["calcs_reversed"][0]["beta_eigenvalues"]

            alpha_eigenvalues = np.array(alpha_eigenvalues)
            beta_eigenvalues = np.array(beta_eigenvalues)

            interp_eigenvalues.append([alpha_eigenvalues, beta_eigenvalues])

            alpha_base_quantities = BaseQuantitiesQChem(
                fock_matrix=alpha_fock_matrix,
                coeff_matrix=alpha_coeff_matrix,
                eigenvalues=alpha_eigenvalues,
            )

            beta_base_quantities = BaseQuantitiesQChem(
                fock_matrix=beta_fock_matrix,
                coeff_matrix=beta_coeff_matrix,
                eigenvalues=beta_eigenvalues,
            )

            alpha_ortho_coeff_matrix = alpha_base_quantities.get_ortho_coeff_matrix()
            beta_ortho_coeff_matrix = beta_base_quantities.get_ortho_coeff_matrix()

            ortho_coeff_matrices.append(
                [alpha_ortho_coeff_matrix, beta_ortho_coeff_matrix]
            )

            orthogonalisation_matrices.append(
                [
                    alpha_base_quantities.orthogonalisation_matrix,
                    beta_base_quantities.orthogonalisation_matrix,
                ],
            )

        ortho_coeff_matrices = np.array(ortho_coeff_matrices)
        interp_eigenvalues = np.array(interp_eigenvalues)
        orthogonalisation_matrices = np.array(orthogonalisation_matrices)

        try:
            if not expected_interpolation(final_energies, state):
                print("Interpolation is not accepted, storing for training.")
                tags_failure += "+interp_failure"
        except AssertionError:
            print("Incorrect dimension of data.")
            continue

        num_accepted_ts += 1
        print("Accepted TS: ", num_accepted_ts)
        print("Failure: ", tags_failure)

        data_to_store = {}
        data_to_store = {
            "eigenvalues": interp_eigenvalues.tolist(),
            "state": state,
            "coeff_matrices": ortho_coeff_matrices.tolist(),
            "structures": interp_structures,
            "final_energy": final_energies,
            "tags": {
                "label": label,
                "failure": tags_failure,
            },
        }

        if collection_data_new.count_documents({"tags.label": label}) == 0:
            collection_data_new.insert_one(data_to_store)
