from typing import List, Tuple, Union

import numpy.typing as npt
import numpy as np

from pymongo.cursor import Cursor

import tqdm


class BaseQuantitiesQChem:
    def __init__(
        self,
        fock_matrix: npt.ArrayLike,
        eigenvalues: npt.ArrayLike,
        coeff_matrix: npt.ArrayLike,
    ):
        """QChem provides the Fock matrix, the eigenvalues and the
        coeffient matrix. The other quantities in the current basis
        and orthogonal form are provided by the methods in this class.

        Args:
            fock_matrix (npt.ArrayLike): The Fock matrix.
            eigenvalues (npt.ArrayLike): The eigenvalues.
            coeff_matrix (npt.ArrayLike): The coefficient matrix.

        """

        self.fock_matrix = np.array(fock_matrix)
        self.eigenvalues = np.array(eigenvalues)
        self.coeff_matrix = np.array(coeff_matrix)

        self.overlap_matrix = None
        self.orthogonalisation_matrix = None
        self.orthogonal_fock_matrix = None
        self.ortho_coeff_matrix = None
        self.eigenval_ortho_fock = None

        self.diagonalised_eigen = np.zeros(
            (len(self.eigenvalues), len(self.eigenvalues))
        )
        np.fill_diagonal(self.diagonalised_eigen, self.eigenvalues)

    def get_overlap_matrix(self) -> npt.ArrayLike:
        self.generate_overlap_matrix()
        return self.overlap_matrix

    def get_orthogonalization_matrix(self) -> npt.ArrayLike:
        self.generate_orthogonalization_matrix()
        return self.orthogonalisation_matrix

    def get_orthogonal_fock_matrix(self) -> npt.ArrayLike:
        self.generate_orthogonal_fock_matrix()
        return self.orthogonal_fock_matrix

    def get_ortho_coeff_matrix(self) -> npt.ArrayLike:
        self.generate_overlap_matrix()
        self.generate_orthogonalization_matrix()
        self.generate_orthogonal_fock_matrix()
        self.generate_orthogonal_coeff_matrix()
        return self.ortho_coeff_matrix

    def get_eigenvalues_ortho_fock(self) -> npt.ArrayLike:
        self.generate_orthogonal_fock_matrix()
        self.generate_orthogonal_coeff_matrix()
        return self.eigenval_ortho_fock

    def generate_overlap_matrix(self) -> None:
        """Determine the overlap element from the Hamiltonian, the
        eigen energies and the cofficient matrix."""
        self.overlap_matrix = np.dot(
            np.dot(self.fock_matrix, self.coeff_matrix),
            np.linalg.inv(np.dot(self.coeff_matrix, self.diagonalised_eigen)),
        )

    def generate_orthogonalization_matrix(self):
        """Determine the orthogonalization matrix from the overlap matrix."""
        eigenval_overlap, eigenvec_overlap = np.linalg.eigh(self.overlap_matrix)
        self.orthogonalisation_matrix = np.dot(
            eigenvec_overlap,
            np.dot(np.diag(1 / np.sqrt(eigenval_overlap)), eigenvec_overlap.T),
        )

    def generate_orthogonal_fock_matrix(self):
        self.orthogonal_fock_matrix = np.dot(
            np.dot(self.orthogonalisation_matrix.T, self.fock_matrix),
            self.orthogonalisation_matrix,
        )

    def generate_orthogonal_coeff_matrix(self):
        """Diagonalize the Fock matrix."""
        eigenval_ortho_fock, eigenvec_ortho_fock = np.linalg.eigh(
            self.orthogonal_fock_matrix
        )
        self.ortho_coeff_matrix = eigenvec_ortho_fock
        self.eigenval_ortho_fock = eigenval_ortho_fock


class TaskdocsToData:
    def __init__(
        self,
        collection,
        filter_collection: dict = {},
        identifier: str = "rxn_number",
        state_identifier: str = "state",
        reactant_tag: str = "reactant",
        product_tag: str = "product",
        transition_state_tag: str = "transition_state",
    ):
        """Convert TaskDocuments to a List[Dict] with reaction information."""

        self.collection = collection
        self.filter_collection = filter_collection
        self.identifier = identifier
        self.state_identifier = state_identifier
        self.reactant_tag = reactant_tag
        self.product_tag = product_tag
        self.transition_state_tag = transition_state_tag

        self.data = []

    def get_all_identifiers(self) -> List[str]:
        """Get all unique identifiers from the TaskDocuments."""
        identifiers = self.collection.find(self.filter_collection).distinct(
            f"tags.{self.identifier}"
        )
        return identifiers

    def get_reaction_data(self, identifier: Union[str, float]) -> None:
        """Parse the output dataset and get the reaction data."""
        cursor = self.collection.find(
            {
                f"tags.{self.identifier}": identifier,
                f"tags.{self.state_identifier}": {
                    "$in": [
                        self.reactant_tag,
                        self.product_tag,
                        self.transition_state_tag,
                    ]
                },
            },
            {
                "output.initial_molecule": 1,
                "output.final_energy": 1,
                "calcs_reversed.alpha_eigenvalues": 1,
                "calcs_reversed.beta_eigenvalues": 1,
                "calcs_reversed.alpha_coeff_matrix": 1,
                "calcs_reversed.beta_coeff_matrix": 1,
                "calcs_reversed.alpha_fock_matrix": 1,
                "calcs_reversed.beta_fock_matrix": 1,
                f"tags.{identifier}": 1,
            },
        ).sort(f"tags.{self.state_identifier}", 1)

        data = {}
        for document in cursor:
            for key, value in document["calcs_reversed"][0].items():
                if key not in data:
                    data[key] = []
                data[key].append(value)
            for key, value in document["output"].items():
                if key not in data:
                    data[key] = []
                data[key].append(value)

        data["tags"] = {}
        data["tags"][self.identifier] = identifier
        self.data.append(data)

    def get_data(self) -> List[dict]:
        """Parse the TaskDocuments and return a list of dictionaries."""
        identifiers = self.get_all_identifiers()

        for identifier in tqdm.tqdm(identifiers):
            self.get_reaction_data(identifier)

        return self.data
