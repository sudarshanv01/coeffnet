from typing import Any, List, Tuple, Union

import numpy.typing as npt
import numpy as np

from collections import defaultdict

from pymongo.cursor import Cursor

import tqdm

import random


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
        self.generate_overlap_matrix()
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
        **kwargs: Any,
    ):
        """Convert TaskDocuments to a List[Dict] with reaction information."""

        self.collection = collection
        self.filter_collection = filter_collection
        self.identifier = identifier
        self.state_identifier = state_identifier
        self.reactant_tag = reactant_tag
        self.product_tag = product_tag
        self.transition_state_tag = transition_state_tag

        if "debug_number_of_reactions" in kwargs:
            self.debug_number_of_reactions = kwargs["debug_number_of_reactions"]
        else:
            self.debug_number_of_reactions = 100

        self.data = []

    def _get_all_identifiers(self) -> List[str]:
        """Get all unique identifiers from the TaskDocuments."""
        identifiers = self.collection.find(self.filter_collection).distinct(
            f"tags.{self.identifier}"
        )
        return identifiers

    def _get_reaction_data(self, identifier: Union[str, float]) -> None:
        """Parse the output dataset and get the reaction data."""

        find_filter = {
            f"tags.{self.identifier}": identifier,
            f"tags.{self.state_identifier}": {
                "$in": [
                    self.reactant_tag,
                    self.product_tag,
                    self.transition_state_tag,
                ]
            },
        }

        find_filter.update(self.filter_collection)
        cursor = self.collection.find(
            find_filter,
            {
                "output.initial_molecule": 1,
                "output.final_energy": 1,
                "calcs_reversed.alpha_eigenvalues": 1,
                "calcs_reversed.beta_eigenvalues": 1,
                "calcs_reversed.alpha_coeff_matrix": 1,
                "calcs_reversed.beta_coeff_matrix": 1,
                "calcs_reversed.alpha_fock_matrix": 1,
                "calcs_reversed.beta_fock_matrix": 1,
                f"tags.{self.state_identifier}": 1,
            },
        ).sort(f"tags.{self.state_identifier}", 1)

        data = defaultdict(list)
        for document in cursor:

            alpha_eigenvalues = document["calcs_reversed"][0]["alpha_eigenvalues"]
            beta_eigenvalues = (
                document["calcs_reversed"][0]["beta_eigenvalues"]
                if "beta_eigenvalues" in document["calcs_reversed"][0]
                else None
            )

            alpha_coeff_matrix = document["calcs_reversed"][0]["alpha_coeff_matrix"]
            beta_coeff_matrix = (
                document["calcs_reversed"][0]["beta_coeff_matrix"]
                if "beta_coeff_matrix" in document["calcs_reversed"][0]
                else None
            )

            alpha_fock_matrix = document["calcs_reversed"][0]["alpha_fock_matrix"]
            beta_fock_matrix = (
                document["calcs_reversed"][0]["beta_fock_matrix"]
                if "beta_fock_matrix" in document["calcs_reversed"][0]
                else None
            )

            state = document[f"tags"][f"{self.state_identifier}"]

            base_quantities_qchem = BaseQuantitiesQChem(
                fock_matrix=alpha_fock_matrix,
                coeff_matrix=alpha_coeff_matrix,
                eigenvalues=alpha_eigenvalues,
            )
            alpha_orthogonalization_matrix = (
                base_quantities_qchem.get_orthogonalization_matrix()
            )
            alpha_orthonal_coeff_matrix = base_quantities_qchem.get_ortho_coeff_matrix()

            if beta_eigenvalues is not None:
                base_quantities_qchem = BaseQuantitiesQChem(
                    fock_matrix=beta_fock_matrix,
                    coeff_matrix=beta_coeff_matrix,
                    eigenvalues=beta_eigenvalues,
                )
                beta_orthogonalization_matrix = (
                    base_quantities_qchem.get_orthogonalization_matrix()
                )
                beta_orthonal_coeff_matrix = (
                    base_quantities_qchem.get_ortho_coeff_matrix()
                )

            eigenvalues = [alpha_eigenvalues]
            coeff_matrix = [alpha_orthonal_coeff_matrix]
            orthogonalization_matrix = [alpha_orthogonalization_matrix]

            if beta_eigenvalues is not None:
                eigenvalues.append(beta_eigenvalues)
                coeff_matrix.append(beta_orthonal_coeff_matrix)
                orthogonalization_matrix.append(beta_orthogonalization_matrix)

            data["eigenvalues"].append(eigenvalues)
            data["coeff_matrices"].append(coeff_matrix)
            data["orthogonalization_matrices"].append(orthogonalization_matrix)
            data["state"].append(state)
            data["structures"].append(document["output"]["initial_molecule"])
            data["identifiers"].append(identifier)
            data["final_energy"].append(document["output"]["final_energy"])

        data = {key: np.array(value) for key, value in data.items()}

        if len(data["state"]) == 3:
            self.data.append(data)

    def _parse_data(self, debug: bool = False) -> None:
        identifiers = self._get_all_identifiers()

        if debug:
            identifiers = identifiers[: self.debug_number_of_reactions]

        for identifier in tqdm.tqdm(identifiers):
            self._get_reaction_data(identifier)

    def get_data(self, debug: bool = False) -> List[dict]:
        """Get the data."""
        self._parse_data(debug=debug)
        return self.data

    def get_random_split_data(
        self,
        debug: bool = False,
        reparse: bool = True,
        train_frac: float = 0.8,
        test_frac: float = 0.1,
        validate_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[dict], List[dict]]:
        """Random split the data into a train and test set."""

        if reparse:
            self._parse_data(debug=debug)

        random.seed(seed)
        random.shuffle(self.data)

        self.train_list, self.test_list, self.validate_list = self._random_split(
            train_frac=train_frac, test_frac=test_frac, validate_frac=validate_frac
        )

        return self.train_list, self.test_list, self.validate_list

    def _random_split(self, train_frac: float, test_frac: float, validate_frac: float):
        """Randomly split the data into a train, test and validation set."""
        train_list = self.data[: int(train_frac * len(self.data))]
        test_list = self.data[
            int(train_frac * len(self.data)) : int(
                (train_frac + test_frac) * len(self.data)
            )
        ]
        validate_list = self.data[int((train_frac + test_frac) * len(self.data)) :]

        return train_list, test_list, validate_list
