from typing import Any, List, Tuple, Union, Dict

import logging

import numpy.typing as npt
import numpy as np

from scipy.linalg import eigh as scipy_eigh

from collections import defaultdict

from ase import data as ase_data

from pymatgen.core.structure import Molecule

import tqdm

import random

from minimal_basis.predata.cart_to_sph import cart_to_sph_d, norm_cart_d_gauss

logger = logging.getLogger(__name__)


class BaseMatrices:
    def __init__(
        self,
        fock_matrix: npt.ArrayLike,
        eigenvalues: npt.ArrayLike,
        coeff_matrix: npt.ArrayLike,
    ):
        """Start form the Fock matrix, the eigenvalues and the
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

    def get_fock_matrix(self) -> npt.ArrayLike:
        """Get the Fock matrix."""
        return self.fock_matrix

    def get_coeff_matrix(self) -> npt.ArrayLike:
        """Get the coefficient matrix."""
        return self.coeff_matrix

    def get_eigenvalues(self) -> npt.ArrayLike:
        """Get the eigenvalues."""
        return self.eigenvalues

    def get_overlap_matrix(self) -> npt.ArrayLike:
        """Get the overlap matrix."""
        if self.overlap_matrix is None:
            self._generate_overlap_matrix()
        return self.overlap_matrix

    def get_orthogonalization_matrix(self) -> npt.ArrayLike:
        """Get the orthogonalization matrix."""
        if self.overlap_matrix is None:
            self._generate_overlap_matrix()
        if self.orthogonalisation_matrix is None:
            self._generate_orthogonalization_matrix()
        return self.orthogonalisation_matrix

    def get_orthogonal_fock_matrix(self) -> npt.ArrayLike:
        """Get the orthogonal Fock matrix."""
        if self.overlap_matrix is None:
            self._generate_overlap_matrix()
        if self.orthogonalisation_matrix is None:
            self._generate_orthogonalization_matrix()
        if self.orthogonal_fock_matrix is None:
            self._generate_orthogonal_fock_matrix()
        return self.orthogonal_fock_matrix

    def get_ortho_coeff_matrix(self) -> npt.ArrayLike:
        """Get the orthogonal coefficient matrix."""
        if self.overlap_matrix is None:
            self._generate_overlap_matrix()
        if self.orthogonalisation_matrix is None:
            self._generate_orthogonalization_matrix()
        if self.orthogonal_fock_matrix is None:
            self._generate_orthogonal_fock_matrix()
        if self.ortho_coeff_matrix is None:
            self._generate_orthogonal_coeff_matrix()
        return self.ortho_coeff_matrix

    def _generate_overlap_matrix(self) -> None:
        """Determine the overlap element from the Hamiltonian, the
        eigen energies and the cofficient matrix.
        The transformation to get the overlap matrix is:

        S = F * C * (C * E)^-1

        where, F is the Fock matrix, C is the coefficient matrix and
        E is the diagonalised eigenvalues."""
        self.overlap_matrix = np.dot(
            np.dot(self.fock_matrix, self.coeff_matrix),
            np.linalg.inv(np.dot(self.coeff_matrix, self.diagonalised_eigen)),
        )

    def _generate_orthogonalization_matrix(self):
        """Determine the orthogonalization matrix from the overlap matrix.
        The transformation to get the orthogonalization matrix is X = S^(-1/2)
        S * D = L * D
        X = L * D^(-1/2) * L^T

        where S is the overlap matrix, L is the matrix of eigenvectors
        and D is the diagonal matrix of eigenvalues."""
        eigenval_overlap, eigenvec_overlap = np.linalg.eigh(self.overlap_matrix)
        self.orthogonalisation_matrix = np.dot(
            eigenvec_overlap,
            np.dot(np.diag(1 / np.sqrt(eigenval_overlap)), eigenvec_overlap.T),
        )

    def _generate_orthogonal_fock_matrix(self):
        """Determine the orthogonal Fock matrix from the orthogonalisation
        F' = X^T * F * X

        where F is the Fock matrix and X is the orthogonalisation matrix and F'
        is the orthogonal Fock matrix."""
        self.orthogonal_fock_matrix = np.dot(
            np.dot(self.orthogonalisation_matrix.T, self.fock_matrix),
            self.orthogonalisation_matrix,
        )

    def _generate_orthogonal_coeff_matrix(self):
        """Diagonalize the Fock matrix to get the orthogonal coefficient matrix.
        F' * C' = E' * C'
        where F' is the orthogonal Fock matrix, C' is the orthogonal coefficient
        matrix and E' is the diagonalised eigenvalues of the orthogonal Fock
        matrix.
        """
        eigenval_ortho_fock, eigenvec_ortho_fock = np.linalg.eigh(
            self.orthogonal_fock_matrix
        )
        self.ortho_coeff_matrix = eigenvec_ortho_fock
        self.eigenval_ortho_fock = eigenval_ortho_fock


class ReducedBasisMatrices(BaseMatrices):
    def __init__(
        self,
        fock_matrix: npt.ArrayLike,
        eigenvalues: npt.ArrayLike,
        coeff_matrix: npt.ArrayLike,
        indices_to_keep: List[int],
    ):
        """Reduce the basis set of the Hamiltonian and calculate
        all the matrices for this reduced basis set. This class
        would be useful to create a minimal basis representation.

        Args:
            fock_matrix (npt.ArrayLike): Fock matrix
            eigenvalues (npt.ArrayLike): Eigenvalues of the Fock matrix
            coeff_matrix (npt.ArrayLike): Coefficient matrix
            indices_to_keep (List[int]): Indices of the basis functions
                to keep.
        """

        super().__init__(fock_matrix, eigenvalues, coeff_matrix)
        self.indices_to_keep = indices_to_keep

        if len(self.indices_to_keep) > self.coeff_matrix.shape[0]:
            raise ValueError(
                "indices_to_keep cannot be greater than the number of basis functions"
            )

        if len(self.indices_to_keep) < self.coeff_matrix.shape[0]:
            self.set_reduced_overlap_matrix()
            self.set_reduced_fock_matrix()
            self.set_reduced_coeff_matrix_and_eigenvalues()
        elif len(self.indices_to_keep) == self.coeff_matrix.shape[0]:
            # If the number of basis functions to keep is the same as the
            # number of basis functions in the original basis set, then
            # the matrices are the same as the original matrices, only the
            # ordering may be different
            self.get_overlap_matrix()
            self.overlap_matrix = self.overlap_matrix[self.indices_to_keep, :][
                :, self.indices_to_keep
            ]
            self.fock_matrix = self.fock_matrix[self.indices_to_keep, :][
                :, self.indices_to_keep
            ]
            self.coeff_matrix = self.coeff_matrix[self.indices_to_keep, :]

    def set_reduced_overlap_matrix(
        self,
    ):
        """Get the overlap matrix for the reduced basis set.
        The overlap matrix is reduced by removing the rows and columns
        corresponding to the indices_to_remove."""
        overlap_matrix = self.get_overlap_matrix()
        reduced_overlap_matrix = overlap_matrix[self.indices_to_keep, :][
            :, self.indices_to_keep
        ]
        self.full_basis_overlap_matrix = self.overlap_matrix
        self.overlap_matrix = reduced_overlap_matrix

    def set_reduced_fock_matrix(self):
        """Reduce the number of basis functions in the Hamiltonian. This is done
        by removing the rows and columns corresponding to the indices_to_remove.
        F' = B * E' * B^T
        where F' is the reduced Fock matrix, B is the projection matrix,
        and E' is the diagonalised eigenvalues of the reduced coefficient matrix.

        The projection matrix is given by:
        B = S' * C'
        where S' is the reduced overlap matrix (of dimensions MxN) and C' is the
        reduced coefficient matrix (of dimensions NxN) where N is the number of
        basis functions in the full basis set and M is the number of basis functions
        in the reduced basis set.
        """
        diagonalised_eigen = np.zeros(
            (self.eigenvalues.shape[0], self.eigenvalues.shape[0])
        )
        np.fill_diagonal(diagonalised_eigen, self.eigenvalues)

        reduced_overlap_matrix = self.full_basis_overlap_matrix[self.indices_to_keep, :]
        projector = reduced_overlap_matrix @ self.coeff_matrix

        reduced_fock_matrix = np.dot(
            np.dot(projector, diagonalised_eigen),
            projector.T,
        )
        self.full_basis_fock_matrix = self.fock_matrix
        self.fock_matrix = reduced_fock_matrix

    def set_reduced_coeff_matrix_and_eigenvalues(self):
        """Solve the generalised eigenvalue problem for the reduced basis set.
        F' * C' = E' * S * C'
        where F' is the reduced Fock matrix, C' is the reduced coefficient matrix
        and E' is the diagonalised eigenvalues of the reduced coefficient matrix and
        S is the overlap matrix.
        """
        try:
            eigenvalues, eigenvectors = scipy_eigh(
                self.fock_matrix, self.overlap_matrix
            )
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "The overlap matrix is not positive definite. "
                "This is likely due to the basis set being linearly dependent."
            )
        self.coeff_matrix = eigenvectors
        self.eigenvalues = eigenvalues


class SphericalBaseMatrices(ReducedBasisMatrices):
    def __init__(
        self,
        fock_matrix: npt.ArrayLike,
        eigenvalues: npt.ArrayLike,
        coeff_matrix: npt.ArrayLike,
        indices_s_orbitals: List[int],
        indices_p_orbitals: List[List[int]],
        indices_d_orbitals: List[List[int]],
        generate_spherical_rep: bool = True,
        indices_to_keep: List[int] = None,
        **kwargs: Any,
    ):
        """Get the matrices in spherical basis."""
        super().__init__(
            fock_matrix, eigenvalues, coeff_matrix, indices_to_keep=indices_to_keep
        )

        self.indices_s_orbitals = indices_s_orbitals
        self.indices_p_orbitals = indices_p_orbitals
        self.indices_d_orbitals = indices_d_orbitals
        self.generate_spherical_rep = generate_spherical_rep
        self.projector_d_orbitals = cart_to_sph_d()

        self.spherical_ortho_coeff_matrix = None

    def get_spherical_ortho_coeff_matrix(self):
        """Get the orthogonal coefficient matrix in spherical basis."""
        if self.spherical_ortho_coeff_matrix is None:
            self.get_ortho_coeff_matrix()
            if self.generate_spherical_rep:
                self._generate_spherical_ortho_coeff_matrix()
            else:
                self.spherical_ortho_coeff_matrix = self.ortho_coeff_matrix

        return self.spherical_ortho_coeff_matrix

    def _generate_spherical_ortho_coeff_matrix(self):
        """Get the orthogonal coefficient matrix in spherical basis."""
        dim_cartesian = self.ortho_coeff_matrix.shape[0]
        # The only point of difference between the cartesian and spherical
        # is that the d basis functions are different. The d basis functions
        # for the cartesian basis have 6 functions, whereas the d basis
        # functions for the spherical basis have 5 functions.
        assert dim_cartesian == 1 * len(self.indices_s_orbitals) + 3 * len(
            self.indices_p_orbitals
        ) + 6 * len(self.indices_d_orbitals)
        dim_spherical = (
            1 * len(self.indices_s_orbitals)
            + 3 * len(self.indices_p_orbitals)
            + 5 * len(self.indices_d_orbitals)
        )

        spherical_ortho_coeff_matrix = np.zeros((dim_cartesian, dim_cartesian))
        row_idx_to_remove = []

        for idx_s in self.indices_s_orbitals:
            spherical_ortho_coeff_matrix[idx_s, :] = self.ortho_coeff_matrix[idx_s, :]
        for idx_p in self.indices_p_orbitals:
            spherical_ortho_coeff_matrix[idx_p, :] = self.ortho_coeff_matrix[idx_p, :]
        for idx_d in self.indices_d_orbitals:
            _cartesian_d_ortho_coeff = self.ortho_coeff_matrix[idx_d, :]
            _d_spherical_ortho_coeff = (
                self.projector_d_orbitals @ _cartesian_d_ortho_coeff
            )
            _sphericalidx_d = idx_d[:-1]
            spherical_ortho_coeff_matrix[_sphericalidx_d, :] = _d_spherical_ortho_coeff
            spherical_ortho_coeff_matrix[idx_d[-1], :] = None
            row_idx_to_remove.append(idx_d[-1])

        # Remove the extra rows stored in row_idx_to_remove
        spherical_ortho_coeff_matrix = np.delete(
            spherical_ortho_coeff_matrix, row_idx_to_remove, axis=0
        )
        self.spherical_ortho_coeff_matrix = spherical_ortho_coeff_matrix

        assert spherical_ortho_coeff_matrix.shape == (dim_spherical, dim_cartesian)


class TaskdocsToData:
    l_to_n_basis = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    n_to_l_basis = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}

    def __init__(
        self,
        collection,
        filter_collection: dict = {},
        identifier: str = "rxn_number",
        state_identifier: str = "state",
        reactant_tag: str = "reactant",
        product_tag: str = "product",
        transition_state_tag: str = "transition_state",
        basis_set_type: str = "full",
        basis_info_raw: Dict[str, Any] = None,
        s_orbital_specs: Dict[str, Any] = None,
        p_orbital_specs: Dict[str, Any] = None,
        d_orbital_specs: Dict[str, Any] = None,
        f_orbital_specs: Dict[str, Any] = None,
        g_orbital_specs: Dict[str, Any] = None,
        switch_to_spherical: bool = False,
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
        self.s_orbital_specs = s_orbital_specs
        self.p_orbital_specs = p_orbital_specs
        self.d_orbital_specs = d_orbital_specs
        self.f_orbital_specs = f_orbital_specs
        self.g_orbital_specs = g_orbital_specs
        self.switch_to_spherical = switch_to_spherical

        if self.switch_to_spherical:
            raise NotImplementedError(
                "Switching to spherical basis is not yet implemented."
            )
        if not self.p_orbital_specs["basis_are_spherical"]:
            raise NotImplementedError(
                "Only spherical p orbitals are supported at the moment."
            )
        if not self.d_orbital_specs["basis_are_spherical"]:
            raise NotImplementedError(
                "Only spherical d orbitals are supported at the moment."
            )

        if "store_extra_tags" in kwargs:
            if isinstance(kwargs["store_extra_tags"], list):
                self.store_extra_tags = kwargs["store_extra_tags"]
            elif isinstance(kwargs["store_extra_tags"], str):
                self.store_extra_tags = [kwargs["store_extra_tags"]]

        self.basis_info = None
        self.basis_info_raw = basis_info_raw

        assert basis_set_type in ["full", "minimal"]
        "Basis set type must be either full or minimal"
        self.basis_set_type = basis_set_type

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

    def _parse_basis_data(self):
        """Parse the basis information from data from basissetexchange.org
        json format to a dict containing the number of s, p and d functions
        for each atom. The resulting dictionary, self.basis_info contains the
        total set of basis functions for each atom.
        """

        self.basis_info = {}

        for atom_number in self.basis_info_raw["elements"]:
            angular_momentum_all = []
            for basis_index, basis_functions in enumerate(
                self.basis_info_raw["elements"][atom_number]["electron_shells"]
            ):
                no_of_repeats = len(basis_functions["coefficients"])
                no_of_basis_functions = len(basis_functions["angular_momentum"])
                if no_of_basis_functions == no_of_repeats:
                    angular_momentum_all.extend(basis_functions["angular_momentum"])
                else:
                    angular_momentum_all.extend(
                        no_of_repeats * basis_functions["angular_momentum"]
                    )
            angular_momentum_all = [
                self.n_to_l_basis[element] for element in angular_momentum_all
            ]
            self.basis_info[int(atom_number)] = angular_momentum_all

    def _get_atomic_number(cls, symbol):
        """Get the atomic number of an element based on the symbol."""
        return ase_data.atomic_numbers[symbol]

    def get_indices_info(self, molecule: Molecule):
        """Decide on the indices to keep for the minimal basis set."""

        atom_basis_counter = 0

        indices_to_keep = []  # Ordered list of indices
        orbital_info = []  # List of all orbital names
        irreps = ""  # String of all irreps

        indices_s_orbitals = []  # List of indices of s orbitals
        indices_p_orbitals = []  # List of indices of p orbitals
        indices_d_orbitals = []  # List of indices of d orbitals
        indices_f_orbitals = []  # List of indices of f orbitals
        indices_g_orbitals = []  # List of indices of g orbitals

        self._parse_basis_data()

        for idx, atom in enumerate(molecule):

            atomic_number = self._get_atomic_number(atom.species_string)
            basis_functions = self.basis_info[atomic_number]

            for basis_function in basis_functions:
                if basis_function == "s":
                    indices_to_keep.append(atom_basis_counter)
                    indices_s_orbitals.append([atom_basis_counter])
                    atom_basis_counter += 1
                    orbital_info.append([f"{atom.species_string}", f"{idx}", "s", ""])
                    irreps += "+1x0e"

                elif basis_function == "p":
                    if self.p_orbital_specs["inverted_coordinates"]:
                        indices_to_keep.extend(
                            [
                                atom_basis_counter + 1,
                                atom_basis_counter + 2,
                                atom_basis_counter,
                            ]
                        )
                        orbital_info.extend(
                            [
                                [f"{atom.species_string}", f"{idx}", "p", "y"],
                                [f"{atom.species_string}", f"{idx}", "p", "z"],
                                [f"{atom.species_string}", f"{idx}", "p", "x"],
                            ]
                        )
                    elif not self.p_orbital_specs["inverted_coordinates"]:
                        indices_to_keep.extend(
                            [
                                atom_basis_counter,
                                atom_basis_counter + 1,
                                atom_basis_counter + 2,
                            ]
                        )
                        orbital_info.extend(
                            [
                                [f"{atom.species_string}", f"{idx}", "p", "x"],
                                [f"{atom.species_string}", f"{idx}", "p", "y"],
                                [f"{atom.species_string}", f"{idx}", "p", "z"],
                            ]
                        )
                    indices_p_orbitals.append(
                        [
                            atom_basis_counter,
                            atom_basis_counter + 1,
                            atom_basis_counter + 2,
                        ]
                    )
                    atom_basis_counter += 3
                    irreps += "+1x1o"

                elif basis_function == "d":
                    if self.basis_set_type == "full":
                        if self.d_orbital_specs["basis_are_spherical"]:
                            indices_to_keep.extend(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                ]
                            )
                            if self.d_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "d", "xz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xy"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "d",
                                            "3y2-r2",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "d", "zy"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "d",
                                            "z2-x2",
                                        ],
                                    ]
                                )
                            elif not self.d_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "d", "xy"],
                                        [f"{atom.species_string}", f"{idx}", "d", "yz"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "d",
                                            "3z2-r2",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "d", "xz"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "d",
                                            "x2-y2",
                                        ],
                                    ]
                                )
                            irreps += "+1x2e"
                            indices_d_orbitals.append(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                ]
                            )
                            atom_basis_counter += 5
                        else:
                            indices_to_keep.extend(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                ]
                            )
                            if self.d_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "d", "zz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xx"],
                                        [f"{atom.species_string}", f"{idx}", "d", "yz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xy"],
                                        [f"{atom.species_string}", f"{idx}", "d", "yy"],
                                    ]
                                )
                            elif not self.d_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "d", "xx"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xy"],
                                        [f"{atom.species_string}", f"{idx}", "d", "yy"],
                                        [f"{atom.species_string}", f"{idx}", "d", "xz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "yz"],
                                        [f"{atom.species_string}", f"{idx}", "d", "zz"],
                                    ]
                                )
                            irreps += "+1x2e"
                            indices_d_orbitals.append(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                ]
                            )
                            atom_basis_counter += 6

                    elif self.basis_set_type == "minimal":
                        # No d-functions in minimal basis set
                        if self.d_orbital_specs["basis_are_spherical"]:
                            atom_basis_counter += 5
                        else:
                            atom_basis_counter += 6
                    else:
                        raise ValueError(
                            "Basis set type not recognized. Please choose between 'minimal' and 'full'."
                        )

                elif basis_function == "f":
                    if self.basis_set_type == "full":
                        if self.f_orbital_specs["basis_are_spherical"]:
                            indices_to_keep.extend(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                    atom_basis_counter + 6,
                                ]
                            )
                            if self.f_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "-3",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "f", "-2"],
                                        [f"{atom.species_string}", f"{idx}", "f", "-1"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "0",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "f", "1"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "2",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "3",
                                        ],
                                    ]
                                )
                            elif not self.f_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "f", "-3"],
                                        [f"{atom.species_string}", f"{idx}", "f", "-2"],
                                        [f"{atom.species_string}", f"{idx}", "f", "-1"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "0",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "1",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "2",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "f",
                                            "3",
                                        ],
                                    ]
                                )
                            irreps += "+1x3o"
                            indices_f_orbitals.append(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                    atom_basis_counter + 6,
                                ]
                            )
                            atom_basis_counter += 7
                        else:
                            raise NotImplementedError(
                                "Currently only spherical f-functions are supported."
                            )
                    elif self.basis_set_type == "minimal":
                        if self.f_orbital_specs["basis_are_spherical"]:
                            atom_basis_counter += 7
                        else:
                            raise NotImplementedError(
                                "Currently only spherical f-functions are supported."
                            )
                    else:
                        raise ValueError(
                            "Basis set type not recognized. Please choose between 'minimal' and 'full'."
                        )
                elif basis_function == "g":
                    if self.basis_set_type == "full":
                        if self.g_orbital_specs["basis_are_spherical"]:
                            indices_to_keep.extend(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                    atom_basis_counter + 6,
                                    atom_basis_counter + 7,
                                    atom_basis_counter + 8,
                                ]
                            )

                            if self.g_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "-4",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "g", "-3"],
                                        [f"{atom.species_string}", f"{idx}", "g", "-2"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "-1",
                                        ],
                                        [f"{atom.species_string}", f"{idx}", "g", "0"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "1",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "2",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "3",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "4",
                                        ],
                                    ]
                                )
                            elif not self.g_orbital_specs["inverted_coordinates"]:
                                orbital_info.extend(
                                    [
                                        [f"{atom.species_string}", f"{idx}", "g", "-4"],
                                        [f"{atom.species_string}", f"{idx}", "g", "-3"],
                                        [f"{atom.species_string}", f"{idx}", "g", "-2"],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "-1",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "0",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "1",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "2",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "3",
                                        ],
                                        [
                                            f"{atom.species_string}",
                                            f"{idx}",
                                            "g",
                                            "4",
                                        ],
                                    ]
                                )
                            irreps += "+1x4e"
                            indices_g_orbitals.append(
                                [
                                    atom_basis_counter,
                                    atom_basis_counter + 1,
                                    atom_basis_counter + 2,
                                    atom_basis_counter + 3,
                                    atom_basis_counter + 4,
                                    atom_basis_counter + 5,
                                    atom_basis_counter + 6,
                                    atom_basis_counter + 7,
                                    atom_basis_counter + 8,
                                ]
                            )
                            atom_basis_counter += 9
                        else:
                            raise NotImplementedError(
                                "Currently only spherical g-functions are supported."
                            )
                    elif self.basis_set_type == "minimal":
                        if self.g_orbital_specs["basis_are_spherical"]:
                            atom_basis_counter += 9
                        else:
                            raise NotImplementedError(
                                "Currently only spherical g-functions are supported."
                            )
                else:
                    raise ValueError(
                        "Basis function not recognized. Please choose between 's', 'p', 'd', 'f' and 'g'."
                    )

        indices_to_keep = np.array(indices_to_keep)
        return {
            "indices_to_keep": indices_to_keep,
            "orbital_info": orbital_info,
            "irreps": irreps[1:],
            "indices_s_orbitals": indices_s_orbitals,
            "indices_p_orbitals": indices_p_orbitals,
            "indices_d_orbitals": indices_d_orbitals,
            "indices_f_orbitals": indices_f_orbitals,
        }

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
        project = {
            "output.initial_molecule": 1,
            "output.final_energy": 1,
            "calcs_reversed.alpha_eigenvalues": 1,
            "calcs_reversed.beta_eigenvalues": 1,
            "calcs_reversed.alpha_coeff_matrix": 1,
            "calcs_reversed.beta_coeff_matrix": 1,
            "calcs_reversed.alpha_fock_matrix": 1,
            "calcs_reversed.beta_fock_matrix": 1,
            f"tags.{self.state_identifier}": 1,
        }
        if hasattr(self, "store_extra_tags"):
            for tag in self.store_extra_tags:
                project[f"tags.{tag}"] = 1

        find_filter.update(self.filter_collection)
        cursor = self.collection.find(
            find_filter,
            project,
        ).sort(f"tags.{self.state_identifier}", 1)

        data = defaultdict(list)
        for document in cursor:
            alpha_eigenvalues = document["calcs_reversed"][0]["alpha_eigenvalues"]
            alpha_eigenvalues = np.array(alpha_eigenvalues)
            beta_eigenvalues = (
                document["calcs_reversed"][0]["beta_eigenvalues"]
                if "beta_eigenvalues" in document["calcs_reversed"][0]
                else None
            )
            if beta_eigenvalues is not None:
                beta_eigenvalues = np.array(beta_eigenvalues)

            alpha_coeff_matrix = document["calcs_reversed"][0]["alpha_coeff_matrix"]
            alpha_coeff_matrix = np.array(alpha_coeff_matrix)
            beta_coeff_matrix = (
                document["calcs_reversed"][0]["beta_coeff_matrix"]
                if "beta_coeff_matrix" in document["calcs_reversed"][0]
                else None
            )
            if beta_coeff_matrix is not None:
                beta_coeff_matrix = np.array(beta_coeff_matrix)

            alpha_fock_matrix = document["calcs_reversed"][0]["alpha_fock_matrix"]
            alpha_fock_matrix = np.array(alpha_fock_matrix)
            beta_fock_matrix = (
                document["calcs_reversed"][0]["beta_fock_matrix"]
                if "beta_fock_matrix" in document["calcs_reversed"][0]
                else None
            )
            if beta_fock_matrix is not None:
                beta_fock_matrix = np.array(beta_fock_matrix)

            state = document[f"tags"][f"{self.state_identifier}"]

            molecule = document["output"]["initial_molecule"]
            molecule = Molecule.from_dict(molecule)
            indices_info = self.get_indices_info(
                molecule,
            )
            indices_to_keep = indices_info["indices_to_keep"]
            if self.basis_set_type == "full":
                # print(indices_to_keep.shape)
                # print(alpha_coeff_matrix.shape)
                # print(indices_to_keep)
                assert len(indices_to_keep) == alpha_coeff_matrix.shape[0]
            if self.basis_set_type == "minimal":
                assert len(indices_to_keep) == len(
                    indices_info["indices_s_orbitals"]
                ) + 3 * len(indices_info["indices_p_orbitals"])
            orbital_info = indices_info["orbital_info"]
            irreps = indices_info["irreps"]

            try:
                base_quantities_qchem = ReducedBasisMatrices(
                    fock_matrix=alpha_fock_matrix,
                    coeff_matrix=alpha_coeff_matrix,
                    eigenvalues=alpha_eigenvalues,
                    indices_to_keep=indices_to_keep,
                )
            except np.linalg.LinAlgError:
                logger.warning(
                    f"Could not compute the reduced basis matrices for {identifier} "
                    f"with state {state} due to a linear algebra error."
                )
                continue

            alpha_orthogonalization_matrix = (
                base_quantities_qchem.get_orthogonalization_matrix()
            )
            alpha_orthogonal_coeff_matrix = (
                base_quantities_qchem.get_ortho_coeff_matrix()
            )
            alpha_eigenvalues = base_quantities_qchem.eigenvalues

            if beta_eigenvalues is not None:
                try:
                    base_quantities_qchem = ReducedBasisMatrices(
                        fock_matrix=beta_fock_matrix,
                        coeff_matrix=beta_coeff_matrix,
                        eigenvalues=beta_eigenvalues,
                        indices_to_keep=indices_to_keep,
                    )
                except np.linalg.LinAlgError:
                    logger.warning(
                        f"Could not compute the reduced basis matrices for {identifier} "
                        f"with state {state} due to a linear algebra error."
                    )
                    continue
                beta_orthogonalization_matrix = (
                    base_quantities_qchem.get_orthogonalization_matrix()
                )
                beta_orthogonal_coeff_matrix = (
                    base_quantities_qchem.get_ortho_coeff_matrix()
                )
                beta_eigenvalues = base_quantities_qchem.eigenvalues

            eigenvalues = [alpha_eigenvalues]
            coeff_matrix = [alpha_orthogonal_coeff_matrix]
            orthogonalization_matrix = [alpha_orthogonalization_matrix]

            if beta_eigenvalues is not None:
                eigenvalues.append(beta_eigenvalues)
                coeff_matrix.append(beta_orthogonal_coeff_matrix)
                orthogonalization_matrix.append(beta_orthogonalization_matrix)

            data["eigenvalues"].append(eigenvalues)
            data["coeff_matrices"].append(coeff_matrix)
            data["orthogonalization_matrices"].append(orthogonalization_matrix)
            data["state"].append(state)
            data["structures"].append(document["output"]["initial_molecule"])
            data["identifiers"].append(identifier)
            data["final_energy"].append(document["output"]["final_energy"])
            data["indices_to_keep"].append(indices_to_keep)
            data["orbital_info"].append(orbital_info)
            data["irreps"].append(irreps)

            if hasattr(self, "store_extra_tags"):
                for key in self.store_extra_tags:
                    data[key].append(document["tags"][key])

        data = {key: np.array(value) for key, value in data.items()}

        if not data:
            return

        if len(data["state"]) < 3:
            return

        if (
            np.isnan(data["coeff_matrices"]).any()
            or np.isnan(data["orthogonalization_matrices"]).any()
        ):
            return

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
