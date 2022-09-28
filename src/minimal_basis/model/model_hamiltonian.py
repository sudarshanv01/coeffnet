from typing import Dict, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

import torch
from torch_scatter import scatter
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing

from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

import matplotlib.pyplot as plt


class HamiltonianModel(MessagePassing):
    """Create a neural network model to predict
    the transition state energy node and edge attributes
    taken from matrices."""

    # A minimal basis consisting of s, p and d components.
    # Any node or edge feature must be of this irrep.
    irreps_minimal_basis = o3.Irreps("1x0e + 1x1o + 1x2e")
    # The spherical hamonics which we will take tensor products with
    # for both node and edge features.
    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
    # The final output irrep which will be with the tensor product
    # of the node and edge features.
    irreps_out = o3.Irreps("1x0e + 1x1o + 1x2e")
    # Also store the dimensions of the minimal basis representation.
    minimal_basis_matrix_size = 9
    # Truncate the energy range for the operators
    MIN_ENERGY = -10
    MAX_ENERGY = 10

    def __init__(self, device, max_radius=10):
        super().__init__()

        # Information about creating a minimal basis representation.
        logger.info("Creating neural network model.")

        self.device = device
        self.max_radius = max_radius

        # The first tensor product is between node (or edge) features
        # and spherical harmonics. The weights for this tensor product
        # will be provided by the network.
        self.tensor_product_1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_minimal_basis,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_minimal_basis,
            shared_weights=False,
        )
        # The second tensor product is between node and edge features.
        self.tensor_product_2 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_minimal_basis,
            irreps_in2=self.irreps_minimal_basis,
            irreps_out=self.irreps_out,
            shared_weights=False,
        )
        # Create a fully connected nn to perform the tensor products
        self.fc_network_1 = nn.FullyConnectedNet(
            [self.minimal_basis_matrix_size, 16, self.tensor_product_1.weight_numel],
            torch.relu,
        )
        self.fc_network_1 = self.fc_network_1.to(self.device)
        # Create a fully connected nn to perform products between node and edge features.
        self.fc_network_2 = nn.FullyConnectedNet(
            [self.minimal_basis_matrix_size, 16, self.tensor_product_2.weight_numel],
            torch.relu,
        )
        self.fc_network_2 = self.fc_network_2.to(self.device)

    def determine_spd_functions(cls, basis_functions):
        """For a list of basis functions, determine the number of s, p and d functions.
        Typically useful for finding the relevant index of a sub-matrix."""
        indices_s = []
        indices_p = []
        indices_d = []
        for i, basis_functions in enumerate(basis_functions):
            if basis_functions == "s":
                indices_s.append(i)
            elif basis_functions == "p":
                indices_p.append(i)
            elif basis_functions == "d":
                indices_d.append(i)

        logger.debug(
            "There are {} s, {} p and {} d basis functions.".format(
                len(indices_s), len(indices_p), len(indices_d)
            )
        )

        # Group the indices into tuples, s has length = 1, p has length 3 and d has length 5.
        indices_s = [[index] for index in indices_s]
        # Split indices_p into tuples of size 3.
        indices_p = [indices_p[i : i + 3] for i in range(0, len(indices_p), 3)]
        # Split indices_d into tuples of size 5.
        indices_d = [indices_d[i : i + 5] for i in range(0, len(indices_d), 5)]

        return indices_s, indices_p, indices_d

    def construct_minimal_basis(
        cls,
        indices_s: List[list],
        indices_p: List[list],
        indices_d: List[list],
        matrix: torch.tensor,
    ) -> torch.Tensor:
        """For any set of combined s,p and d matrices, get the minimal basis
        representation consisting of only 1s, 1p and 1d matrix."""

        # First construct the diagonal blocks of the matrix.
        # Starting with the s-block.
        tensor_s = torch.zeros(1, 1, 2)
        index_s1, index_s2 = indices_s
        for s1 in index_s1:
            for s2 in index_s2:
                segment_H = matrix[s1[0], s2[0], ...]
                tensor_s += segment_H
        tensor_s /= len(index_s1) * len(index_s2)

        # Now construct the p-block.
        tensor_p = torch.zeros(3, 3, 2)
        index_p1, index_p2 = indices_p
        for p1 in index_p1:
            for p2 in index_p2:
                segment_H = matrix[p1[0] : p1[-1] + 1, p2[0] : p2[-1] + 1, ...]
                tensor_p += segment_H
        if len(index_p1) > 0 and len(index_p2) > 0:
            tensor_p /= len(index_p1) * len(index_p2)

        # And finally the d-block.
        tensor_d = torch.zeros(5, 5, 2)
        index_d1, index_d2 = indices_d
        for d1 in index_d1:
            for d2 in index_d2:
                segment_H = matrix[d1[0] : d1[-1] + 1, d2[0] : d2[-1] + 1, ...]
                tensor_d += segment_H
        if len(index_d1) > 0 and len(index_d2) > 0:
            tensor_d /= len(index_d1) * len(index_d2)

        # Now construct the off-diagonal blocks of the matrix.
        # Starting with the s-p block.
        tensor_sp = torch.zeros(1, 3, 2)
        for s in index_s1:
            for p in index_p2:
                segment_H = matrix[s[0], p[0] : p[-1] + 1, ...]
                tensor_sp += segment_H
        if len(index_s1) > 0 and len(index_p2) > 0:
            tensor_sp /= len(index_s1) * len(index_p2)

        # Now construct the s-d block.
        tensor_sd = torch.zeros(1, 5, 2)
        for s in index_s1:
            for d in index_d2:
                segment_H = matrix[s[0], d[0] : d[-1] + 1, ...]
                tensor_sd += segment_H
        if len(index_s1) > 0 and len(index_d2) > 0:
            tensor_sd /= len(index_s1) * len(index_d2)

        # And finally the p-d block.
        tensor_pd = torch.zeros(3, 5, 2)
        for p in index_p1:
            for d in index_d2:
                segment_H = matrix[p[0] : p[-1] + 1, d[0] : d[-1] + 1, ...]
                tensor_pd += segment_H
        if len(index_p1) > 0 and len(index_d2) > 0:
            tensor_pd /= len(index_p1) * len(index_d2)

        # Create transposed versions of the off-diagonal blocks.
        tensor_sp_t = torch.zeros(3, 1, 2)
        tensor_sd_t = torch.zeros(5, 1, 2)
        tensor_pd_t = torch.zeros(5, 3, 2)
        for i in range(2):
            tensor_sp_t[..., i] = tensor_sp[..., i].T
            tensor_sd_t[..., i] = tensor_sd[..., i].T
            tensor_pd_t[..., i] = tensor_pd[..., i].T

        # Put the blocks together to form the minimal basis representation.
        minimal_basis = torch.zeros(
            cls.minimal_basis_matrix_size, cls.minimal_basis_matrix_size, 2
        )
        minimal_basis[:1, :1, :] = tensor_s
        minimal_basis[1:4, 1:4, :] = tensor_p
        minimal_basis[4:, 4:, :] = tensor_d
        minimal_basis[:1, 1:4, :] = tensor_sp
        minimal_basis[1:4, :1, :] = tensor_sp_t
        minimal_basis[:1, 4:, :] = tensor_sd
        minimal_basis[4:, :1, :] = tensor_sd_t
        minimal_basis[1:4, 4:, :] = tensor_pd
        minimal_basis[4:, 1:4, :] = tensor_pd_t

        logger.debug("Minimal basis representation of the Hamiltonian (alpha):")
        logger.debug(minimal_basis[..., 0])
        logger.debug("Minimal basis representation of the Hamiltonian (beta):")
        logger.debug(minimal_basis[..., 1])

        return minimal_basis

    def graph_generator(self, dataset):
        """Generate node and edge attributes as well as other critical information
        about the graph. This generator is meant to be used in the forward for loop."""

        # Iterate over the dataset to get the characteristics of the graph.
        for data in dataset:
            logger.info("Processing new data graph.")

            # === Information about the graph ===
            # Node information
            num_nodes = data["num_nodes"]

            # Edge information
            edge_index = data.edge_index
            edge_src, edge_dst = edge_index
            logger.debug(f"Edge src: {edge_src}")
            logger.debug(f"Edge dst: {edge_dst}")
            edge_vec = data.pos[edge_dst] - data.pos[edge_src]
            logger.debug(f"Edge vec: {edge_vec}")
            num_edges = edge_src.shape[0]

            # === Intialise the node and edge attributes ===
            # The minimal basis representation of the node features of
            # the Hamiltonian. The shape of this matrix is (all_nodes, num_basis, num_basis, 2).
            # The last dimension is for the spin up and spin down components.
            minimal_basis_node = torch.zeros(
                num_nodes,
                self.minimal_basis_matrix_size,
                self.minimal_basis_matrix_size,
                2,
                device=self.device,
            )
            # Minimal basis representation of the edge features of the Hamiltonian.
            minimal_basis_edge = torch.zeros(
                num_edges,
                self.minimal_basis_matrix_size,
                self.minimal_basis_matrix_size,
                2,
                device=self.device,
            )
            # Keep track of the atom index in the overall data object.
            index_atom = 0

            # === Node attributes ===
            # Since we are looking at the node features, it is sufficient to
            # iterate over all atoms (as they correspond to the nodes).
            for mol_index, hamiltonian in sorted(data["x"].items()):
                # Note that we are sorting the keys to make sure that the
                # atom data is the same as the one in Data object, and that
                # way we have a consistent measure of the node features.
                for i, atom_basis in enumerate(data["indices_atom_basis"][mol_index]):

                    logger.info("Processing atom {}.".format(i))

                    # Get the basis functions, in the right order, that make up
                    # atom_basis_1 and atom_basis_2
                    basis_functions = data["atom_basis_functions"][mol_index][
                        atom_basis[0] : atom_basis[1]
                    ]

                    # Find the indices of the 's', 'p' and 'd' basis functions
                    # The indices are relative to the submatrix of the Hamiltonian
                    # that this loop is working on.
                    indices_s, indices_p, indices_d = self.determine_spd_functions(
                        basis_functions
                    )

                    # Get the minimal basis representation of the Hamiltonian
                    # for each node. Since we are on the diagonal block elements,
                    # each list of indices can just be doubled for the x, y labelling.
                    minimal_basis = self.construct_minimal_basis(
                        [indices_s, indices_s],
                        [indices_p, indices_p],
                        [indices_d, indices_d],
                        hamiltonian,
                    )
                    # Store the minimal basis representation for each node.
                    minimal_basis_node[index_atom, ...] = minimal_basis

                    # Add up the index of the atoms
                    index_atom += 1

            # === Edge attributes ===
            # Start by determining the mapping between the edge list and the
            # molecule from which they came from. This is stored seprately
            # for each data object. We will also need a mapping between
            # the edge index and the internal index of each molecule.
            edge_molecule_mapping = data["edge_molecule_mapping"]
            edge_internal_mol_mapping = data["edge_internal_mol_mapping"]
            # Iterate over the edge features taking information from the source and destination nodes. These
            # source and destination nodes will provide information about the
            # sub-blocks of the coupling matrix that are needed to construct
            # the minimal basis representation of the edge features.
            for edge_i, (src_, dst_) in enumerate(zip(edge_src, edge_dst)):
                # Convert src_ and dst_ to floats
                src = int(src_)
                dst = int(dst_)
                # The source and destination nodes should give the index
                # of the features that we want to construct.
                src_mol = edge_molecule_mapping[src]
                dst_mol = edge_molecule_mapping[dst]
                # If the source and destination nodes are from the same molecule,
                # then it is fine to assume that they have some inter-orbital interaction
                # because they come fromt he same calculation. If not, then we
                # have to assume that the interaction is zero, as a matrix.
                if src_mol == dst_mol:
                    src_internal = edge_internal_mol_mapping[src]
                    dst_internal = edge_internal_mol_mapping[dst]
                    start_src, end_src = data["indices_atom_basis"][src_mol][
                        src_internal
                    ]
                    start_dst, end_dst = data["indices_atom_basis"][dst_mol][
                        dst_internal
                    ]
                    # Carve out the sub-blocks of the coupling matrix that are needed.
                    # The matrix is Hermitian, so we only need to take the upper triangle.
                    submatrix_V = data["edge_attr"][src_mol][
                        start_src:end_src, start_dst:end_dst, ...
                    ]
                    # Determine the basis functions in submatrix_V
                    basis_functions_src = data["atom_basis_functions"][src_mol][
                        start_src:end_src
                    ]
                    basis_functions_dst = data["atom_basis_functions"][dst_mol][
                        start_dst:end_dst
                    ]
                    (
                        indices_s_src,
                        indices_p_src,
                        indices_d_src,
                    ) = self.determine_spd_functions(basis_functions_src)
                    (
                        indices_s_dst,
                        indices_p_dst,
                        indices_d_dst,
                    ) = self.determine_spd_functions(basis_functions_dst)
                    minimal_basis = self.construct_minimal_basis(
                        [indices_s_src, indices_s_dst],
                        [indices_p_src, indices_p_dst],
                        [indices_d_src, indices_d_dst],
                        submatrix_V,
                    )
                    minimal_basis_edge[edge_i, ...] = minimal_basis
                else:
                    # If we are considering inter-orbitals interaction between two molecules
                    # then we assume that the interaction is zero.
                    pass

            yield num_nodes, edge_src, edge_dst, edge_vec, edge_index, minimal_basis_node, minimal_basis_edge

    def forward(self, dataset) -> torch.Tensor:
        """Forward pass of the model."""

        # Store all the energies here.
        output_vector = torch.zeros(len(dataset))

        for index, (
            num_nodes,
            edge_src,
            edge_dst,
            edge_vec,
            edge_index,
            node_features,
            edge_features,
        ) in enumerate(self.graph_generator(dataset)):

            # === Move tensors to the correct device ===
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_vec = edge_vec.to(self.device)
            edge_index = edge_index.to(self.device)
            node_features = node_features.to(self.device)
            edge_features = edge_features.to(self.device)

            # === Construct embedding ===
            embedding = soft_one_hot_linspace(
                x=edge_vec.norm(dim=1),
                start=0.0,
                end=self.max_radius,
                number=self.minimal_basis_matrix_size,
                basis="smooth_finite",
                cutoff=True,
            ).mul(self.minimal_basis_matrix_size**0.5)
            embedding = embedding.to(self.device)

            # === Construct the spherical harmonics ===
            sh = o3.spherical_harmonics(
                self.irreps_sh, edge_vec, normalize=True, normalization="component"
            )
            sh = sh.to(self.device)

            # === Constuct appropriate tensor products ===
            # Mean over the last dimension of node and edge features
            # Physically, this means that we are averaging over the
            # the two spins.
            edge_features = edge_features.mean(dim=-1)
            node_features = node_features.mean(dim=-1)

            # Now construct diagonal version of these matrices.
            diag_nodes = torch.zeros(
                node_features.shape[0],
                self.minimal_basis_matrix_size,
                device=self.device,
            )
            diag_edges = torch.zeros(
                edge_features.shape[0],
                self.minimal_basis_matrix_size,
                device=self.device,
            )
            # Take the diagonal elements and store them
            # in the diagonal matrices.
            # Set elements that are lower than MIN_ENERGY and higher
            # than MAX_ENERGY to 0 in diag_nodes and diag_edges
            for i in range(node_features.shape[0]):
                diag_nodes_ = torch.diag(node_features[i, ...])
                diag_nodes_ = torch.where(
                    diag_nodes_ < self.MIN_ENERGY,
                    torch.zeros_like(diag_nodes_),
                    diag_nodes_,
                )
                diag_nodes_ = torch.where(
                    diag_nodes_ > self.MAX_ENERGY,
                    torch.zeros_like(diag_nodes_),
                    diag_nodes_,
                )
                diag_nodes[i, :] = diag_nodes_
            for i in range(edge_features.shape[0]):
                diag_edges_ = torch.diag(edge_features[i, ...])
                diag_edges_ = torch.where(
                    diag_edges_ < self.MIN_ENERGY,
                    torch.zeros_like(diag_edges_),
                    diag_edges_,
                )
                diag_edges_ = torch.where(
                    diag_edges_ > self.MAX_ENERGY,
                    torch.zeros_like(diag_edges_),
                    diag_edges_,
                )
                diag_edges[i, :] = diag_edges_

            logger.debug("Diagonal node features are:")
            logger.debug(diag_nodes)
            logger.debug("Diagonal edge features are:")
            logger.debug(diag_edges)

            # === First batch of tensor products ===
            tp_diag_edges = self.tensor_product_1(
                diag_edges[edge_src],
                sh,
                self.fc_network_1(embedding),
            )
            tp_diag_nodes = self.tensor_product_1(
                diag_nodes[edge_dst],
                sh,
                self.fc_network_1(embedding),
            )

            # === Second batch of tensor products ===
            output = self.tensor_product_2(
                tp_diag_edges[edge_src],
                tp_diag_nodes[edge_src],
                self.fc_network_2(embedding),
            )
            logger.debug("Output of the tensor product is:")
            logger.debug(output)

            # Propogate the output through the network.
            output = self.propagate(edge_index, x=output, norm=num_nodes)

            # Append the summed output_vector to the list of output_vectors.
            # This operation corresponds to the summing of all atom-centered features.
            norm_output = torch.sum(output)
            # Normalise by the number of nodes.
            norm_output /= num_nodes

            # Return the absolute value of the output.
            norm_output = torch.abs(norm_output)

            output_vector[index] = norm_output

        return output_vector

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return x_j
