import os
import logging
import glob
import datetime
from typing import Dict, Union, List, Tuple
from pprint import pprint

import torch
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace

from dataset import HamiltonianDataset

import matplotlib.pyplot as plt
from plot_params import get_plot_params

get_plot_params()


class TSBarrierModel(MessagePassing):
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

    def __init__(self, device, max_radius=10):
        super().__init__()

        # Information about creating a minimal basis representation.
        logger.info("Creating neural network model.")

        self.device = device
        self.max_radius = max_radius

        # The first tensor product is between node (or edge) features
        # and spherical harmonics.
        self.tensor_product_1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_minimal_basis,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_minimal_basis,
        )
        # The second tensor product is between node and edge features.
        self.tensor_product_2 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_minimal_basis,
            irreps_in2=self.irreps_minimal_basis,
            irreps_out=self.irreps_out,
        )

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

        logging.debug("Minimal basis representation of the Hamiltonian (alpha):")
        logging.debug(minimal_basis[..., 0])
        logging.debug("Minimal basis representation of the Hamiltonian (beta):")
        logging.debug(minimal_basis[..., 1])

        return minimal_basis

    def graph_generator(self, dataset):
        """Generate node and edge attributes as well as other critical information
        about the graph. This generator is meant to be used in the forward for loop."""

        # Iterate over the dataset to get the characteristics of the graph.
        for data in dataset:
            logger.info("Processing new data graph.")
            # STRATEGY: Generate a minimal basis representation where the
            # Hamiltonians (of both spin up and down) are condensed
            # into a single matrix consisting of an irreducible
            # representation of only 1s, 1p and 1d.
            num_nodes = data["num_nodes"]

            # Edge information, both source and destination.
            edge_index = data.edge_index
            edge_src, edge_dst = edge_index
            logger.debug(f"Edge src: {edge_src}")
            logger.debug(f"Edge dst: {edge_dst}")
            edge_vec = data.pos[edge_dst] - data.pos[edge_src]
            logger.debug(f"Edge vec: {edge_vec}")
            num_edges = edge_src.shape[0]

            # The minimal basis representation of the node features of
            # the Hamiltonian. The shape of this matrix is (all_nodes, num_basis, num_basis, 2).
            # The last dimension is for the spin up and spin down components.
            minimal_basis_node = torch.zeros(
                num_nodes,
                self.minimal_basis_matrix_size,
                self.minimal_basis_matrix_size,
                2,
            )
            # Similar minimal basis construction for the edge features.
            minimal_basis_edge = torch.zeros(
                num_edges,
                self.minimal_basis_matrix_size,
                self.minimal_basis_matrix_size,
                2,
            )

            # Keep track of the atom index in the overall data object.
            index_atom = 0

            # Start with constructing the node features.
            # Since we are looking at the node features, it is sufficient to
            # iterate over all atoms (as they correspond to the nodes).
            for mol_index, hamiltonian in data["x"].items():
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

            # Now construct the edge features. Iterate over the edge features
            # taking information from the source and destination nodes. These
            # source and destination nodes will provide information about the
            # sub-blocks of the coupling matrix that are needed to construct
            # the minimal basis representation of the edge features.
            for edge_i, (src, dst) in enumerate(zip(edge_src, edge_dst)):
                # The source and destination nodes should give the index
                # of the features that we want to construct.
                start_src, end_src = data["indices_atom_basis"][src]
                start_dst, end_dst = data["indices_atom_basis"][dst]
                # Carve out the sub-blocks of the coupling matrix that are needed.
                # The matrix is Hermitian, so we only need to take the upper triangle.
                submatrix_V = data["edge_attr"][
                    start_src:end_src, start_dst:end_dst, ...
                ]
                # Determine the basis functions in submatrix_V
                basis_functions_src = data["atom_basis_functions"][src][
                    start_src:end_src
                ]
                basis_functions_dst = data["atom_basis_functions"][dst][
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

            adsdsa

            # Determine the spherical harmonics to perform the convolution
            # The spherical harmonics of the chosen degree will be performed
            # on the (normalised) edge vector.
            sh = o3.spherical_harmonics(
                self.irreps_sh, edge_vec, normalize=True, normalization="component"
            )

            # Also add an embedding for the MLP.
            embedding = soft_one_hot_linspace(
                x=edge_vec.norm(dim=1),
                start=0.0,
                end=self.max_radius,
                number=self.num_basis,
                basis="smooth_finite",
                cutoff=True,
            ).mul(self.num_basis**0.5)

            # Transfer everything to the GPU.
            sh = sh.to(self.device)
            embedding = embedding.to(self.device)
            input_features = input_features.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_dst = edge_dst.to(self.device)

            output = self.tensor_product(
                input_features[edge_src], sh, self.fc_network(embedding)
            )
            output = scatter(output, edge_dst, dim=0, dim_size=num_nodes).div(
                num_nodes**0.5
            )
            logger.debug(f"Output: {output}")

            # Propogate the output through the network.
            output = self.propagate(edge_index, x=output, norm=num_nodes)

            # Append the summed output_vector to the list of output_vectors.
            # This operation corresponds to the summing of all atom-centered features.
            norm_output = torch.sum(output)
            # Normalise by the number of nodes.
            norm_output /= num_nodes
            # Reutn the absolute value of the output.
            norm_output = torch.abs(norm_output)

            output_vector[index] = norm_output

        return output_vector


def visualize_results(predicted, calculated, epoch=None, loss=None):
    """Plot the DFT calculated vs the fit results."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    predicted = predicted.detach().cpu().numpy()
    calculated = calculated.detach().cpu().numpy()
    ax.scatter(calculated, predicted, cmap="Set2")
    if epoch is not None and loss is not None:
        ax.set_xlabel(f"Epoch: {epoch}, Loss: {loss.item():.3f}")
    ax.set_xlabel("DFT calculated (eV)")
    ax.set_ylabel("Fit results (eV)")
    fig.savefig(f"{PLOT_FOLDER}/step_{step:03d}.png", dpi=300)
    plt.close(fig)


def avail_checkpoint(CHECKPOINT_DIR):
    """Check if a checkpoint file is available, and if it is
    return the name of the largest checkpoint file."""
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt"))
    if len(checkpoint_files) == 0:
        return None
    else:
        return max(checkpoint_files, key=os.path.getctime)


if __name__ == "__main__":
    """Test a convolutional Neural Network"""

    LOGFILES_FOLDER = "log_files"
    logging.basicConfig(
        filename=os.path.join(LOGFILES_FOLDER, "model.log"),
        filemode="w",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # Prefix tag to the output folders
    today = datetime.datetime.now()
    folder_string = today.strftime("%Y%m%d_%H%M%S")

    # Read in the dataset inputs.
    JSON_FILE = "input_files/output_QMrxn20_debug.json"
    BASIS_FILE = "input_files/sto-3g.json"
    CHECKPOINT_FOLDER = "checkpoints"
    PLOT_FOLDER = f"plots/{folder_string}"
    GRAPH_GENERTION_METHOD = "sn2"

    # Create the folder if it does not exist.
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    # Get details of the checkpoint
    checkpoint_file = avail_checkpoint(CHECKPOINT_FOLDER)

    data_point = HamiltonianDataset(
        JSON_FILE, BASIS_FILE, graph_generation_method=GRAPH_GENERTION_METHOD
    )
    data_point.load_data()
    data_point.parse_basis_data()
    datapoint = data_point.get_data()

    # Instantiate the model.
    model = TSBarrierModel(DEVICE)
    model.to(DEVICE)
    if checkpoint_file is not None:
        model.load_state_dict(torch.load(checkpoint_file))
        logger.info(f"Loaded checkpoint file: {checkpoint_file}")
        model.eval()
    else:
        logger.info("No checkpoint file found, starting from scratch")

    # Get the training y
    train_y = []
    for data in datapoint:
        train_y.append(data.y)

    # Make train_y a tensor.
    train_y = torch.tensor(train_y, dtype=torch.float)

    # Training the model.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Write header of log file
    with open(os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "w") as f:
        f.write("Epoch\t Loss\t Accuracy\n")

    for step in range(500):

        optim.zero_grad()
        pred = model(datapoint)
        loss = (pred - train_y).pow(2).sum()

        loss.backward()

        # Write out the epoch and loss to the log file.
        # TODO: Change this to better reflect the loss.
        # Right now it is just the mean absolute error.
        accuracy = (pred - train_y).abs().sum() / len(train_y)
        with open(
            os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "a"
        ) as f:
            f.write(f"{step:5d} \t {loss:<10.1f} \t {accuracy:5.1f}\n")

        if step % 10 == 0:

            # Plot the errors for each step.
            visualize_results(pred, train_y, epoch=step, loss=loss)

            # Save the model params.
            logger.debug(f"Saving model parameters for step {step}")
            for param_tensor in model.state_dict():
                logger.debug(param_tensor)
                logger.debug(model.state_dict()[param_tensor].size())
            for name, param in model.named_parameters():
                logger.debug(name)
                logger.debug(param.grad)
            torch.save(model.state_dict(), f"{CHECKPOINT_FOLDER}/step_{step:03d}.pt")

        optim.step()
