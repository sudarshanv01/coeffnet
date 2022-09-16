import os
import logging
import glob

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

    # Irreducible representation of the minimal basis hamiltonian.
    minimal_basis_irrep = o3.Irreps("1x0e + 1x1o + 1x2e")
    minimal_basis_size = 1 + 9 + 25
    minimal_basis_matrix_size = 1 + 3 + 5
    num_basis = 1 + 3 + 5
    irreps_out = o3.Irreps("1x0e")
    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
    LOWER_BOUND = -20
    UPPER_BOUND = 20

    def __init__(self, device, max_radius=10):
        super().__init__()

        # Information about creating a minimal basis representation.
        logger.info("Creating neural network model.")
        logger.info("Irrep of minimal basis: {}".format(self.minimal_basis_irrep))
        logger.info("Size of minimal basis: {}".format(self.minimal_basis_size))

        self.device = device
        self.max_radius = max_radius
        # Define the tensor product that is needed
        # As of now, we will take the following expression
        # out = 1/sqrt(z) sum_{nearest neighbor} f_in * h(rel_pos) * Y(rel_pos)
        # where Y are the spherical harmonics, h is an MLP and f_in is the input feature.
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_in1=self.minimal_basis_irrep,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            shared_weights=False,
        )

        # Create a Fully Connected Tensor Product for the NN. The three integers of the
        # list in the first input are the dimensions of the input, intermediate and output.
        # The second option is the activation function.
        self.fc_network = nn.FullyConnectedNet(
            [self.num_basis, 32, self.tensor_product.weight_numel], torch.relu
        )

    def graph_generator(self, dataset):
        """Return the graph, including node and edge attributes,
        for a given dataset."""

        # Iterate over the dataset to get the characteristics of the graph.
        for data in dataset:
            logger.info("Processing new data graph.")
            # STRATEGY: Generate a minimal basis representation where the
            # Hamiltonians (of both spin up and down) are condensed
            # into a single matrix consisting of an irreducible
            # representation of only 1s, 1p and 1d.
            num_nodes = data["num_nodes"]

            minimal_basis = torch.zeros(
                num_nodes,
                self.minimal_basis_matrix_size,
                self.minimal_basis_matrix_size,
            )

            # Keep track of the atom index in the overall data object.
            index_atom = 0

            for mol_index, data_x in data["x"].items():
                # Within each molecule, the basis representation
                # of each atom must interact with the basis representation
                # of the other atoms. Therefore, we need to loop over all
                # pairs of atoms.

                # Generate the mean Hamiltonian for each spin.
                hamiltonian = data_x.mean(dim=-1)
                logger.debug(
                    "Shape of spin averaged Hamiltonian: {}".format(hamiltonian.shape)
                )

                logger.info(
                    "Constructing minimal basis representation for molecule {}.".format(
                        mol_index
                    )
                )

                # Iterate over all atoms, the minimal basis representation is for each atom.
                for i, atom_basis in enumerate(data["indices_atom_basis"][mol_index]):

                    logger.info("Processing atom {}.".format(i))

                    # Get the basis functions, in the right order, that make up
                    # atom_basis_1 and atom_basis_2
                    basis_functions = data["atom_basis_functions"][mol_index][
                        atom_basis[0] : atom_basis[1]
                    ]

                    # Find the indices of the 's', 'p' and 'd' basis functions
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
                    indices_p = [
                        indices_p[i : i + 3] for i in range(0, len(indices_p), 3)
                    ]
                    # Split indices_d into tuples of size 5.
                    indices_d = [
                        indices_d[i : i + 5] for i in range(0, len(indices_d), 5)
                    ]

                    # Create a tensor that contains the basis functions for the atom.
                    tensor_s = torch.zeros(1, 1)
                    for s in indices_s:
                        segment_H = torch.tensor([[hamiltonian[s[0], s[0]]]])
                        tensor_s += segment_H
                    logger.info("Shape of tensor_s: {}".format(tensor_s.shape))
                    tensor_s /= len(indices_s)

                    tensor_p = torch.zeros(3, 3)
                    for p in indices_p:
                        segment_H = hamiltonian[p[0] : p[-1] + 1, p[0] : p[-1] + 1]
                        tensor_p += segment_H
                    logger.info("Shape of tensor_p: {}".format(tensor_p.shape))
                    if len(indices_p) > 0:
                        tensor_p /= len(indices_p)

                    tensor_d = torch.zeros(5, 5)
                    for d in indices_d:
                        segment_H = hamiltonian[d[0] : d[-1] + 1, d[0] : d[-1] + 1]
                        tensor_d += segment_H
                    if len(indices_d) > 0:
                        tensor_d /= len(indices_d)

                    logger.info("Shape of tensor_d: {}".format(tensor_d.shape))

                    logger.debug("Minimal basis (s): \n {}".format(tensor_s))
                    logger.debug("Minimal basis (p): \n {}".format(tensor_p))
                    logger.debug("Minimal basis (d): \n {}".format(tensor_d))

                    minimal_basis[index_atom, 0, 0] = tensor_s
                    minimal_basis[index_atom, 1:4, 1:4] = tensor_p
                    minimal_basis[index_atom, 4:, 4:] = tensor_d

                    logger.info(
                        "Shape of minimal basis representation: {}".format(
                            minimal_basis[i].shape
                        )
                    )
                    logger.debug(
                        "Minimal basis representation: \n {}".format(minimal_basis[i])
                    )

                    # Add up the index of the atoms
                    index_atom += 1

            logger.debug(f"Minimal basis shape: {minimal_basis.shape}")

            edge_index = data.edge_index
            edge_src, edge_dst = edge_index
            logger.debug(f"Edge src: {edge_src}")
            logger.debug(f"Edge dst: {edge_dst}")
            edge_vec = data.pos[edge_dst] - data.pos[edge_src]
            logger.debug(f"Edge vec: {edge_vec}")

            minimal_basis_diag = torch.zeros(num_nodes, self.minimal_basis_matrix_size)
            for i, minimal_basis_atom in enumerate(minimal_basis):
                diagonal_rep_total = torch.diag(minimal_basis_atom)
                # Filter the diagonal representation such that
                # only elements between UPPER and LOWER are included.
                # Elements not obeying this condition are set to 0.
                diagonal_rep_filtered = torch.where(
                    (diagonal_rep_total > self.UPPER_BOUND)
                    | (diagonal_rep_total < self.LOWER_BOUND),
                    torch.zeros_like(diagonal_rep_total),
                    diagonal_rep_total,
                )
                minimal_basis_diag[i, ...] = diagonal_rep_filtered

            logger.debug(f"Minimal basis diag: {minimal_basis_diag}")
            logger.info(f"Minimal basis diag shape: {minimal_basis_diag.shape}")

            yield num_nodes, edge_src, edge_dst, edge_vec, edge_index, minimal_basis_diag

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
            input_features,
        ) in enumerate(self.graph_generator(dataset)):

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
    fig.savefig(f"plots/step_{step:03d}.png", dpi=300)
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

    logging.basicConfig(filename="model.log", filemode="w", level=logging.INFO)
    logger = logging.getLogger(__name__)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # Read in the dataset inputs.
    JSON_FILE = "input_files/output_QMrxn20_debug.json"
    BASIS_FILE = "input_files/sto-3g.json"
    CHECKPOINT_FOLDER = "checkpoints"

    # Create the folder if it does not exist.
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Get details of the checkpoint
    checkpoint_file = avail_checkpoint(CHECKPOINT_FOLDER)

    data_point = HamiltonianDataset(JSON_FILE, BASIS_FILE)
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
    with open("training.log", "w") as f:
        f.write("Epoch\t Loss\t Accuracy\n")

    for step in range(300):

        optim.zero_grad()
        pred = model(datapoint)
        loss = (pred - train_y).pow(2).sum()

        loss.backward()

        # Write out the epoch and loss to the log file.
        accuracy = (pred - train_y).abs().sum()
        with open("training.log", "a") as f:
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
