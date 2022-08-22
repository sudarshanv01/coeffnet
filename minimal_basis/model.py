import logging

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace

from e3nn.util.test import equivariance_error

from dataset import HamiltonianDataset


class TSBarrierModel(torch.nn.Module):
    """Create a neural network model to predict
    the transition state energy node and edge attributes
    taken from matrices."""

    # Irreducible representation of the minimal basis hamiltonian.
    minimal_basis_irrep = o3.Irreps("1x0e + 1x1o + 1x2e")
    minimal_basis_size = 1 + 9 + 25
    irreps_out = o3.Irreps("1x0e")
    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)

    def __init__(self, max_radius=5, num_neighbors=10, num_basis=10):

        # Information about creating a minimal basis representation.
        logging.info("Creating neural network model.")
        logging.info("Irrep of minimal basis: {}".format(self.minimal_basis_irrep))
        logging.info("Size of minimal basis: {}".format(self.minimal_basis_size))

        super().__init__()

        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_basis = num_basis

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

        # Define the tensor products for each basis function
        self.tensor_product_s = o3.FullyConnectedTensorProduct(
            irreps_in1="1x0e",
            irreps_in2="1x0e",
            irreps_out="1x0e",
        )
        self.tensor_product_p = o3.FullyConnectedTensorProduct(
            irreps_in1="1x1o",
            irreps_in2="1x1o",
            irreps_out="1x1o",
        )
        self.tensor_product_d = o3.FullyConnectedTensorProduct(
            irreps_in1="1x2e",
            irreps_in2="1x2e",
            irreps_out="1x2e",
        )
        # Create a Fully Connected Tensor Product for the NN. The three integers of the
        # list in the first input are the dimensions of the input, intermediate and output.
        # The second option is the activation function.
        self.fc_network = nn.FullyConnectedNet(
            [self.num_basis, 16, self.tensor_product.weight_numel], torch.relu
        )

    def graph_generator(self, dataset):
        """Return the graph, including node and edge attributes,
        for a given dataset."""

        # Iterate over the dataset to get the characteristics of the graph.
        for data in dataset:
            logging.info("Processing new data graph.")
            # STRATEGY: Generate a minimal basis representation where the
            # Hamiltonians (of both spin up and down) are condensed
            # into a single matrix consisting of an irreducible
            # representation of only 1s, 1p and 1d.
            num_nodes = data["num_nodes"]
            minimal_basis = torch.zeros(num_nodes, self.minimal_basis_size)
            edge_dst = []
            edge_src = []
            edge_vec = []

            for mol_index, data_x in data["x"].items():
                # Within each molecule, the basis representation
                # of each atom must interact with the basis representation
                # of the other atoms. Therefore, we need to loop over all
                # pairs of atoms.

                # Generate the mean Hamiltonian for each spin.
                hamiltonian = data_x.mean(dim=-1)
                logging.debug(
                    "Shape of spin averaged Hamiltonian: {}".format(hamiltonian.shape)
                )

                logging.info(
                    "Constructing minimal basis representation for molecule {}.".format(
                        mol_index
                    )
                )

                # Iterate over all atoms, the minimal basis representation is for each atom.
                for i, atom_basis in enumerate(data["indices_atom_basis"][mol_index]):

                    logging.info("Processing atom {}.".format(i))

                    # Get the basis functions, in the right order, that make up
                    # atom_basis_1 and atom_basis_2
                    basis_functions = data["atom_basis_functions"][mol_index][
                        atom_basis[0] : atom_basis[1]
                    ]

                    # Find the indices of the 's', 'p' and 'd' basis functions
                    indices_s = [
                        i
                        for i, basis_function in enumerate(basis_functions)
                        if basis_function == "s"
                    ]
                    indices_p = [
                        i
                        for i, basis_function in enumerate(basis_functions)
                        if basis_function == "p"
                    ]
                    indices_d = [
                        i
                        for i, basis_function in enumerate(basis_functions)
                        if basis_function == "d"
                    ]
                    logging.debug(
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
                    tensor_s = torch.eye(1)
                    for s in indices_s:
                        segment_H = torch.tensor([[hamiltonian[s[0], s[0]]]])
                        tensor_s = self.tensor_product_s(tensor_s, segment_H)

                    logging.info("Shape of tensor_s: {}".format(tensor_s.shape))

                    tensor_p = torch.eye(3)
                    for p in indices_p:
                        segment_H = hamiltonian[p[0] : p[-1] + 1, p[0] : p[-1] + 1]
                        tensor_p = self.tensor_product_p(tensor_p, segment_H)

                    logging.info("Shape of tensor_p: {}".format(tensor_p.shape))

                    tensor_d = torch.eye(5)
                    for d in indices_d:
                        segment_H = hamiltonian[d[0] : d[-1] + 1, d[0] : d[-1] + 1]
                        tensor_d = self.tensor_product_d(tensor_d, segment_H)

                    logging.info("Shape of tensor_d: {}".format(tensor_d.shape))

                    # Make each of the tensors Hermitean.
                    tensor_s = torch.mm(tensor_s, tensor_s.t())
                    tensor_p = torch.mm(tensor_p, tensor_p.t())
                    tensor_d = torch.mm(tensor_d, tensor_d.t())
                    # Normalise by the number of basis functions.
                    tensor_s = tensor_s / len(indices_s) ** 2
                    tensor_p = tensor_p / len(indices_p) ** 2
                    tensor_d = tensor_d / len(indices_d) ** 2
                    logging.debug("Minimal basis (s): \n {}".format(tensor_s))
                    logging.debug("Minimal basis (p): \n {}".format(tensor_p))
                    logging.debug("Minimal basis (d): \n {}".format(tensor_d))

                    # Concatenate the tensors.
                    minimal_basis_flat = torch.cat(
                        (tensor_s.flatten(), tensor_p.flatten(), tensor_d.flatten())
                    )
                    minimal_basis[i, :] = minimal_basis_flat

                # Generate the graph for the current dataset.
                # Most of the edge dependent quantities are already stored in datapoint.
                # We just have to modify them into the form that we require here.
                # Again: Every molecule is independent in terms of the graph,
                # so all of these parameters can be independetly generated.
                edge_src_atom, edge_dst_atom = radius_graph(
                    data["pos"][mol_index],
                    self.max_radius,
                    max_num_neighbors=self.num_neighbors,
                )
                edge_vec_atom = (
                    data["pos"][mol_index][edge_dst_atom]
                    - data["pos"][mol_index][edge_src_atom]
                )
                # Append the atom indices to the lists.
                edge_src.append(edge_src_atom)
                edge_dst.append(edge_dst_atom)
                edge_vec.append(edge_vec_atom)

            # Make the edge lists into tensors.
            # Done here by concatenating the lists.
            edge_src = torch.cat(edge_src)
            edge_dst = torch.cat(edge_dst)
            edge_vec = torch.cat(edge_vec)

            yield num_nodes, edge_src, edge_dst, edge_vec, minimal_basis

    def forward(self, dataset):
        """Forward pass of the model."""

        output_vector = []

        for (
            num_nodes,
            edge_src,
            edge_dst,
            edge_vec,
            input_features,
        ) in self.graph_generator(dataset):

            # Determine the spherical harmonics to perform the convolution
            # The spherical harmonics of the chosen degree will be performed
            # on the (normalised) edge vector.
            # sh = o3.spherical_harmonics(
            #     self.irreps_sh, edge_vec, normalize=True, normalization="component"
            # )

            # # Also add an embedding for the MLP.
            embedding = soft_one_hot_linspace(
                x=edge_vec.norm(dim=1),
                start=0.0,
                end=self.max_radius,
                number=self.num_basis,
                basis="smooth_finite",
                cutoff=True,
            ).mul(self.num_basis**0.5)

            # # Construct the summation of the required dimension
            # output = self.tensor_product(
            #     input_features[edge_src], sh, self.fc_network(embedding)
            # )
            # output = scatter(output, edge_dst, dim=0, dim_size=num_nodes).div(
            #     self.num_neighbors**0.5
            # )

            # Append the summed output_vector to the list of output_vectors.
            # This operation corresponds to the summing of all atom-centered features.
            # output_vector.append(output.sum())
            output_vector.append(input_features.sum())

        # Make the output_vector a tensor.
        output_vector = torch.tensor(output_vector, requires_grad=True)

        return output_vector


if __name__ == "__main__":
    """Test a convolutional Neural Network"""

    # Read in the dataset inputs.
    JSON_FILE = "input_files/output_ts_calc_debug.json"
    BASIS_FILE = "input_files/def2-tzvppd.json"
    logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG)

    data_point = HamiltonianDataset(JSON_FILE, BASIS_FILE)
    data_point.load_data()
    data_point.parse_basis_data()
    datapoint = data_point.get_data()

    # Instantiate the model.
    model = TSBarrierModel()

    # Get the training y
    train_y = []
    for data in datapoint:
        train_y.append(data.y)

    # Make train_y a tensor.
    train_y = torch.tensor(train_y, dtype=torch.float)

    # Training the model.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(300):
        optim.zero_grad()
        pred = model(datapoint)
        loss = (pred - train_y).pow(2).sum()
        loss.backward()

        optim.step()

        if step % 10 == 0:
            accuracy = (pred - train_y).abs().sum()
            print(f"epoch {step:5d} | loss {loss:<10.1f} | {accuracy:5.1f} accuracy")
