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
    minimal_basis_irrep = o3.Irreps('1x0e + 1x1o + 1x2e')
    minimal_basis_size = (9, 9)
    irreps_out = o3.Irreps('1x0e')
    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)

    def __init__(self,
                 max_radius=1.0,
                 num_neighbors=10,
                 num_basis=10):

        # Information about creating a minimal basis representation.
        logging.info('Creating neural network model.')
        logging.info('Irrep of minimal basis: {}'.format(self.minimal_basis_irrep))
        logging.info('Size of minimal basis: {}'.format(self.minimal_basis_size))

        super().__init__()

        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_basis = num_basis

        # Define the tensor product that is needed
        # As of now, we will take the following expression
        # out = 1/sqrt(z) sum_{nearest neighbor} f_in * h(rel_pos) * Y(rel_pos)
        # where Y are the spherical harmonics, h is an MLP and f_in is the input feature.   
        self.tensor_product = o3.FullyConnectedTensorProduct(irreps_in1=self.minimal_basis_irrep,
                                                             irreps_in2=self.irreps_sh,
                                                             irreps_out=self.irreps_out, 
                                                             shared_weights=False)

        # Create a Fully Connected Tensor Product for the NN. The three integers of the
        # list in the first input are the dimensions of the input, intermediate and output. 
        # The second option is the activation function.
        self.fc_network = nn.FullyConnectedNet([self.num_basis, 16, self.tensor_product.weight_numel], torch.relu)
    
    def graph_generator(self, dataset):
        """Return the graph, including node and edge attributes, 
        for a given dataset."""
        
        # Iterate over the dataset to get the characteristics of the graph. 
        for data in dataset:
            logging.info('Processing new data graph.') 
            # STRATEGY: Generate a minimal basis representation where the
            # Hamiltonians (of both spin up and down) are condensed
            # into a single matrix consisting of an irreducible
            # representation of only 1s, 1p and 1d. Each molecule,
            # either reactant or product will have one such matrix
            # associated with it.
            # Store the minimal basis in the same keys as data['x'].
            minimal_basis = {} 
            edge_dst = []
            edge_src = []
            edge_vec = []

            for mol_index, data_dict in data['x'].items():
                # Within each molecule, the basis representation
                # of each atom must interact with the basis representation
                # of the other atoms. Therefore, we need to loop over all
                # pairs of atoms.
                minimal_basis_rep = torch.zeros(size=self.minimal_basis_size, dtype=torch.float)
                logging.info('Constructing minimal basis representation for molecule {}.'.format(mol_index))

                for i, atom_basis_1 in enumerate(data['indices_atom_basis'][mol_index]):
                    # Atoms can interact with themselves as well

                    for j, atom_basis_2 in enumerate(data['indices_atom_basis'][mol_index]):
                        # Perform tensor products to reduce all representations
                        # to the minimal basis representation.

                        logging.debug(f"Performing tensor product for {atom_basis_1} and {atom_basis_2}.")
                        irrep_in1 = o3.Irreps(data['irrep_atoms'][mol_index][i])
                        irrep_in2 = o3.Irreps(data['irrep_atoms'][mol_index][j])
                        tensor_product = o3.FullyConnectedTensorProduct(irreps_in1=irrep_in1,
                                                                        irreps_in2=irrep_in2,
                                                                        irreps_out=self.minimal_basis_irrep,
                        )

                        logging.debug('Irrep of input 1: {}'.format(irrep_in1))
                        logging.debug('Irrep of input 2: {}'.format(irrep_in2))
                        # Generate the fragements of the Hamiltonian for the two atoms.
                        fragment_H1 = data_dict[atom_basis_1[0]:atom_basis_1[1], atom_basis_1[0]:atom_basis_1[1], ...]
                        fragment_H2 = data_dict[atom_basis_2[0]:atom_basis_2[1], atom_basis_2[0]:atom_basis_2[1], ...]
                        logging.debug('Shape of fragment 1: {}'.format(fragment_H1.shape))
                        logging.debug('Shape of fragment 2: {}'.format(fragment_H2.shape))

                        # Average over the spin up and down components for both fragments
                        fragment_H1 = torch.mean(fragment_H1, dim=-1)
                        fragment_H2 = torch.mean(fragment_H2, dim=-1)
                        logging.debug('Shape of fragment 1 after taking mean of spin up and down: {}'.format(fragment_H1.shape))
                        logging.debug('Shape of fragment 2 after taking mean of spin up and down: {}'.format(fragment_H2.shape))

                        # Perform the tensor product to get the minimal basis representation.
                        logging.debug('Performing tensor product.')

                        # Make sure that the ensuing tensor product for this particular
                        # pair of atoms is equivariant
                        equivariance_error(tensor_product,
                                           args_in = [fragment_H1, fragment_H2],
                                           irreps_in = [irrep_in1, irrep_in2],
                                           irreps_out = self.minimal_basis_irrep
                                           )
                        logging.debug('Passed equivariance test.')

                        minimal_basis_tp = tensor_product(fragment_H1, fragment_H2)
                        logging.debug(f"Shape of tensor product {minimal_basis_tp.shape}")
                        # minimal_basis_rep += minimal_basis_tp

                # Store the minimal basis representation in the same key as the
                # Hamiltonian.
                minimal_basis[mol_index] = minimal_basis_rep

                # Generate the graph for the current dataset.
                # Most of the edge dependent quantities are already stored in datapoint.
                # We just have to modify them into the form that we require here.
                # Again: Every molecule is independent in terms of the graph, 
                # so all of these parameters can be independetly generated.
                edge_src_atom, edge_dst_atom = radius_graph(data['pos'], self.max_radius, max_num_neighbors=self.num_neighbors)
                edge_vec_atom = data['pos'][edge_dst_atom] - data['pos'][edge_src_atom]
                # Append the atom indices to the lists.
                edge_src.append(edge_src_atom)
                edge_dst.append(edge_dst_atom)
                edge_vec.append(edge_vec_atom)
            
            num_nodes = data['num_nodes']

            yield num_nodes, edge_src, edge_dst, edge_vec, minimal_basis 

    def forward(self, dataset): 
        """Forward pass of the model."""

        output_vector = []

        for num_nodes, edge_src, edge_dst, edge_vec, input_features in self.graph_generator(dataset):

            # Determine the spherical harmonics to perform the convolution
            # The spherical harmonics of the chosen degree will be performed
            # on the (normalised) edge vector.
            sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')

            # Also add an embedding for the MLP.
            embedding = soft_one_hot_linspace(x = edge_vec.norm(dim=1),
                                        start = 0.0,
                                        end = self.max_radius,
                                        number = self.num_basis,
                                        basis = 'smooth_finite',
                                        cutoff = True).mul(self.num_basis**0.5)


            # Construct the summation of the required dimension
            output = self.tensor_product( input_features[edge_src], sh, self.fc_network(embedding) )
            output = scatter(output, edge_dst, dim=0, dim_size=num_nodes).div(self.num_neighbors**0.5)

            # Append the summed output_vector to the list of output_vectors.
            # This operation corresponds to the summing of all atom-centered features.
            output_vector.append(output.sum())
        
        # Make the output_vector a tensor.
        output_vector = torch.tensor(output_vector, requires_grad=True)

        return output_vector

if __name__ == '__main__':
    """Test a convolutional Neural Network"""

    # Read in the dataset inputs.
    JSON_FILE = 'input_files/output_ts_calc_debug.json'
    BASIS_FILE = 'input_files/def2-tzvppd.json'
    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

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