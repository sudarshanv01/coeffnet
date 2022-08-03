import logging
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt

from dataset import HamiltonianDataset


class TSBarrierModel(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, max_radius=1.0, num_neighbors=10, num_basis=10):
        """Create a neural network model to predict 
        the transition state energy node and edge attributes
        taken from matrices."""
        super().__init__()
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_basis = num_basis
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = irreps_sh

        # Define the tensor product that is needed
        # As of now, we will take the following expression
        # out = 1/sqrt(z) sum_{nearest neighbor} f_in * h(rel_pos) * Y(rel_pos)
        # where Y are the spherical harmonics, h is an MLP and f_in is the input feature.   
        self.tensor_product = o3.FullyConnectedTensorProduct(irreps_in1=irreps_in,
                                                             irreps_in2=irreps_sh,
                                                             irreps_out=irreps_out, 
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
            
            # Get the positions and input_features
            positions = data.pos
            input_features = data.x

            # The number of nodes are equivalent to the number of atoms.
            num_nodes = len(positions)

            # The number of edges is equivalent to the number of bonds.
            edge_src, edge_dst = radius_graph(positions, self.max_radius, max_num_neighbors=self.num_neighbors)
            edge_vec = positions[edge_dst] - positions[edge_src]

            yield num_nodes, edge_src, edge_dst, edge_vec, input_features


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
    JSON_FILE = 'input_files/predict_data_ML.json'
    BASIS_FILE = 'input_files/basis_info.json'
    BASIS_SET = 'def2-mSVP'
    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

    data_point_obj = HamiltonianDataset(JSON_FILE, BASIS_FILE, BASIS_SET)
    data_point_obj.load_data()
    data_point_obj.validate_json_input()
    data_point = data_point_obj.get_data()

    # For now we will keep the input and output irreps as going from
    # 1s, 1p and 1d function to a 1s (single float), which is the energy.
    # We will arbitrarily also choose an spherical harmonic of l=2.
    input_irrep = o3.Irreps('1x0e + 1x1o + 1x2e')
    output_irrep = o3.Irreps('1x0e')
    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)

    # Instantiate the model.
    model = TSBarrierModel(input_irrep, output_irrep, irreps_sh)

    # Get the training y
    train_y = []
    for data in data_point:
        train_y.append(data.y)
    
    # Make train_y a tensor.
    train_y = torch.tensor(train_y, dtype=torch.float)

    # Training the model.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(300):
        optim.zero_grad()
        pred = model(data_point)
        loss = (pred - train_y).pow(2).sum()
        loss.backward()

        optim.step()

        if step % 10 == 0:
            accuracy = (pred - train_y).abs().sum() 
            print(f"epoch {step:5d} | loss {loss:<10.1f} | {accuracy:5.1f} accuracy")