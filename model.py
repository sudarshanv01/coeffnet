"""Train the model."""

import logging
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt

from dataset import HamiltonianDataset

def conv(f_in, pos, max_radius=1.0, num_neighbors=10):
    num_nodes = len(pos)
    edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos) - 1)
    edge_vec = pos[edge_dst] - pos[edge_src]
    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
    emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, basis='smooth_finite', cutoff=True).mul(num_basis**0.5)
    return scatter(tp(f_in[edge_src], sh, fc(emb)), edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)

if __name__ == '__main__':
    """Create a simple convolutional neural network."""

    JSON_FILE = 'input_files/predict_data_ML.json'
    BASIS_FILE = 'input_files/basis_info.json'
    BASIS_SET = 'def2-mSVP'
    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

    data_point_obj = HamiltonianDataset(JSON_FILE, BASIS_FILE, BASIS_SET)
    data_point_obj.load_data()
    data_point_obj.validate_json_input()
    data_point = data_point_obj.get_data()

    # Store the input and output irreps
    input_irrep = o3.Irreps('1x0e + 1x1o + 1x2e')
    output_irrep = o3.Irreps('1x0e') # At the moment, only TS energies are predicted

    # Test the equivariance
    rot = o3.rand_matrix()
    D_in = input_irrep.D_from_matrix(rot)
    D_out = output_irrep.D_from_matrix(rot)
    num_basis=10

    for data in data_point:
        # get the node features
        pos = data.pos
        f_in = data.x

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        tp = o3.FullyConnectedTensorProduct(input_irrep, irreps_sh, output_irrep, shared_weights=False)
        fc = nn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)

        # rotate before
        f_before = conv(f_in @ D_in.T, pos @ rot.T)
        print(f_before)

        # rotate after
        f_after = conv(f_in, pos) @ D_out.T
        print(f_after)

        torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)
        print('Test passed')

    

