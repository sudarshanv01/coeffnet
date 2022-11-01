import numpy as np

import torch

from e3nn import o3

from conftest import sn2_reaction_input, get_basis_file_info

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset

from minimal_basis.model.model_hamiltonian import EquivariantNodeConv


def test_hamiltonian_dataset_sn2_graph(sn2_reaction_input, tmp_path):
    """Test that the Hamiltonian dataset."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()
    assert dataset.input_data is not None
    assert dataset.basis_info is not None


def test_equivariance_equivariantnodeconv(sn2_reaction_input, tmp_path):
    """Test the equivariance of the EquivariantNodeConv layer."""

    filename = sn2_reaction_input
    basis_file = get_basis_file_info()

    # Create the Hamiltonian dataset.
    GRAPH_GENERTION_METHOD = "sn2"
    dataset = HamiltonianDataset(
        root=tmp_path,
        filename=filename,
        basis_file=basis_file,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )

    dataset.process()
    # Analyse a single datapoint
    datapoint = dataset[0]
    node_features = datapoint.x

    num_nodes = node_features.shape[0]

    # Define minimal basis irreps
    irreps_input = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_output = o3.Irreps("1x0e + 1x1o + 1x2e")

    rot = o3.rand_matrix()
    D_in = irreps_input.D_from_matrix(rot)
    D_out = irreps_output.D_from_matrix(rot)

    # Perform the same operation for each node
    D_in = D_in.repeat(num_nodes, 1, 1)
    D_out = D_out.repeat(num_nodes, 1, 1)

    # Create a convolution input
    conv = EquivariantNodeConv(
        irreps_in=irreps_input,
        irreps_out=irreps_output,
        num_basis=10,
    )

    # Reshape the node features to make a square symmetric matrix
    dimensions_x_symm = np.sqrt(node_features.shape[1] / 2).astype(int)
    f_in = node_features.reshape(num_nodes, dimensions_x_symm, dimensions_x_symm, 2)
    pos = datapoint.pos
    max_radius = 10

    # Rotate features
    f_in_rotate = torch.zeros_like(f_in)
    f_in_rotate[..., 0] = f_in[..., 0] @ D_in
    f_in_rotate[..., 1] = f_in[..., 1] @ D_in
    pos_rotate = pos @ rot

    # rotate before
    f_before = conv(
        f_in_rotate, datapoint.edge_index, pos_rotate, max_radius, num_nodes
    )

    # rotate after
    f_after = conv(f_in, datapoint.edge_index, pos, max_radius, num_nodes) @ D_out.T

    torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)
