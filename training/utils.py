import os
import datetime

import numpy as np

import torch
import logging

logger = logging.getLogger(__name__)


def create_folders(foldername):
    """Create a folder if it doesnt exist."""
    # Create the folder if it does not exist.
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def create_plot_folder():
    """Place where plots go."""
    # Prefix tag to the output folders
    today = datetime.datetime.now()
    folder_string = today.strftime("%Y%m%d_%H%M%S")
    PLOT_FOLDER = f"plots/{folder_string}"
    create_folders(PLOT_FOLDER)
    return folder_string, PLOT_FOLDER


def read_inputs_yaml(input_filename):
    """Read the inputs from the yaml file."""
    import yaml

    with open(input_filename, "r") as stream:
        try:
            inputs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return inputs


def check_no_of_gpus():
    """Check that there are GPUs are appropriate."""
    if torch.cuda.device_count() > 1:
        logging.info("Using multiple GPUs")
        raise NotImplementedError("Multiple GPUs not yet implemented")


def get_test_data_path():
    """The tests folder in the conftest file."""
    confpath = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(confpath, "datasets_chkpoint", "test_data")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path, exist_ok=True)
    return test_data_path


def get_train_data_path():
    """The trains folder in the conftrain file."""
    confpath = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(confpath, "datasets_chkpoint", "train_data")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path, exist_ok=True)
    return train_data_path


def get_validation_data_path():
    """The validations folder in the confvalidation file."""
    confpath = os.path.dirname(os.path.abspath(__file__))
    validation_data_path = os.path.join(
        confpath, "datasets_chkpoint", "validation_data"
    )
    if not os.path.exists(validation_data_path):
        os.makedirs(validation_data_path, exist_ok=True)
    return validation_data_path


def rotate_three_dimensions(alpha, beta, gamma):
    """Rotate the molecule by arbitrary angles alpha
    beta and gamma."""
    cos = np.cos
    sin = np.sin

    r_matrix = [
        [
            cos(alpha) * cos(beta),
            cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
            cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
        ],
        [
            sin(alpha) * cos(beta),
            sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
            sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
        ],
        [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)],
    ]

    r_matrix = np.array(r_matrix)

    return r_matrix


def subdiagonalize_matrix(indices, matrix_H, matrix_S):
    """Subdiagonalise the matrix."""
    sub_matrix_H = matrix_H.take(indices, axis=0).take(indices, axis=1)
    sub_matrix_S = matrix_S.take(indices, axis=0).take(indices, axis=1)

    eigenval, eigenvec = np.linalg.eig(np.linalg.solve(sub_matrix_S, sub_matrix_H))

    # Normalise the eigenvectors
    for col in eigenvec.T:
        col /= np.sqrt(np.dot(col.conj(), np.dot(sub_matrix_S, col)))

    t_matrix = np.identity(matrix_H.shape[0])

    for i in range(len(indices)):
        for j in range(len(indices)):
            t_matrix[indices[i], indices[j]] = eigenvec[i, j]

    # Unitary transform to get the rearranged Hamiltonian and overlap
    H_r = np.dot(np.transpose(np.conj(t_matrix)), np.dot(matrix_H, t_matrix))
    S_r = np.dot(np.transpose(np.conj(t_matrix)), np.dot(matrix_S, t_matrix))

    return H_r, S_r, eigenval
