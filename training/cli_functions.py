import argparse

__input_folder__ = "input"


def get_basis_set_name(basis_set):
    """Sanitize basis set name for use in file paths."""

    basis_set_name = basis_set.replace("*", "star")
    basis_set_name = basis_set_name.replace("+", "plus")
    basis_set_name = basis_set_name.replace("(", "")
    basis_set_name = basis_set_name.replace(")", "")
    basis_set_name = basis_set_name.replace(",", "")
    basis_set_name = basis_set_name.replace(" ", "_")
    basis_set_name = basis_set_name.lower()

    return basis_set_name


def get_command_line_arguments() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, the calculation is a DEBUG calculation.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Name of the folder to store the input files.",
        default=__input_folder__,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to use.",
        default="reaction",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/rudorff_lilienfeld_sn2_dataset.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="coeff_matrix",
        help="Mode of prediction. Can be either 'coeff_matrix' or 'relative_energy'.",
    )
    parser.add_argument(
        "--wandb_username",
        type=str,
        default="sudarshanvj",
    )
    parser.add_argument(
        "--basis_set_type",
        type=str,
        default="full",
        help="Type of basis set. Can be either 'full' or 'minimal'.",
    )
    parser.add_argument(
        "--basis_set",
        type=str,
        default="6-31g*",
        help="Name of the basis set to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=250,
        help="Maximum number of epochs to train for.",
    )

    args = parser.parse_args()

    return args
