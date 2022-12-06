import os
import logging
import pickle
import argparse

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from ase import io as ase_io

from monty.serialization import loadfn, dumpfn

from minimal_basis.predata import GenerateParametersClassifier

logging.basicConfig(
    filename=os.path.join("check_pretrain_interpolate_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_transition_states(datapoint, model):
    """Get the model predicted transition states."""

    # Extract data from datapoint
    reactant_graph = datapoint["reactant_molecule_graph"]
    product_graph = datapoint["product_molecule_graph"]
    transition_state_graph = datapoint["transition_state_molecule_graph"]
    transition_state_coords = transition_state_graph.molecule.cart_coords
    reaction_energy = datapoint["reaction_energy"]

    # Get the interpolated transition states
    interpolated_transition_states = (
        paramclass.get_interpolated_transition_state_positions(
            is_positions=reactant_graph.molecule.cart_coords,
            fs_positions=product_graph.molecule.cart_coords,
            mu=args.mu,
            sigma=args.sigma,
            alpha=args.alpha,
            deltaG=reaction_energy,
        )
    )
    interpolated_transition_state_molecule = Molecule(
        transition_state_graph.molecule.species, interpolated_transition_states
    )

    # Get a list of the atomic numbers
    atomic_numbers = np.array(transition_state_graph.molecule.atomic_numbers)

    # The features to the Kernel ridge model is simply the interpolated
    # transitions state coords and the atom numbers of the species
    # of the transition state
    input_features = np.concatenate(
        (interpolated_transition_states, atomic_numbers.reshape(-1, 1)), axis=1
    )

    # Get the predicted transition states
    predicted_transition_states = model.predict(input_features)
    # Create a molecule out of the predicted transition states
    predicted_transition_state_molecule = Molecule(
        transition_state_graph.molecule.species,
        predicted_transition_states,
    )

    return (
        interpolated_transition_state_molecule,
        predicted_transition_state_molecule,
        transition_state_graph.molecule,
    )


def get_error(predicted, actual):
    """Get the mean of the norm of the difference between the predicted and actual transition states."""
    return np.mean(np.linalg.norm(predicted.cart_coords - actual.cart_coords, axis=1))


def get_cli():
    """Get the hyperparameters from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Hyperparamer for the number of samples to generate",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Hyperparameter alpha")
    parser.add_argument("--mu", type=float, default=0.5, help="Hyperparameter mu")
    parser.add_argument("--sigma", type=float, default=2, help="Hyperparameter sigma")
    return parser.parse_args()


if __name__ == "__main__":
    """Test the kernel ridge model on the validation set."""

    args = get_cli()
    logger.debug("Hyperparameters: {}".format(args))

    # Read in the training data
    train_data = loadfn("input_files/train_RAPTER.json")
    validate_data = loadfn("input_files/validate_RAPTER.json")

    # Initialize the class
    paramclass = GenerateParametersClassifier(num_samples=args.num_samples)

    # Read in the kernel ridge model
    with open("output/kernel_ridge_model.pickle", "rb") as f:
        model = pickle.load(f)

    output_dir = os.path.join("output", "interpolated_transition_states")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # --- Training set errors
    total_train_error = 0
    for idx, datapoint in enumerate(train_data):
        interpolated, predicted, actual = get_transition_states(datapoint, model)
        error_train = get_error(predicted, actual)
        error_interpolated = get_error(interpolated, actual)
        total_train_error += error_train
        logger.debug(
            "Training set error interpolated:{} krr:{} ".format(
                error_interpolated, error_train
            )
        )

        # Write out the predicted and actual transition states
        interpolated_atoms = AseAtomsAdaptor.get_atoms(interpolated)
        predicted_atoms = AseAtomsAdaptor.get_atoms(predicted)
        actual_atoms = AseAtomsAdaptor.get_atoms(actual)
        label = f"train_{idx}_{error_train:.2f}.xyz"

        ase_io.write(
            os.path.join(output_dir, label),
            [interpolated_atoms, predicted_atoms, actual_atoms],
        )
    average_train_error = total_train_error / len(train_data)
    logger.info("Average training set error: {}".format(average_train_error))

    # --- Validation set errors
    total_validate_error = 0
    for idx, datapoint in enumerate(validate_data):
        interpolated, predicted, actual = get_transition_states(datapoint, model)
        error_validate = get_error(predicted, actual)
        total_validate_error += error_validate
        error_interpolated = get_error(interpolated, actual)
        logger.debug(
            "Training set error interpolated:{} krr:{} ".format(
                error_interpolated, error_validate
            )
        )

        # Write out the predicted and actual transition states
        interpolated_atoms = AseAtomsAdaptor.get_atoms(interpolated)
        predicted_atoms = AseAtomsAdaptor.get_atoms(predicted)
        actual_atoms = AseAtomsAdaptor.get_atoms(actual)
        label = f"validate_{idx}_{error_validate:.2f}.xyz"

        ase_io.write(
            os.path.join(output_dir, label),
            [interpolated_atoms, predicted_atoms, actual_atoms],
        )
    average_validate_error = total_validate_error / len(validate_data)
    logger.info("Average validation set error: {}".format(average_validate_error))
