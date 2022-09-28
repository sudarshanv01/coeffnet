import os
import logging
import datetime

import torch

from minimal_basis.utils import avail_checkpoint, visualize_results
from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
from minimal_basis.model.model_hamiltonian import HamiltonianModel

if __name__ == "__main__":
    """Test a convolutional Neural Network"""

    LOGFILES_FOLDER = "log_files"
    logging.basicConfig(
        filename=os.path.join(LOGFILES_FOLDER, "model.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Determine split rations
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

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

    # Split the data into training, validation and test sets.
    num_train, num_val = int(train_ratio * len(datapoint)), int(
        val_ratio * len(datapoint)
    )
    num_test = len(datapoint) - num_train - num_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        datapoint,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Instantiate the model.
    model = HamiltonianModel(DEVICE)
    if torch.cuda.device_count() > 1:
        logging.info("Using multiple GPUs")
        raise NotImplementedError("Multiple GPUs not yet implemented")
    if checkpoint_file is not None:
        model.load_state_dict(torch.load(checkpoint_file))
        logger.info(f"Loaded checkpoint file: {checkpoint_file}")
        model.eval()
    else:
        logger.info("No checkpoint file found, starting from scratch")
    model = model.to(DEVICE)

    # Get the training y
    train_y = []
    for data in train_dataset:
        train_y.append(data.y)
    train_y = torch.tensor(train_y, dtype=torch.float)

    # Get the validation y
    val_y = []
    for data in val_dataset:
        val_y.append(data.y)
    val_y = torch.tensor(val_y, dtype=torch.float)

    # Get the test y
    test_y = []
    for data in test_dataset:
        test_y.append(data.y)
    test_y = torch.tensor(test_y, dtype=torch.float)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    logging.info(f"Test set size: {len(test_dataset)}")

    # Training the model.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Write header of log file
    with open(os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "w") as f:
        f.write("Epoch\t Loss\t AccuracyTrain\t AccuracyVal\n")

    for step in range(500):

        optim.zero_grad()
        pred = model(train_dataset)
        loss = (pred - train_y).pow(2).sum()

        loss.backward()

        accuracy_train = (pred - train_y).abs().sum() / len(train_y)

        # Check the validation dataset
        predict_val_y = model(val_dataset)
        accuracy_validation = (predict_val_y - val_y).abs().sum() / len(val_y)

        with open(
            os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "a"
        ) as f:
            f.write(
                f"{step:5d} \t {loss:<10.1f} \t {accuracy_train:5.1f}\t {accuracy_validation:5.1f}\n"
            )

        if step % 10 == 0:

            # Plot the errors for each step.
            visualize_results(
                pred, train_y, predict_val_y, val_y, PLOT_FOLDER, epoch=step, loss=loss
            )

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
