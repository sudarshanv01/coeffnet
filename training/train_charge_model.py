import os
import logging
import datetime

import torch

import torch_geometric
from torch_geometric.transforms import RandomLinkSplit

from minimal_basis.utils import avail_checkpoint, visualize_results
from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel

from utils import get_test_data_path


if __name__ == "__main__":
    """Test a convolutional Neural Network based on the charge model."""

    LOGFILES_FOLDER = "log_files"
    logging.basicConfig(
        filename=os.path.join(LOGFILES_FOLDER, "charge_model.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    # Prefix tag to the output folders
    today = datetime.datetime.now()
    folder_string = "charge_" + today.strftime("%Y%m%d_%H%M%S")

    # Read in the dataset inputs.
    JSON_FILE = "input_files/output_QMrxn20_debug.json"
    CHECKPOINT_FOLDER = "charge_checkpoints"
    PLOT_FOLDER = f"plots/{folder_string}"
    GRAPH_GENERTION_METHOD = "sn2"

    # Create the folder if it does not exist.
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    # Get details of the checkpoint
    checkpoint_file = avail_checkpoint(CHECKPOINT_FOLDER)

    # Create the Charge dataset
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=JSON_FILE,
        graph_generation_method=GRAPH_GENERTION_METHOD,
    )
    dataset.process()

    logging.info("Dataset created.")
    logging.info(f"Number of datapoints: {dataset.len()}")

    # Instantiate the model.
    model = ChargeModel(out_channels=1)
    if torch.cuda.device_count() > 1:
        logging.info("Using multiple GPUs")
        raise NotImplementedError("Multiple GPUs not yet implemented")

    if checkpoint_file is not None:
        model.load_state_dict(torch.load(checkpoint_file))
        logger.info(f"Loaded checkpoint file: {checkpoint_file}")
        model.eval()
    else:
        logger.info("No checkpoint file found, starting from scratch")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Write header of log file
    with open(os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "w") as f:
        f.write("Epoch\t Loss\t AccuracyTrain\t AccuracyVal\n")

    train_y = []
    for data in dataset:
        train_y.append(data.y)
    train_y = torch.as_tensor(train_y, dtype=torch.float)

    for step in range(500):

        optim.zero_grad()
        pred = model(dataset)
        loss = (pred - train_y).pow(2).sum()

        loss.backward()

        accuracy_train = (pred - train_y).abs().sum() / len(train_y)

        # Check the validation dataset
        # predict_val_y = model(val_dataset)
        # accuracy_validation = (predict_val_y - val_y).abs().sum() / len(val_y)

        with open(
            os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "a"
        ) as f:
            f.write(
                f"{step:5d} \t {loss:<10.1f} \t {accuracy_train:5.1f}\t {accuracy_train:5.1f}\n"
            )

        if step % 10 == 0:

            # Plot the errors for each step.
            visualize_results(
                pred,
                train_y,
                torch.tensor([]),
                torch.tensor([]),
                PLOT_FOLDER,
                epoch=step,
                loss=loss,
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
