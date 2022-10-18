import os
import logging
import datetime

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader

from minimal_basis.utils import avail_checkpoint, visualize_results
from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel

from utils import (
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
    create_folders,
    check_no_of_gpus,
)

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "charge_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Test a convolutional Neural Network based on the charge model."""

    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    logging.info(f"Device: {DEVICE}")

    folder_string, PLOT_FOLDER = create_plot_folder()

    inputs = read_inputs_yaml(os.path.join("input_files", "charge_model.yaml"))
    CHECKPOINT_FOLDER = inputs["checkpoint_folder"]
    create_folders(CHECKPOINT_FOLDER)
    checkpoint_file = avail_checkpoint(CHECKPOINT_FOLDER)

    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    INPUT_JSON_FILENAME = inputs["input_json_filename"]
    BATCH_SIZE = inputs["batch_size"]

    # Create the Charge dataset
    dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=INPUT_JSON_FILENAME,
        graph_generation_method=GRAPH_GENERATION_METHOD,
    )
    dataset.process()
    dataset.shuffle()

    # Instantiate the model.
    model = ChargeModel()
    model.to(DEVICE)
    check_no_of_gpus()

    if checkpoint_file is not None:
        model.load_state_dict(torch.load(checkpoint_file))
        logger.info(f"Loaded checkpoint file: {checkpoint_file}")
        model.eval()
    else:
        logger.info("No checkpoint file found, starting from scratch")

    # Decide the optimizer details
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Write header of log file
    with open(os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "w") as f:
        f.write("Epoch\t Loss\t AccuracyTrain\t AccuracyVal\n")

    # Begin training the model.
    model.train()

    for epoch in range(20):

        optim.zero_grad()

        pred = model(dataset)
        loss = (pred - dataset.data.y).pow(2).mean()

        loss.backward()

        with open(
            os.path.join(LOGFILES_FOLDER, f"training_{folder_string}.log"), "a"
        ) as f:
            f.write(f"{epoch:5d} \t {loss:<10.1f} \t {loss:<10.1f}\n")

            # Save the model params.
            logger.debug(f"Saving model parameters for step {epoch}")
            for param_tensor in model.state_dict():
                logger.debug(param_tensor)
                logger.debug(model.state_dict()[param_tensor].size())
            for name, param in model.named_parameters():
                logger.debug(name)
                logger.debug(param.grad)
            torch.save(model.state_dict(), f"{CHECKPOINT_FOLDER}/step_{epoch:03d}.pt")

        optim.step()
