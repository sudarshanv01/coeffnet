import os
import logging

import argparse

import torch
from torch_geometric.loader import DataLoader

from minimal_basis.utils import avail_checkpoint, visualize_results
from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel

from utils import (
    get_test_data_path,
    read_inputs_yaml,
)

import torch.nn.functional as F

LOGFILES_FOLDER = "log_files"
logging.basicConfig(
    filename=os.path.join(LOGFILES_FOLDER, "charge_model.log"),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, the calculation is a DEBUG calculation.",
)
parser.add_argument(
    "--hidden_channels",
    type=int,
    default=64,
    help="Number of hidden channels in the neural network.",
)
args = parser.parse_args()


if __name__ == "__main__":
    """Test a Graph Convolutional Neural Network based on the charge model."""

    if args.debug:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {DEVICE}")

    inputs = read_inputs_yaml(os.path.join("input_files", "charge_model.yaml"))
    GRAPH_GENERATION_METHOD = inputs["graph_generation_method"]
    if args.debug:
        train_json_filename = inputs["debug_train_json"]
    else:
        train_json_filename = inputs["train_json"]

    BATCH_SIZE = inputs["batch_size"]

    # Create the training and test datasets
    train_dataset = ChargeDataset(
        root=get_test_data_path(),
        filename=train_json_filename,
        graph_generation_method=GRAPH_GENERATION_METHOD,
    )
    train_dataset.process()

    # Create a dataloader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create the optimizer
    model = ChargeModel(
        in_channels=train_dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=1,
    )
    model = model.to(DEVICE)
    # Create the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create the model
    for train_batch in train_loader:
        data = train_batch.to(DEVICE)
        optim.zero_grad()
        predicted_y = model(data)
        loss = F.mse_loss(predicted_y, data.y)
        loss.backward()
        optim.step()
