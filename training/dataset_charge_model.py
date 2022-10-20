import os
import logging

import torch

from torch_geometric.loader import DataLoader

from minimal_basis.utils import avail_checkpoint, visualize_results
from minimal_basis.dataset.dataset_charges import ChargeDataset
from minimal_basis.model.model_charges import ChargeModel

from utils import (
    get_test_data_path,
    read_inputs_yaml,
    create_plot_folder,
    create_folders,
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
    train_json_filename = inputs["train_json"]
    validate_json_filename = inputs["validate_json"]
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

    for train_batch in train_loader:
        print(train_batch)
        print(train_batch.num_features)
        print(train_batch.num_graphs)
