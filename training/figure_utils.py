from typing import List, Tuple, Dict, Union, Optional

from pathlib import Path

import pandas as pd

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from minimal_basis.dataset.reaction import ReactionDataset

from utils import (
    get_train_data_path,
    get_validation_data_path,
    get_test_data_path,
)

from model_functions import construct_model_name

import wandb

wandb_api = wandb.Api()


def get_sanitized_basis_set_name(basis_set: str) -> str:
    """Get the sanitized basis set name."""
    basis_set_name = basis_set.replace("*", "star")
    basis_set_name = basis_set_name.replace("+", "plus")
    basis_set_name = basis_set_name.replace("(", "")
    basis_set_name = basis_set_name.replace(")", "")
    basis_set_name = basis_set_name.replace(",", "")
    basis_set_name = basis_set_name.replace(" ", "_")
    basis_set_name = basis_set_name.lower()
    return basis_set_name


def get_dataloaders(
    input_foldername: Path,
    model_name: str,
    basis_set_type: str,
    basis_set_name: str,
    debug=False,
    **dataset_options,
) -> Dict[str, DataLoader]:
    """Get the dataloaders for the given model."""
    _debug_string = "_debug" if debug else ""
    train_json_filename = input_foldername / f"train{_debug_string}.json"
    validate_json_filename = input_foldername / f"validate{_debug_string}.json"

    train_dataset = ReactionDataset(
        root=get_train_data_path(
            model_name + "_" + basis_set_type + "_" + basis_set_name
        ),
        filename=train_json_filename,
        **dataset_options,
    )
    validate_dataset = ReactionDataset(
        root=get_validation_data_path(
            model_name + "_" + basis_set_type + "_" + basis_set_name
        ),
        filename=validate_json_filename,
        **dataset_options,
    )
    test_dataset = ReactionDataset(
        root=get_test_data_path(
            model_name + "_" + basis_set_type + "_" + basis_set_name
        ),
        filename=validate_json_filename,
        **dataset_options,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return {
        "train": train_loader,
        "validation": validate_loader,
        "test": test_loader,
    }


def get_model_data(
    basis_set_type: str, basis_set: str, dataset_name: str, debug=False
) -> Tuple[pd.DataFrame, List[wandb.apis.public.Run]]:
    """Get the model data from wandb."""
    model_name = construct_model_name(
        dataset_name=dataset_name,
        debug=debug,
    )
    runs = wandb_api.runs(f"sudarshanvj/{model_name}")
    df = pd.DataFrame()
    for run in runs:
        if run.state != "finished":
            continue
        if (
            run.config.get("basis_set_type") == basis_set_type
            and run.config.get("basis_set") == basis_set
        ):
            data_to_store = {}
            data_to_store.update(run.config)
            train_loss = run.summary.get("train_loss", None)
            val_loss = run.summary.get("val_loss", None)
            data_to_store.update({"train_loss": train_loss, "val_loss": val_loss})
            data_to_store.update({"wandb_model_name": run.name})
            df = pd.concat(
                [df, pd.DataFrame(data_to_store, index=[0])], ignore_index=True
            )
    return df, runs
