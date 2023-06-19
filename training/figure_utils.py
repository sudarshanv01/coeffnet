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

from cli_functions import create_timestamp

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


def get_dataloader_info(
    input_foldername: Path,
    model_name: str,
    debug=False,
    device: Optional[torch.device] = None,
    **dataset_options,
) -> Dict[str, DataLoader]:
    """Get the dataloaders for the given model."""
    _debug_string = "_debug" if debug else ""
    train_json_filename = input_foldername / f"train{_debug_string}.json"
    validate_json_filename = input_foldername / f"validate{_debug_string}.json"

    timestamp = create_timestamp()

    transform = T.ToDevice(device)

    train_dataset = ReactionDataset(
        root=get_train_data_path(model_name + "_" + timestamp),
        filename=train_json_filename,
        transform=transform,
        **dataset_options,
    )
    validate_dataset = ReactionDataset(
        root=get_validation_data_path(model_name + "_" + timestamp),
        filename=validate_json_filename,
        transform=transform,
        **dataset_options,
    )
    test_dataset = ReactionDataset(
        root=get_test_data_path(model_name + "_" + timestamp),
        filename=validate_json_filename,
        transform=transform,
        **dataset_options,
    )
    max_s, max_p, max_d, max_f, max_g = (
        train_dataset.max_s_functions,
        train_dataset.max_p_functions,
        train_dataset.max_d_functions,
        train_dataset.max_f_functions,
        train_dataset.max_g_functions,
    )
    max_s = max_s
    max_p = max_p * 3
    max_d = max_d * 5
    max_f = max_f * 7
    max_g = max_g * 9

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return [
        {
            "train": train_loader,
            "validation": validate_loader,
            "test": test_loader,
        },
        {
            "max_s": max_s,
            "max_p": max_p,
            "max_d": max_d,
            "max_f": max_f,
            "max_g": max_g,
        },
    ]


def get_model_data(
    basis_set_type: str,
    basis_set: str,
    dataset_name: str,
    debug=False,
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


def get_best_model(
    prediction_mode: str,
    basis_set: str,
    basis_set_type: str,
    df: pd.DataFrame,
    all_runs,
    device: torch.device,
) -> torch.nn.Module:
    """Get the best model for the given prediction mode."""
    df_options = df[
        (df["basis_set"] == basis_set)
        & (df["basis_set_type"] == basis_set_type)
        & (df["prediction_mode"] == prediction_mode)
    ]
    df_options = df_options[~df_options["val_loss"].isna()]
    df_options["val_loss"] = df_options["val_loss"].astype(float)
    while True:
        best_model_row = df_options.sort_values(by="val_loss").iloc[0]
        best_run = [
            run for run in all_runs if run.name == best_model_row["wandb_model_name"]
        ][0]
        # print(f"Best model: {best_run.name}")
        best_artifacts = best_run.logged_artifacts()
        best_model = [
            artifact for artifact in best_artifacts if artifact.type == "model"
        ][0]
        best_model.download()
        try:
            best_model = torch.load(best_model.file())
        except RuntimeError:
            print("Failed to load model, skipping")
            df_options = df_options[
                df_options["val_loss"] != best_model_row["val_loss"]
            ]
            continue
        break
    best_model.eval()
    best_model.to(device)
    return best_model
