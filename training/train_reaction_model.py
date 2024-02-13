import logging
import os
from pathlib import Path

import torch
import torch_geometric.transforms as T
from coeffnet.dataset.reaction import ReactionDataset as Dataset
from e3nn import o3
from torch_geometric.loader import DataLoader

import wandb
from cli_functions import (
    create_timestamp,
    get_basis_set_name,
    get_command_line_arguments,
)
from model_functions import construct_model_name, train, validate
from utils import get_train_data_path, get_validation_data_path, read_inputs_yaml

logging.basicConfig(level=logging.INFO)
logging.getLogger("minimal_basis").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = get_command_line_arguments()
    basis_set_name = get_basis_set_name(args.basis_set)
    logger.info(f"Basis set name: {basis_set_name}")
    if args.loss_function is None:
        if args.prediction_mode == "coeff_matrix":
            args.loss_function = "coeffnet.loss.eigenvectors.UnsignedL1Loss"
        elif args.prediction_mode == "relative_energy":
            args.loss_function = "torch.nn.L1Loss"
    model_name = construct_model_name(
        dataset_name=args.dataset_name,
        debug=args.debug,
    )
    timestamp = create_timestamp()
    logger.info(f"Model name: {model_name}")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")
    inputs = read_inputs_yaml(os.path.join(args.model_config))
    input_foldername = (
        Path(args.input_folder)
        / args.dataset_name
        / args.basis_set_type
        / basis_set_name
    )
    dataset_options = inputs["dataset_options"][f"{args.basis_set_type}_basis"]
    if args.invert_coordinates:
        dataset_options["invert_coordinates"] = args.invert_coordinates
    learning_options = {}
    learning_options["batch_size"] = args.batch_size
    learning_options["learning_rate"] = args.learning_rate
    learning_options["num_epochs"] = args.num_epochs
    logger.info(f"Learning options: {learning_options}")
    model_options = inputs["model_options"][args.prediction_mode]
    if args.debug:
        wandb.init(project=model_name, entity=args.wandb_username, mode="dryrun")
    else:
        wandb.init(project=model_name, entity=args.wandb_username)
    wandb.config.update(args)
    wandb.config.update({"dataset_options": dataset_options})
    model_class = args.model_class
    nn_name = model_class.split(".")[-1]
    path_to_model = ".".join(model_class.split(".")[:-1])
    module_model = __import__(path_to_model, fromlist=[nn_name])
    Model = getattr(module_model, nn_name)
    logger.info(f"Model used: {Model}")
    loss_function = args.loss_function
    loss_class_name = loss_function.split(".")[-1]
    path_to_loss = ".".join(loss_function.split(".")[:-1])
    module_loss = __import__(path_to_loss, fromlist=[loss_class_name])
    Loss = getattr(module_loss, loss_class_name)
    logger.info(f"Loss function used: {Loss}")
    transform = T.ToDevice(DEVICE)
    if args.debug:
        train_json_filename = input_foldername / "train_debug.json"
        validate_json_filename = input_foldername / "validate_debug.json"
    else:
        train_json_filename = input_foldername / "train.json"
        validate_json_filename = input_foldername / "validate.json"
    train_dataset = Dataset(
        root=get_train_data_path(model_name + "_" + timestamp),
        filename=train_json_filename,
        transform=transform,
        **dataset_options,
    )
    validate_dataset = Dataset(
        root=get_validation_data_path(model_name + "_" + timestamp),
        filename=validate_json_filename,
        transform=transform,
        **dataset_options,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=learning_options["batch_size"], shuffle=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=learning_options["batch_size"], shuffle=False
    )
    model_options["irreps_node_attr"] = train_dataset.irreps_node_attr
    model_options["irreps_in"] = train_dataset.irreps_in
    model_options["irreps_out"] = train_dataset.irreps_out
    model_options["lmax"] = o3.Irreps(model_options["irreps_out"]).lmax
    model_options["mul"] = args.mul
    model_options["layers"] = args.layers
    model_options["max_radius"] = args.max_radius
    model_options["num_basis"] = args.num_basis
    model_options["radial_neurons"] = args.radial_neurons
    model_options["num_neighbors"] = args.num_neighbors
    logger.info(f"Model options: {model_options}")
    wandb.config.update({"model_options": model_options})
    model = Model(**model_options)
    model = model.to(DEVICE)
    logger.info(f"Model: {model}")
    wandb.watch(model)
    optim = torch.optim.Adam(model.parameters(), lr=learning_options["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=10, verbose=True
    )
    for epoch in range(1, learning_options["num_epochs"] + 1):
        train_loss = train(
            train_loader=train_loader,
            model=model,
            optim=optim,
            prediction_mode=args.prediction_mode,
            loss_function=Loss(reduction=args.reduction),
        )
        validate_loss = validate(
            val_loader=validate_loader,
            model=model,
            prediction_mode=args.prediction_mode,
            loss_function=Loss(reduction=args.reduction),
        )
        scheduler.step(validate_loss)
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": validate_loss})
        logger.info(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validate_loss}"
        )
    torch.save(model, f"output/{model_name}_{timestamp}.pt")
    artifact = wandb.Artifact(f"{model_name}", type="model")
    artifact.add_file(f"output/{model_name}_{timestamp}.pt")
    logger.debug(f"Added model to artifact: {artifact}.")
    wandb.run.log_artifact(artifact)
    logger.debug(f"Logged artifact: {artifact}.")
    wandb.finish()
