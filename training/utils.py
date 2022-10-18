import os


def create_folders(foldername):
    """Create a folder if it doesnt exist."""
    # Create the folder if it does not exist.
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def create_plot_folder():
    """Place where plots go."""
    # Prefix tag to the output folders
    today = datetime.datetime.now()
    folder_string = today.strftime("%Y%m%d_%H%M%S")
    PLOT_FOLDER = f"plots/{folder_string}"
    create_folders(PLOT_FOLDER)
    return folder_string, PLOT_FOLDER


def read_inputs_yaml(input_filename):
    """Read the inputs from the yaml file."""
    import yaml

    with open(input_filename, "r") as stream:
        try:
            inputs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return inputs


def check_no_of_gpus():
    """Check that there are GPUs are appropriate."""
    if torch.cuda.device_count() > 1:
        logging.info("Using multiple GPUs")
        raise NotImplementedError("Multiple GPUs not yet implemented")


def get_test_data_path():
    """The tests folder in the conftest file."""
    confpath = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(confpath, "test_data")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path, exist_ok=True)
    return test_data_path
