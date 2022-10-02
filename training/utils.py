import os


def get_test_data_path():
    """The tests folder in the conftest file."""
    confpath = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(confpath, "test_data")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path, exist_ok=True)
    return test_data_path
