import pytest

from pathlib import Path

from monty.serialization import loadfn


@pytest.fixture
def get_dataset_options(tmp_path):
    """Returns dataset options common to all tests."""
    basis_filename = Path(__file__).parent / "input" / "basis.json"
    filename = Path(__file__).parent / "input" / "reaction.json"
    root = tmp_path / "root"
    return {
        "filename": filename,
        "basis_filename": basis_filename,
        "root": root,
        "max_s_functions": 5,
        "max_p_functions": 3,
        "max_d_functions": 0,
        "idx_eigenvalue": 0,
        "use_minimal_basis_node_features": False,
        "reactant_tag": "reactant",
        "product_tag": "product",
        "transition_state_tag": "transition_state",
        "is_minimal_basis": True,
        "calculated_using_cartesian_basis": True,
    }


@pytest.fixture
def get_mapping_idx_to_euler_angles():
    """Read in the idx to euler angles mapping."""
    filename = Path(__file__).parent / "input" / "idx_to_euler_angles.yaml"
    return loadfn(filename)
