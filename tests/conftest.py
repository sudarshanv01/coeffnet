import json
from pathlib import Path
from typing import Sequence, Union

import mongomock
import pytest
from monty.serialization import loadfn

from coeffnet.model.reaction import GateReactionModel


@pytest.fixture
def dataset_options_factory(tmp_path):
    """Returns dataset options common to all tests."""

    def _dataset_options_factory(
        basis_type: str = "full", idx_eigenvalue: Union[int, Sequence] = 0, **kwargs
    ):
        options = {
            "filename": Path(__file__).parent
            / "input"
            / f"reaction_{basis_type}_basis.json",
            "root": tmp_path / "root",
            "idx_eigenvalue": idx_eigenvalue,
            "reactant_tag": "reactant",
            "product_tag": "product",
            "transition_state_tag": "transition_state",
        }
        options.update(kwargs)
        return options

    return _dataset_options_factory


@pytest.fixture
def model_options_factory():
    """Returns the model options common to all tests."""

    def _model_options_factory(
        prediction_mode: str = "relative_energy", basis_type: str = "full", **kwargs
    ):
        options = {
            "irreps_node_attr": "5x0e",
            "radial_layers": 2,
            "radial_neurons": 64,
            "max_radius": 5,
            "num_basis": 12,
            "num_neighbors": 4,
            "typical_number_of_nodes": 12,
            "mul": 100,
        }

        if basis_type == "full":
            options.update(
                {
                    "irreps_in": "4x0e+3x1o+1x2e",
                    "irreps_out": "4x0e+3x1o+1x2e",
                }
            )
        elif basis_type == "minimal":
            options.update(
                {
                    "irreps_in": "4x0e+3x1o",
                    "irreps_out": "4x0e+3x1o",
                }
            )
        else:
            raise ValueError(f"Unknown basis type {basis_type}")

        if prediction_mode == "relative_energy":
            options.update(
                {
                    "reduce_output": True,
                    "reference_reduced_output_to_initial_state": True,
                    "make_absolute": False,
                    "mask_extra_basis": True,
                    "normalize_sumsq": True,
                }
            )
        elif prediction_mode == "coeff_matrix":
            options.update(
                {
                    "reduce_output": False,
                    "reference_reduced_output_to_initial_state": False,
                    "make_absolute": False,
                    "mask_extra_basis": True,
                    "normalize_sumsq": True,
                }
            )

        options.update(kwargs)

        return options

    return _model_options_factory


@pytest.fixture
def network_factory(model_options_factory):
    """Returns the network for different prediction modes."""

    def _network_factory(
        prediction_mode: str = "relative_energy", basis_type="full", **kwargs
    ):
        options = model_options_factory(prediction_mode, basis_type, **kwargs)
        return GateReactionModel(**options)

    return _network_factory


@pytest.fixture
def get_mapping_idx_to_euler_angles():
    """Read in the idx to euler angles mapping."""
    filename = Path(__file__).parent / "input" / "idx_to_euler_angles.yaml"
    return loadfn(filename)


@pytest.fixture
def mock_database():
    """Mock a MongoDatabase for the purposes of testing."""
    documents_file = Path(__file__).parent / "input" / "collection.json"
    db = mongomock.MongoClient().db
    collection = db.collection
    documents_from_file = json.loads(documents_file.read_text())
    for document in documents_from_file:
        collection.insert_one(document)
    return db


@pytest.fixture
def dataset_config_factory():
    """Return the dataset configuration."""

    def _dataset_config_factory(basis_type: str = "full"):
        filename = Path(__file__).parent / "input" / "dataset.yaml"
        config = loadfn(filename)
        if basis_type == "minimal":
            config["basis_set_type"] = "minimal"
        elif basis_type == "full":
            config["basis_set_type"] = "full"
        else:
            raise ValueError(f"Unknown basis type {basis_type}")
        return config

    return _dataset_config_factory


@pytest.fixture
def basis_raw():
    """Get the basis raw data."""
    filename = Path(__file__).parent / "input" / "basis.json"
    return loadfn(filename)
