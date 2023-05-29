import pytest

from pathlib import Path

from monty.serialization import loadfn

from minimal_basis.model.reaction import ReactionModel


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
def model_options_factory():
    """Returns the model options common to all tests."""

    def _model_options_factory(prediction_mode, **kwargs):
        options = {
            "irreps_in": "5x0e+3x1o",
            "irreps_hidden": "64x0e+288x1o",
            "irreps_out": "5x0e+3x1o",
            "irreps_node_attr": "1x0e",
            "irreps_edge_attr": "12x0e",
            "radial_layers": 2,
            "radial_neurons": 64,
            "max_radius": 5,
            "num_basis": 12,
            "num_neighbors": 4,
            "typical_number_of_nodes": 12,
        }
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

    def _network_factory(prediction_mode, **kwargs):
        options = model_options_factory(prediction_mode, **kwargs)
        return ReactionModel(**options)

    return _network_factory


@pytest.fixture
def get_mapping_idx_to_euler_angles():
    """Read in the idx to euler angles mapping."""
    filename = Path(__file__).parent / "input" / "idx_to_euler_angles.yaml"
    return loadfn(filename)
