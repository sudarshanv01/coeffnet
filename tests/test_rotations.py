import pytest

import numpy as np

from scipy.spatial.transform import Rotation as R

from minimal_basis.transforms.rotations import RotationMatrix

"""Check the rotation matrices against those from scipy."""

radian_angles = [
    (0.0, 0.0, 0.0),
    (np.pi / 2, 0.0, 0.0),
    (0.0, np.pi / 2, 0.0),
    (0.0, 0.0, np.pi / 2),
    (np.pi / 2, np.pi / 2, np.pi / 2),
    (np.pi / 4, np.pi / 4, np.pi / 4),
    (np.pi / 4, np.pi / 4, np.pi / 2),
    (np.pi / 4, np.pi / 2, np.pi / 4),
    (np.pi / 2, np.pi / 4, np.pi / 4),
    (np.pi / 4, np.pi / 2, np.pi / 2),
    (np.pi / 2, np.pi / 4, np.pi / 2),
    (np.pi / 2, np.pi / 2, np.pi / 4),
    (np.pi / 2, np.pi / 2, np.pi / 2),
]


@pytest.mark.parametrize("alpha, beta, gamma", radian_angles)
def test_euler_rotation_matrix(alpha, beta, gamma):
    """Test the Euler rotation matrix."""
    angles = np.array([alpha, beta, gamma])
    rotation_matrix = RotationMatrix("euler", angles)()
    scipy_rotation = R.from_euler("xyz", angles, degrees=False).as_matrix()

    assert np.allclose(rotation_matrix, scipy_rotation)
