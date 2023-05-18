import numpy as np
import numpy.typing as npt


class RotationMatrix:
    def __init__(self, angle_type: str, angles: npt.ArrayLike) -> None:
        """Create a rotation matrix based on the specified angle types and angles."""
        self.angle_type = angle_type
        self.angles = angles

    def __call__(self):
        """Return the rotation matrix."""
        if self.angle_type.lower() == "euler":
            return self.euler_rotation_matrix()
        elif self.angle_type.lower() == "tait-bryan":
            return self.tait_bryan_rotation_matrix()
        else:
            raise ValueError("Angle type not recognized.")

    def euler_rotation_matrix(self) -> np.ndarray:
        """Return the rotation matrix for the Euler angles."""
        alpha, beta, gamma = self.angles

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0, 0] = np.cos(beta) * np.cos(gamma)
        rotation_matrix[0, 1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(
            alpha
        ) * np.sin(gamma)
        rotation_matrix[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(
            alpha
        ) * np.sin(gamma)
        rotation_matrix[1, 0] = np.cos(beta) * np.sin(gamma)
        rotation_matrix[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(
            alpha
        ) * np.cos(gamma)
        rotation_matrix[1, 2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(
            alpha
        ) * np.cos(gamma)
        rotation_matrix[2, 0] = -np.sin(beta)
        rotation_matrix[2, 1] = np.sin(alpha) * np.cos(beta)
        rotation_matrix[2, 2] = np.cos(alpha) * np.cos(beta)

        return rotation_matrix
