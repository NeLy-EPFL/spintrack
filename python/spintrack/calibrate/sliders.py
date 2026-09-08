"""Camera-to-animal rotation from where the camera sits around the animal.

An alternative to the calibration square when no square can be placed: describe the
camera position in the animal frame (x forward, y right, z down) with an azimuth (degrees
around the animal's vertical axis, 0 = in front of the animal, 90 = at its right), an
elevation (degrees above the animal's horizontal plane) and an optional twist (roll of the
camera about its optical axis, degrees, positive = clockwise in the image). The camera is
assumed to look at the ball center with its image-down axis as aligned with animal-down as
the geometry allows.
"""

from __future__ import annotations

import numpy as np

from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix


def camera_to_lab_from_angles(
    elevation_deg: float, azimuth_deg: float, twist_deg: float = 0.0
) -> np.ndarray:
    """Rotation matrix `R` with `v_lab = R @ v_camera`."""
    el, az, tw = np.radians([elevation_deg, azimuth_deg, twist_deg])
    position = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    z_cam = -normalize(position)  # optical axis points from the camera to the animal
    down = np.array([0.0, 0.0, 1.0])
    y_cam = down - (down @ z_cam) * z_cam
    if np.linalg.norm(y_cam) < 1e-9:  # camera straight above or below: use forward
        forward = np.array([1.0, 0.0, 0.0])
        y_cam = forward - (forward @ z_cam) * z_cam
    y_cam = normalize(y_cam)
    x_cam = np.cross(y_cam, z_cam)
    R = np.column_stack([x_cam, y_cam, z_cam])
    return R @ rotvec_to_matrix(np.array([0.0, 0.0, tw]))


def c2a_from_angles(
    elevation_deg: float, azimuth_deg: float, twist_deg=0.0
) -> list[float]:
    """FicTrac-style `c2a_r` (rotation vector of camera-to-lab) from the three angles."""
    return matrix_to_rotvec(
        camera_to_lab_from_angles(elevation_deg, azimuth_deg, twist_deg)
    ).tolist()
