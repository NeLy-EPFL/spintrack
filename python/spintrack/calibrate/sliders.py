"""Camera-to-animal rotation from where the camera sits around the animal.

An alternative to the calibration square when no square can be placed: describe the
camera position in the animal frame (x forward, y left, z up) with an azimuth (degrees
around the animal's vertical axis, 0 = in front of the animal, 90 = at its right), an
elevation (degrees above the animal's horizontal plane) and an optional twist (roll of
the camera about its optical axis, degrees, positive = clockwise in the image). The
camera is assumed to look at the ball center with its image-down axis as aligned with
animal-down as the geometry allows.
"""

from __future__ import annotations

import numpy as np

from spintrack.geometry import normalize, rotvec_to_matrix

# FicTrac's animal frame (x forward, y right, z down) to this one: a half turn about x.
FICTRAC_TO_ANIMAL = np.diag([1.0, -1.0, -1.0])


def camera_to_lab_from_angles(
    elevation_deg: float, azimuth_deg: float, twist_deg: float = 0.0
) -> np.ndarray:
    """Rotation matrix `R` with `v_lab = R @ v_camera`."""
    el, az, tw = np.radians([elevation_deg, azimuth_deg, twist_deg])
    position = np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    z_cam = -normalize(position)  # optical axis points from the camera to the animal
    down = np.array([0.0, 0.0, -1.0])
    y_cam = down - (down @ z_cam) * z_cam
    if np.linalg.norm(y_cam) < 1e-9:  # camera straight above or below: use forward
        forward = np.array([1.0, 0.0, 0.0])
        y_cam = forward - (forward @ z_cam) * z_cam
    y_cam = normalize(y_cam)
    x_cam = np.cross(y_cam, z_cam)
    R = np.column_stack([x_cam, y_cam, z_cam])
    return R @ rotvec_to_matrix(np.array([0.0, 0.0, tw]))


def angles_from_camera_to_lab(R: np.ndarray) -> tuple[float, float, float]:
    """The elevation, azimuth and twist (degrees) that `camera_to_lab_from_angles` turns
    into `R`, for any rotation whose optical axis is not vertical.

    The angles then describe where the optical axis points, which is where the camera
    sits only when it looks at the ball's center.
    """
    R = np.asarray(R, dtype=np.float64)
    # The optical axis, R[:, 2], points from the camera's position toward the animal.
    elevation = np.degrees(np.arcsin(np.clip(-R[2, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(R[1, 2], -R[0, 2]))
    level = camera_to_lab_from_angles(elevation, azimuth, 0.0)
    roll = level.T @ R
    twist = np.degrees(np.arctan2(roll[1, 0], roll[0, 0]))
    return float(elevation), float(azimuth), float(twist)


def rotation_from_fictrac(c2a_r) -> tuple[float, float, float]:
    """`camera.rotation` for a FicTrac `c2a_r` (its animal frame: y right, z down)."""
    from spintrack.geometry import matrix_to_rotvec

    R = FICTRAC_TO_ANIMAL @ rotvec_to_matrix(np.asarray(c2a_r, dtype=np.float64))
    return tuple(float(v) for v in matrix_to_rotvec(R))
