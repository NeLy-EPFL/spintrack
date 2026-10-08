"""Camera-to-animal rotation from a calibration square drawn in an animal plane.

The user clicks the four corners of a square lying in one of the animal's coordinate
planes, in the order FicTrac's configGui asks for, which names them in the animal's
terms (see `SQUARE_CORNERS`) and so holds from any side and image orientation. The
order is what tells the sides apart: corners clicked in mirrored order fit just as
well, by the square turned over, and the rotation comes out a half turn off. Solving
the square's pose gives the rotation between camera and animal frames; the square's
true size is irrelevant for the rotation.

`SQUARE_CORNERS` and `square_pose` are in FicTrac's animal frame (x forward, y right,
z down), as FicTrac's own corners are; `camera_to_lab_from_square` converts to
spintrack's (x forward, y left, z up).
"""

from __future__ import annotations

import cv2
import numpy as np

from spintrack.calibrate.sliders import FICTRAC_TO_ANIMAL
from spintrack.camera import Camera
from spintrack.geometry import normalize, rotvec_to_matrix

# FicTrac's unit-square corners (its XY_CNRS, YZ_CNRS, XZ_CNRS) in click order: xy
# front-left, front-right, back-right, back-left; yz left-up, right-up, right-down,
# left-down; xz front-up, back-up, back-down, front-down.
SQUARE_CORNERS: dict[str, np.ndarray] = {
    "xy": np.array(
        [[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]]
    ),
    "yz": np.array(
        [[0.0, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5], [0.0, -0.5, 0.5]]
    ),
    "xz": np.array(
        [[0.5, 0.0, -0.5], [-0.5, 0.0, -0.5], [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]
    ),
}


def square_pose(
    corners_xy, camera: Camera, plane: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pose `(R, t)` of the square with `camera_point = R @ animal_point + t`, the
    animal point in FicTrac's frame.

    `plane` is `"xy"`, `"yz"` or `"xz"`.
    Initialized with a PnP solve (SQPnP) on normalized coordinates (so any camera model
    works), then refined by Gauss-Newton on the direction residuals.
    """
    if plane not in SQUARE_CORNERS:
        raise ValueError(f"unknown square plane {plane!r}")
    obj = SQUARE_CORNERS[plane]
    pts = np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("a square needs exactly four corners")
    rays = camera.rays(pts[:, 0], pts[:, 1])
    if np.any(rays[:, 2] <= 0):
        raise ValueError("square corners must be in front of the camera")
    normalized = (rays[:, :2] / rays[:, 2:3]).reshape(-1, 1, 2)
    # Not IPPE: its rotation vector is garbage at an exact half turn, the pose of an
    # untwisted camera at the animal's left or of one straight below it.
    ok, rvec, tvec = cv2.solvePnP(
        obj, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        raise RuntimeError("planar pose initialization failed")
    params = np.concatenate([rvec.ravel(), tvec.ravel()])

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = normalize(obj @ rotvec_to_matrix(p[:3]).T + p[3:])
        return (pred - rays).ravel()

    for _ in range(50):
        r0 = residuals(params)
        jac = np.empty((r0.size, 6))
        for j in range(6):
            step = np.zeros(6)
            step[j] = 1e-6
            jac[:, j] = (residuals(params + step) - residuals(params - step)) / 2e-6
        delta, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        params = params + delta
        if np.linalg.norm(delta) < 1e-12:
            break
    return rotvec_to_matrix(params[:3]), params[3:]


def camera_to_lab_from_square(corners_xy, camera: Camera, plane: str) -> np.ndarray:
    """Rotation matrix `R` with `v_lab = R @ v_camera` in spintrack's animal frame, the
    matrix of `camera.rotation`, from a square's corners as `square_pose` takes them."""
    R, _ = square_pose(corners_xy, camera, plane)
    return FICTRAC_TO_ANIMAL @ R.T
