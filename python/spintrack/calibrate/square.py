"""Camera-to-animal rotation from a calibration square drawn in an animal plane.

The user marks the four corners (top-left, top-right, bottom-right, bottom-left as seen
in the image) of a square lying in one of the animal's coordinate planes. Solving the
square's pose gives the rotation between camera and animal frames; the square's true
size is irrelevant for the rotation.

Animal frame (FicTrac convention): x forward, y right, z down.
"""

from __future__ import annotations

import cv2
import numpy as np

from spintrack.camera import Camera
from spintrack.geometry import normalize, rotvec_to_matrix

# Unit-square corners in animal coordinates, order TL, TR, BR, BL, per plane. The first
# two corners span the plane's first axis, the last coordinate pairs the other axis.
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
    """Pose `(R, t)` of the square with `camera_point = R @ animal_point + t`.

    `plane` is `"xy"`, `"yz"` or `"xz"`.
    Initialized with a planar PnP solve on normalized coordinates (so any camera model
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
    ok, rvec, tvec = cv2.solvePnP(
        obj, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise RuntimeError("planar pose initialisation failed")
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
