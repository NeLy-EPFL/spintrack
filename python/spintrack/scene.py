"""What a view of a run draws, kept up frame by frame.

The debug video and the web page both show the ball's outline, the trail the animal has
walked over the ball, the ball's axes, the animal's at where it stands, and the fictive
path. `Scene` keeps them in source pixels, so that each view only scales and draws.
"""

from __future__ import annotations

import numpy as np

from spintrack.sphere import ball_outline
from spintrack.tracker import FrameResult, Tracker

# The animal's trail over the ball: how many frames of it to keep, as FicTrac's
# DRAW_SPHERE_HIST_LENGTH.
TRAIL_FRAMES = 1024
# Draw the trail only where the surface faces the camera by more than this cosine: near
# the limb it is so foreshortened that any trail there hugs the outline.
TRAIL_LIMB_COS = 0.1
# The drawn axes, in ball radii: the ball's from its center, the animal's from where it
# stands, which is near the frame's edge on most rigs.
AXIS_LENGTH = 0.35
ANIMAL_AXIS_LENGTH = 0.2


class Scene:
    """The path, the trail, the outline and the axes of a tracker's run so far."""

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.sin_half = np.sin(tracker.half_angle)
        # The lab frame's up (+z) is the ball's normal where the animal stands, so the
        # surface point it touches is that axis in camera coordinates.
        self.up_cam = tracker.cam_to_lab[2].copy()
        self.geometry_version = -1
        self._update_outline()
        self._path = np.empty((1024, 2), np.float64)  # grown by doubling
        self._n_path = 0
        self.path_bbox = [-0.1, 0.1, -0.1, 0.1]
        # Where the animal has touched the ball, in its body frame, oldest first.
        self._trail = np.empty((TRAIL_FRAMES, 3), np.float64)
        self.n_trail = 0
        self.R_cam = None  # last tracked orientation, so a dropped frame still draws

    def update(self, result: FrameResult | None) -> None:
        """Take in a frame's result (None: dropped)."""
        if self.tracker.geometry_version != self.geometry_version:
            self._update_outline()
        if result is not None:
            self.R_cam = result.R_cam
            self._append_trail(result.R_cam.T @ self.up_cam)
            self._append_path(float(result.values[14]), float(result.values[15]))

    @property
    def path_pts(self) -> np.ndarray:
        """The integrated path so far, as an (n, 2) array of (x, y)."""
        return self._path[: self._n_path]

    def _append_path(self, x: float, y: float) -> None:
        if self._n_path == len(self._path):
            self._path = np.resize(self._path, (2 * len(self._path), 2))
        self._path[self._n_path] = (x, y)
        self._n_path += 1
        b = self.path_bbox
        b[:] = min(b[0], x), max(b[1], x), min(b[2], y), max(b[3], y)

    def _append_trail(self, contact_body: np.ndarray) -> None:
        """Keep the last `TRAIL_FRAMES` contact points, oldest first."""
        if self.n_trail == TRAIL_FRAMES:
            self._trail[:-1] = self._trail[1:]
            self.n_trail -= 1
        self._trail[self.n_trail] = contact_body
        self.n_trail += 1

    def _update_outline(self) -> None:
        """Re-project the ball outline; the window may have moved onto a moved ball."""
        tr = self.tracker
        self.geometry_version = tr.geometry_version
        self.outline = ball_outline(tr.camera, tr.center, tr.half_angle, 90)
        cx, cy, _ = tr.camera.project(tr.center)
        self.center_px = (float(cx), float(cy))

    def trail_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the animal's past contact points sit now: (n, 2) pixels, oldest
        first, and which of them the camera can see.

        The animal stays put while the ball turns under it, so the surface point it
        touched at frame `i` is now `R_cam @ R_cam(i).T @ up`: the animal's path,
        inverted, painted on the ball.
        """
        if self.R_cam is None:
            return np.zeros((0, 2)), np.zeros(0, bool)
        tr = self.tracker
        # Each stored point is body-fixed, so the ball's current orientation says where
        # the surface carried it.
        contact = self._trail[: self.n_trail] @ self.R_cam.T
        seen = contact @ tr.center < -self.sin_half - TRAIL_LIMB_COS
        x, y, inside = tr.camera.project(tr.center + self.sin_half * contact)
        seen &= inside
        pts = np.stack([np.where(seen, x, 0.0), np.where(seen, y, 0.0)], axis=1)
        return pts, seen

    def axes(self) -> dict[str, list | tuple[float, float] | None]:
        """The tips of the ball's axes from its center (None before a tracked frame),
        where the animal stands, and the tips of its axes from there."""
        tracker, R, up = self.tracker, self.R_cam, self.up_cam
        lab = tracker.cam_to_lab
        return {
            "ball": None
            if R is None
            else [self._image(AXIS_LENGTH * R[:, i]) for i in range(3)],
            "contact": self._image(up),
            "animal": [self._image(up + ANIMAL_AXIS_LENGTH * lab[i]) for i in range(3)],
        }

    def _image(self, offset: np.ndarray) -> tuple[float, float]:
        """Where the point `offset` from the ball's center, in ball radii, images."""
        tr = self.tracker
        x, y, _ = tr.camera.project(tr.center + self.sin_half * offset)
        return float(x), float(y)
