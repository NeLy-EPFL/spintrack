"""Calibration state independent of any GUI toolkit: points, fits, overlays, config.

The GUI collects clicks; this module turns them into the ball's rim (`ball.rim`), the
ignore polygons (`mask.ignore`) and the camera position (`camera.position_deg`, or
`camera.rotation` from a calibration square), draws the overlay that lets the user
verify them, and writes the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spintrack.calibrate.headless import save_config
from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.calibrate.square import square_pose
from spintrack.camera import Camera, source_camera
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec, normalize
from spintrack.sphere import ball_outline, fit_ball

AXIS_COLORS = ((255, 80, 80), (80, 220, 80), (80, 140, 255))  # x, y, z in RGB


@dataclass
class CalibrationSession:
    config: Config
    frame: np.ndarray  # gray or BGR uint8 source frame
    config_path: Path | None = None
    circle_points: list[tuple[float, float]] = field(default_factory=list)
    ignore_polygons: list[list[tuple[float, float]]] = field(default_factory=list)
    current_polygon: list[tuple[float, float]] = field(default_factory=list)
    square_points: list[tuple[float, float]] = field(default_factory=list)
    square_plane: str = "xy"
    angles: tuple[float, float, float] | None = None  # elevation, azimuth, twist (deg)
    center: np.ndarray | None = None
    half_angle: float | None = None
    cam_to_lab: np.ndarray | None = None

    def __post_init__(self) -> None:
        h, w = self.frame.shape[:2]
        camera = self.config.camera
        if camera.vfov_deg is None:
            raise ValueError("config needs camera.vfov_deg before calibration")
        self.camera: Camera = source_camera(w, h, camera.vfov_deg, camera.fisheye)
        self.circle_points = list(self.config.ball.rim)
        self.fit_circle()
        self.ignore_polygons = [list(poly) for poly in self.config.mask.ignore]
        if camera.position_deg is not None:
            self.set_angles(*camera.position_deg)
        else:
            self.cam_to_lab = camera.to_animal()

    # ----- ball -----
    def fit_circle(self) -> bool:
        if len(self.circle_points) < 3:
            return False
        self.center, self.half_angle = fit_ball(self.circle_points, self.camera)
        return True

    # ----- ignore regions -----
    def close_polygon(self) -> bool:
        if len(self.current_polygon) < 3:
            return False
        self.ignore_polygons.append(list(self.current_polygon))
        self.current_polygon = []
        return True

    # ----- camera-to-lab -----
    def solve_square(self) -> bool:
        if len(self.square_points) != 4:
            return False
        R, _ = square_pose(self.square_points, self.camera, self.square_plane)
        self.cam_to_lab = R.T
        self.angles = None
        return True

    def set_angles(self, elevation: float, azimuth: float, twist: float = 0.0) -> None:
        self.angles = (elevation, azimuth, twist)
        self.cam_to_lab = camera_to_lab_from_angles(elevation, azimuth, twist)

    # ----- readouts -----
    def cursor_angle(self, x: float, y: float) -> float | None:
        """Angle (deg, counter-clockwise from image +x) of a point about the ball
        center.
        """
        if self.center is None:
            return None
        cx, cy, _ = self.camera.project(self.center)
        return float(np.degrees(np.arctan2(-(y - cy), x - cx)))

    # ----- output -----
    def to_config(self) -> Config:
        cfg = self.config
        if len(self.circle_points) >= 3:
            cfg.ball.rim = self.circle_points
        cfg.mask.ignore = [
            [(round(x), round(y)) for x, y in poly] for poly in self.ignore_polygons
        ]
        if self.cam_to_lab is not None:
            camera = cfg.camera
            camera.position_deg = camera.rotation = None
            if self.angles is not None:
                camera.position_deg = self.angles
            else:
                camera.rotation = tuple(matrix_to_rotvec(self.cam_to_lab))
        return cfg

    def save(self, path: Path | None = None) -> Path:
        return save_config(self.to_config(), path or self.config_path or "config.toml")

    # ----- overlay -----
    def overlay(self) -> np.ndarray:
        """RGB image of the frame with the current calibration drawn on it."""
        img = self.frame
        rgb = cv2.cvtColor(
            img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB
        )
        for pt in self.circle_points:
            cv2.circle(rgb, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1, cv2.LINE_AA)
        if self.center is not None and self.half_angle is not None:
            pts = ball_outline(self.camera, self.center, self.half_angle, 90)
            cv2.polylines(
                rgb, [np.round(pts).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA
            )
            cx, cy, _ = self.camera.project(self.center)
            cv2.drawMarker(
                rgb, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1
            )
            if self.cam_to_lab is not None:
                self._draw_axes(rgb)
        for poly in self.ignore_polygons:
            cv2.polylines(
                rgb, [np.round(poly).astype(np.int32)], True, (255, 60, 60), 1
            )
        if len(self.current_polygon) > 1:
            cv2.polylines(
                rgb,
                [np.round(self.current_polygon).astype(np.int32)],
                False,
                (255, 160, 60),
                1,
            )
        for pt in self.current_polygon:
            cv2.circle(rgb, (int(pt[0]), int(pt[1])), 3, (255, 160, 60), -1)
        for i, pt in enumerate(self.square_points):
            cv2.circle(rgb, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)
            cv2.putText(rgb, ["TL", "TR", "BR", "BL"][i],
                        (int(pt[0]) + 5, int(pt[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)  # fmt: skip
        return rgb

    def _draw_axes(self, rgb: np.ndarray) -> None:
        """Lab axes attached to the ball center: x forward (red), y right (green), z
        down.
        """
        assert (
            self.center is not None
            and self.half_angle is not None
            and self.cam_to_lab is not None
        )
        cx, cy, _ = self.camera.project(self.center)
        length = 0.8 * np.sin(self.half_angle)
        for i, color in enumerate(AXIS_COLORS):
            axis_cam = self.cam_to_lab.T[
                :, i
            ]  # lab axis i expressed in camera coordinates
            tip = normalize(self.center + length * axis_cam)
            tx, ty, _ = self.camera.project(tip)
            cv2.arrowedLine(
                rgb,
                (int(cx), int(cy)),
                (int(tx), int(ty)),
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.2,
            )
            cv2.putText(
                rgb,
                "xyz"[i],
                (int(tx) + 4, int(ty) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
