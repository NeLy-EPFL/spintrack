"""Calibration state independent of any GUI toolkit: points, fits, overlays, config.

The GUI collects clicks; this module turns them into the ball ROI (`roi_c`, `roi_r`,
`roi_circ`), ignore polygons (`roi_ignr`) and the camera-to-lab transform (`c2a_r`), draws
the overlay that lets the user verify them, and writes the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.calibrate.square import square_pose
from spintrack.camera import Camera, source_camera
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.sphere import ball_outline, fit_ball

AXIS_COLOURS = ((255, 80, 80), (80, 220, 80), (80, 140, 255))  # x, y, z in RGB


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
    centre: np.ndarray | None = None
    half_angle: float | None = None
    cam_to_lab: np.ndarray | None = None

    def __post_init__(self) -> None:
        h, w = self.frame.shape[:2]
        if self.config.vfov is None or self.config.vfov <= 0:
            raise ValueError("config needs vfov (degrees) before calibration")
        self.camera: Camera = source_camera(w, h, self.config.vfov, self.config.fisheye)
        cfg = self.config
        if cfg.roi_circ:
            self.circle_points = [
                (float(cfg.roi_circ[i]), float(cfg.roi_circ[i + 1]))
                for i in range(0, len(cfg.roi_circ) - 1, 2)
            ]
        if cfg.roi_c is not None and cfg.roi_r is not None:
            self.centre = normalize(np.asarray(cfg.roi_c, dtype=np.float64))
            self.half_angle = float(cfg.roi_r)
        elif len(self.circle_points) >= 3:
            self.fit_circle()
        for poly in cfg.roi_ignr:
            self.ignore_polygons.append(
                [
                    (float(poly[i]), float(poly[i + 1]))
                    for i in range(0, len(poly) - 1, 2)
                ]
            )
        if cfg.c2a_r is not None and len(cfg.c2a_r) == 3:
            self.cam_to_lab = rotvec_to_matrix(np.asarray(cfg.c2a_r, dtype=np.float64))

    # ----- ball -----
    def fit_circle(self) -> bool:
        if len(self.circle_points) < 3:
            return False
        self.centre, self.half_angle = fit_ball(self.circle_points, self.camera)
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
        """Angle (deg, counter-clockwise from image +x) of a point about the ball centre."""
        if self.centre is None:
            return None
        cx, cy, _ = self.camera.project(self.centre)
        return float(np.degrees(np.arctan2(-(y - cy), x - cx)))

    # ----- output -----
    def to_config(self) -> Config:
        cfg = self.config
        if self.circle_points:
            cfg.roi_circ = [round(v) for pt in self.circle_points for v in pt]
        if self.centre is not None and self.half_angle is not None:
            cfg.roi_c = [float(v) for v in self.centre]
            cfg.roi_r = float(self.half_angle)
        cfg.roi_ignr = [
            [round(v) for pt in poly for v in pt] for poly in self.ignore_polygons
        ]
        if self.cam_to_lab is not None:
            cfg.c2a_r = [float(v) for v in matrix_to_rotvec(self.cam_to_lab)]
            if self.angles is not None:
                cfg.c2a_src = "sliders"
                cfg.extra["c2a_angles"] = [float(v) for v in self.angles]
            elif len(self.square_points) == 4:
                key = f"c2a_cnrs_{self.square_plane}"
                setattr(cfg, key, [round(v) for pt in self.square_points for v in pt])
                cfg.c2a_src = key
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.config_path or "config.txt")
        self.to_config().save(path)
        return path

    # ----- overlay -----
    def overlay(self) -> np.ndarray:
        """RGB image of the frame with the current calibration drawn on it."""
        img = self.frame
        rgb = cv2.cvtColor(
            img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB
        )
        for pt in self.circle_points:
            cv2.circle(rgb, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1, cv2.LINE_AA)
        if self.centre is not None and self.half_angle is not None:
            pts = ball_outline(self.camera, self.centre, self.half_angle, 90)
            cv2.polylines(
                rgb, [np.round(pts).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA
            )
            cx, cy, _ = self.camera.project(self.centre)
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
            cv2.putText(rgb, ["TL", "TR", "BR", "BL"][i], (int(pt[0]) + 5, int(pt[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)  # fmt: skip
        return rgb

    def _draw_axes(self, rgb: np.ndarray) -> None:
        """Lab axes attached to the ball centre: x forward (red), y right (green), z down."""
        assert (
            self.centre is not None
            and self.half_angle is not None
            and self.cam_to_lab is not None
        )
        cx, cy, _ = self.camera.project(self.centre)
        length = 0.8 * np.sin(self.half_angle)
        for i, colour in enumerate(AXIS_COLOURS):
            axis_cam = self.cam_to_lab.T[
                :, i
            ]  # lab axis i expressed in camera coordinates
            tip = normalize(self.centre + length * axis_cam)
            tx, ty, _ = self.camera.project(tip)
            cv2.arrowedLine(
                rgb,
                (int(cx), int(cy)),
                (int(tx), int(ty)),
                colour,
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
                colour,
                1,
            )
