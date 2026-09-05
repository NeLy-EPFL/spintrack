"""Frame-in, data-out tracker: source frames to FicTrac-compatible records.

`Tracker.process_frame(gray, ts_ms)` remaps the frame into the tracking window, runs the
engine, converts the rotation into camera and lab coordinates, integrates the path and
returns a `FrameResult` (or `None` when the frame could not be tracked).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from spintrack.calibrate.square import camera_to_lab_from_square
from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.engine import StepResult, TrackEngine, TrackParams
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.io.dat import N_COLUMNS
from spintrack.io.sources import ms_since_midnight
from spintrack.path import PathIntegrator
from spintrack.sphere import fit_ball, source_mask, window_geometry

log = logging.getLogger("spintrack")


@dataclass
class FrameResult:
    frame: int
    seq: int
    ts_ms: float
    w_cam: np.ndarray  # rotation increment, camera frame
    w_lab: np.ndarray  # rotation increment, lab frame
    R_cam: np.ndarray  # absolute ball orientation, camera frame
    R_lab: np.ndarray
    step: StepResult
    values: np.ndarray  # the 25 FicTrac columns

    @property
    def heading(self) -> float:
        return float(self.values[16])


def params_from_config(cfg: Config, base: TrackParams | None = None) -> TrackParams:
    """Derive solver parameters from FicTrac-style config keys."""
    p = replace(base) if base is not None else TrackParams()
    p.norm_win_pc = float(cfg.thr_win_pc) if cfg.thr_win_pc > 0 else p.norm_win_pc
    p.forget_outside_view = not cfg.accumulate_map
    p.max_bad_frames = int(cfg.max_bad_frames)
    p.global_search = bool(cfg.opt_do_global)
    if cfg.opt_bound > 0:
        p.max_step = float(cfg.opt_bound)
    return p


class Tracker:
    def __init__(
        self, cfg: Config, width: int, height: int, params: TrackParams | None = None
    ):
        if cfg.vfov is None or cfg.vfov <= 0:
            raise ValueError("config needs a positive vfov (degrees)")
        self.cfg = cfg
        self.width, self.height = int(width), int(height)
        self.camera = source_camera(width, height, cfg.vfov, cfg.fisheye)
        if cfg.roi_c is not None and cfg.roi_r is not None and len(cfg.roi_c) == 3:
            self.centre = normalize(np.asarray(cfg.roi_c, dtype=np.float64))
            self.half_angle = float(cfg.roi_r)
        elif len(cfg.roi_circ) >= 6:
            pts = np.asarray(cfg.roi_circ, dtype=np.float64).reshape(-1, 2)
            self.centre, self.half_angle = fit_ball(pts, self.camera)
        else:
            raise ValueError("config must define the ball via roi_c/roi_r or roi_circ")
        mask = source_mask(self.camera, self.centre, self.half_angle, cfg.roi_ignr)
        self.geometry = window_geometry(
            self.camera, self.centre, self.half_angle, cfg.window_size(), mask
        )
        self.R_wc = self.geometry.to_camera  # window -> camera
        self.cam_to_lab = self._camera_to_lab(cfg)
        self.params = params_from_config(cfg, params)
        self.engine = TrackEngine(self.geometry, self.params)
        self.path = PathIntegrator()
        self.frame = 0
        self.seq = 0
        self._prev_ts: float | None = None

    def _camera_to_lab(self, cfg: Config) -> np.ndarray:
        if cfg.c2a_r is not None and len(cfg.c2a_r) == 3:
            return rotvec_to_matrix(np.asarray(cfg.c2a_r, dtype=np.float64))
        src = cfg.c2a_src
        corners = getattr(cfg, src, None) if src.startswith("c2a_cnrs_") else None
        if corners and len(corners) == 8:
            pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            return camera_to_lab_from_square(pts, self.camera, src[-2:])
        log.warning("no camera-to-lab transform in config (c2a_r); using identity")
        return np.eye(3)

    def reset(self) -> None:
        self.engine.reset()
        self.path.reset()
        self.seq = 0

    def process_frame(
        self, gray: np.ndarray, ts_ms: float = -1.0, wall_ms: float | None = None
    ) -> FrameResult | None:
        """Track one grayscale frame (2-D uint8). Returns None if the frame was dropped."""
        window = self.geometry.remap(gray)
        step = self.engine.step(window)
        frame = self.frame
        self.frame += 1
        if not step.ok:
            self.seq += 1
            self._prev_ts = ts_ms
            return None
        if step.source == "reset":
            self.path.reset()
            self.seq = 0

        R_wc = self.R_wc
        w_cam = R_wc @ step.w_win
        R_cam = R_wc @ step.R_win @ R_wc.T
        w_lab = self.cam_to_lab @ w_cam
        R_lab = self.cam_to_lab @ R_cam @ self.cam_to_lab.T
        p = self.path.step(w_lab)
        delta_ts = 0.0 if self._prev_ts is None else ts_ms - self._prev_ts
        self._prev_ts = ts_ms
        values = np.empty(N_COLUMNS)
        values[0] = frame
        values[1:4] = w_cam
        values[4] = step.cost if np.isfinite(step.cost) else 0.0
        values[5:8] = w_lab
        values[8:11] = matrix_to_rotvec(R_cam)
        values[11:14] = matrix_to_rotvec(R_lab)
        values[14:17] = (p.pos_x, p.pos_y, p.heading)
        values[17:19] = (p.step_dir, p.step_mag)
        values[19:21] = (p.int_x, p.int_y)
        values[21] = ts_ms
        values[22] = self.seq
        values[23] = delta_ts
        values[24] = ms_since_midnight() if wall_ms is None else wall_ms
        self.seq += 1
        return FrameResult(
            frame, int(values[22]), ts_ms, w_cam, w_lab, R_cam, R_lab, step, values
        )
