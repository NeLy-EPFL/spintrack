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
from spintrack.maps import load_illumination, load_map, save_map
from spintrack.path import PathIntegrator
from spintrack.photometry import TAU
from spintrack.sphere import (
    center_from_pixel_circle,
    fit_ball,
    pixel_circle,
    source_mask,
    window_geometry,
)

log = logging.getLogger("spintrack")

# Below this ball radius the reported rotation shrinks, and nothing in the run says so:
# the window is resampled to `q_factor` either way and the cost stays low. See
# `docs/guide.md`, "Troubleshooting".
MIN_BALL_RADIUS_PX = 15.0

# Move the window when the followed center has drifted this far from it. Small
# enough that the ball's own movement is not read as rotation, large enough that a
# still ball is not re-fitted on detection noise.
FOLLOW_TOL_PX = 0.25


@dataclass
class FrameResult:
    """One tracked frame. Distances are in ball radii (radians of ball rotation).

    The animal-frame properties follow `spintrack.path`: lab x forward, y right, z down.
    """

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
    def forward(self) -> float:
        """Forward motion over this frame."""
        return float(self.w_lab[1])

    @property
    def side(self) -> float:
        """Sideways motion over this frame, positive to the animal's right."""
        return -float(self.w_lab[0])

    @property
    def turn(self) -> float:
        """Change of heading over this frame in rad, positive turning right."""
        return -float(self.w_lab[2])

    @property
    def heading(self) -> float:
        """Integrated heading in rad, [0, 2 pi), from the first tracked frame's."""
        return float(self.values[16])

    @property
    def x(self) -> float:
        """Integrated position along the starting heading."""
        return float(self.values[14])

    @property
    def y(self) -> float:
        """Integrated position to the right of the starting heading."""
        return float(self.values[15])


def params_from_config(cfg: Config, base: TrackParams | None = None) -> TrackParams:
    """Derive solver parameters from FicTrac-style config keys."""
    p = replace(base) if base is not None else TrackParams()
    p.norm_win_pc = float(cfg.thr_win_pc) if cfg.thr_win_pc > 0 else p.norm_win_pc
    p.forget_outside_view = not cfg.accumulate_map
    p.max_bad_frames = int(cfg.max_bad_frames)
    if not cfg.illumination:
        p.illum_bias = False
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
            self.center = normalize(np.asarray(cfg.roi_c, dtype=np.float64))
            self.half_angle = float(cfg.roi_r)
        elif len(cfg.roi_circ) >= 6:
            pts = np.asarray(cfg.roi_circ, dtype=np.float64).reshape(-1, 2)
            self.center, self.half_angle = fit_ball(pts, self.camera)
        else:
            raise ValueError("config must define the ball via roi_c/roi_r or roi_circ")
        self.params = params_from_config(cfg, params)
        circle = pixel_circle(self.camera, self.center, self.half_angle)
        self.ball_radius_px = circle[2]
        self._circle_px = circle[:2]  # the window's ball circle, for the watch
        if self.ball_radius_px < MIN_BALL_RADIUS_PX:
            log.warning(
                "the ball is only %.1f px in radius; rotations will be reported low, "
                "the one about the optical axis first (sideslip, for a camera behind "
                "the animal). See docs/guide.md, Troubleshooting",
                self.ball_radius_px,
            )
        mask = source_mask(self.camera, self.center, self.half_angle, cfg.roi_ignr)
        self.geometry = window_geometry(
            self.camera,
            self.center,
            self.half_angle,
            cfg.window_size(),
            mask,
            prefilter=self.params.prefilter,
        )
        self.R_wc = self.geometry.to_camera  # window -> camera
        # The reporting convention is fixed to the first window frame, so a later re-fit
        # moves the window without stepping the absolute-orientation columns.
        self.R_wc0 = self.R_wc
        self.cam_to_lab = self._camera_to_lab(cfg)
        if cfg.sphere_map_fn:
            self.params.global_search = True  # needed to localize against the template
        self.engine = TrackEngine(self.geometry, self.params)
        if cfg.sphere_map_fn:
            mean, weight = load_map(cfg.sphere_map_fn, self.engine.map_shape)
            self.engine.load_map(mean, weight, frozen=cfg.map_frozen)
            # The illumination field describes the rig, so a saved one is a head start.
            self._load_illumination(cfg.sphere_map_fn)
        if cfg.illumination_fn:
            self._load_illumination(cfg.illumination_fn, asked=True)
        self.path = PathIntegrator()
        self.frame = 0
        self.seq = 0
        self._prev_ts: float | None = None  # timestamp of the last tracked frame
        self.geometry_version = 0  # bumped by every window move
        self.center_initial = self.center.copy()
        self.watch = None
        if self.params.center_watch:
            from spintrack.refit import CenterWatch

            self.watch = CenterWatch(circle[:2], circle[2])

    def _camera_to_lab(self, cfg: Config) -> np.ndarray:
        """The transform named by `cfg.c2a_source()`; identity (with a warning) if none.

        Also records which key it came from in `self.c2a_source`. The camera-frame
        columns of the output are valid without a transform, so this warns rather than
        raising; `spintrack run` refuses instead, because its lab-frame columns would be
        camera values in disguise.
        """
        self.c2a_source = cfg.c2a_source() or "identity"
        if self.c2a_source == "c2a_r":
            return rotvec_to_matrix(np.asarray(cfg.c2a_r, dtype=np.float64))
        if self.c2a_source.startswith("c2a_cnrs_"):
            corners = getattr(cfg, self.c2a_source)
            pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            return camera_to_lab_from_square(pts, self.camera, self.c2a_source[-2:])
        log.warning("no camera-to-lab transform in config (c2a_r); using identity")
        return np.eye(3)

    def reset(self) -> None:
        self.engine.reset()
        self.path.reset()
        self.seq = 0

    def prime_from(self, other: Tracker) -> None:
        """Start from another tracker's map and illumination field.

        Both trackers must have been built from the same config, so their initial
        windows agree and the map needs no correction: the map lives in the window frame
        at `R = I`, and a window re-fit in `other` carried it unchanged (see
        `TrackEngine.rebuild`). The illumination field is fixed in the *window*, so it
        comes from wherever `other` ended up and is resampled into this one.
        When `other` watched the ball, this tracker's window follows the trajectory
        `other` measured, planned from all of its looks at once (`CenterWatch.replay`),
        instead of following the ball for itself.
        """
        mean, weight = other.engine.export_map()
        self.engine.load_map(mean, weight, frozen=self.cfg.map_frozen, localize=False)
        # No accumulator weight on the field, unlike `illumination_fn`: this pass
        # sees the same lighting and re-measures it within the warmup anyway.
        self.engine.load_illumination(
            other.engine.photometry.state(), other.center, other.half_angle
        )
        if self.watch is not None and other.watch is not None:
            planned = other.watch.replay(other.frame)
            if planned is not None:
                self.watch = planned

    def _load_illumination(self, path, asked: bool = False) -> bool:
        """Take the illumination field out of a map `.npz`, if it carries a usable one.

        The field is window-shaped, so a file saved for a window of another size has
        nothing to give this run; that is worth a warning only when the run asked for
        the file by name.
        """
        fields, saved = load_illumination(path, self.geometry.size)
        if not fields:
            if asked:
                log.warning(
                    "%s carries no illumination field for a %d px window: "
                    "this run estimates its own",
                    path,
                    self.geometry.size,
                )
            return False
        self.engine.load_illumination(
            fields,
            saved.get("center", self.center),
            saved.get("half_angle", self.half_angle),
            # The field describes the rig: it counts as much as this run's own memory
            # until the run's frames outweigh it.
            prior_frames=TAU,
        )
        return True

    def save_map(self, path) -> None:
        """Write the current surface map as a spintrack `.npz` template."""
        mean, weight = self.engine.export_map()
        illum = self.engine.photometry.state()
        save_map(
            path,
            mean,
            weight,
            window_size=self.geometry.size,
            center=self.center,
            half_angle=self.half_angle,
            illum_bias=illum["bias"],
            illum_gain=illum["gain"],
        )

    def process_frame(
        self, gray: np.ndarray, ts_ms: float = -1.0, wall_ms: float | None = None
    ) -> FrameResult | None:
        """Track one grayscale frame (2-D uint8); None if the frame was dropped."""
        # Move the window onto the ball before tracking this frame, so that the window
        # does not trail a moving ball by a frame.
        self._watch_center(gray)
        window = self.geometry.remap(gray)
        step = self.engine.step(window)
        frame = self.frame
        self.frame += 1
        if not step.ok:
            # `delta_ts` spans the lost frames, like the next row's increment does.
            self.seq += 1
            return None
        if step.source == "reset" or (step.source == "global" and self.seq == 0):
            self.path.reset()
            self.seq = 0

        delta_ts = 0.0 if self._prev_ts is None else ts_ms - self._prev_ts
        self._prev_ts = ts_ms
        wall = ms_since_midnight() if wall_ms is None else wall_ms
        values, w_cam, w_lab, R_cam, R_lab = self._values(
            frame, ts_ms, wall, step, delta_ts
        )
        self.seq += 1
        return FrameResult(
            frame, int(values[22]), ts_ms, w_cam, w_lab, R_cam, R_lab, step, values
        )

    def refit_center(self, new_center, circle_px=None) -> np.ndarray:
        """Point the tracking window at a new ball center, keeping the surface map.

        Returns the rotation `Q` from the old window frame to the new one, which the
        caller must apply to any orientation it recorded in the old frame. The ignore
        polygons are image-fixed (the animal has not moved), so they are not translated.
        `circle_px` is the new center's `pixel_circle` center, when already known.
        """
        center = normalize(np.asarray(new_center, dtype=np.float64))
        if circle_px is None:
            circle_px = pixel_circle(self.camera, center, self.half_angle)[:2]
        self._circle_px = circle_px
        mask = source_mask(self.camera, center, self.half_angle, self.cfg.roi_ignr)
        geometry = window_geometry(
            self.camera,
            center,
            self.half_angle,
            self.cfg.window_size(),
            mask,
            prefilter=self.params.prefilter,
        )
        Q = geometry.to_camera.T @ self.R_wc
        self.engine.rebuild(geometry, Q)
        self.geometry = geometry
        self.R_wc = geometry.to_camera
        self.center = center
        self.geometry_version += 1
        return Q

    def _watch_center(self, gray: np.ndarray) -> None:
        """Keep the tracking window on a ball that is moving in its holder."""
        if self.watch is None:
            return
        target = self.watch.update(self.frame, gray)
        if target is None:
            return
        cx, cy = self._circle_px
        if float(np.hypot(target[0] - cx, target[1] - cy)) < FOLLOW_TOL_PX:
            return
        # The radius is left alone: a ball only changes apparent size by moving along
        # the optical axis, and translation on its own is much better conditioned. The
        # target is where the silhouette's circle should sit, which is not where the
        # ball's center projects, so it is inverted rather than read as a direction.
        center, circle = center_from_pixel_circle(
            self.camera, target, self.half_angle, self.center, self._circle_px
        )
        self.refit_center(center, circle)

    def _values(self, frame, ts_ms, wall_ms, step, delta_ts):
        """The 25 FicTrac columns for one tracked frame (also advances the path)."""
        w_cam = self.R_wc @ step.w_win
        R_cam = self.R_wc @ step.R_win @ self.R_wc0.T
        w_lab = self.cam_to_lab @ w_cam
        R_lab = self.cam_to_lab @ R_cam @ self.cam_to_lab.T
        p = self.path.step(w_lab)
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
        values[24] = wall_ms
        return values, w_cam, w_lab, R_cam, R_lab
