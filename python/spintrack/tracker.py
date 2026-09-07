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
from spintrack.sphere import (
    centre_from_pixel_circle,
    fit_ball,
    pixel_circle,
    source_mask,
    window_geometry,
)

log = logging.getLogger("spintrack")

# Move the window when the followed centre has drifted this far from it. Small
# enough that the ball's own movement is not read as rotation, large enough that a
# still ball is not re-fitted on detection noise.
FOLLOW_TOL_PX = 0.25


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
    if not cfg.illumination:
        p.illum_bias = p.illum_gain = p.illum_weight = p.illum_flat = False
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
        self.params = params_from_config(cfg, params)
        mask = source_mask(self.camera, self.centre, self.half_angle, cfg.roi_ignr)
        self.geometry = window_geometry(
            self.camera,
            self.centre,
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
            self.params.global_search = True  # needed to localise against the template
        self.engine = TrackEngine(self.geometry, self.params)
        if cfg.sphere_map_fn:
            mean, weight = load_map(cfg.sphere_map_fn, self.engine.map_shape)
            self.engine.load_map(mean, weight, frozen=cfg.map_frozen)
            # The illumination fields describe the rig, so a saved one is a head start.
            fields, saved = load_illumination(cfg.sphere_map_fn, self.geometry.size)
            if fields:
                self.engine.load_illumination(
                    fields,
                    saved.get("centre", self.centre),
                    saved.get("half_angle", self.half_angle),
                )
        self.path = PathIntegrator()
        self.frame = 0
        self.seq = 0
        self._prev_ts: float | None = None
        self._prev_obs: np.ndarray | None = None
        self.geometry_version = 0
        # The window frame the frame now being reported was tracked in. A re-fit during
        # `process_frame` steps `geometry_version` past it, so anything kept alongside a
        # result - a normalized window, an orientation - must be tagged with this.
        self.tracked_version = 0
        self.centre_initial = self.centre.copy()
        self.refits: list = []
        self._moves: list = []  # one Q per window move, for the offline refinement
        self.watch = None
        if self.params.centre_watch:
            from spintrack.refit import CentreWatch

            cx, cy, radius = pixel_circle(self.camera, self.centre, self.half_angle)
            self.watch = CentreWatch((cx, cy), radius)
        self.scale_check = None
        if self.params.scale_check_stride > 0:
            from spintrack.autofit import ScaleCheck

            check = ScaleCheck(self.geometry, self.params, self.half_angle)
            self.scale_check = check if check.enabled else None

    def _camera_to_lab(self, cfg: Config) -> np.ndarray:
        """The transform named by `cfg.c2a_source()`; identity (with a warning) if none.

        Also records which key it came from in `self.c2a_source`. The camera-frame columns
        of the output are valid without a transform, so this warns rather than raising;
        `spintrack run` refuses instead, because its lab-frame columns would be camera
        values in disguise.
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
        """Start from another tracker's map and illumination fields.

        Both trackers must have been built from the same config, so their initial windows
        agree and the map needs no correction: the map lives in the window frame at
        `R = I`, and a window re-fit in `other` carried it unchanged (see
        `TrackEngine.rebuild`). The illumination fields are fixed in the *window*, so they
        do come from wherever `other` ended up and are resampled into this one.
        """
        mean, weight = other.engine.export_map()
        self.engine.load_map(mean, weight, frozen=self.cfg.map_frozen, localise=False)
        self.engine.load_illumination(
            other.engine.photometry.state(), other.centre, other.half_angle
        )

    def save_map(self, path) -> None:
        """Write the current surface map as a spintrack `.npz` template."""
        mean, weight = self.engine.export_map()
        illum = self.engine.photometry.state()
        save_map(
            path,
            mean,
            weight,
            window_size=self.geometry.size,
            centre=self.centre,
            half_angle=self.half_angle,
            illum_bias=illum["bias"],
            illum_gain=illum["gain"],
            illum_wt=illum["wt"],
        )

    def process_frame(
        self, gray: np.ndarray, ts_ms: float = -1.0, wall_ms: float | None = None
    ) -> FrameResult | None:
        """Track one grayscale frame (2-D uint8). Returns None if the frame was dropped."""
        self.tracked_version = self.geometry_version
        # The window this frame is tracked in; `_watch_centre` below may move it, and
        # the increment and orientation the step returns belong to this one.
        R_wc = self.R_wc
        window = self.geometry.remap(gray)
        # Captured before the step, which overwrites both.
        r_prev, velocity = self.engine.R, self.engine.velocity
        step = self.engine.step(window)
        self._check_scale(step, r_prev, velocity)
        self._watch_centre(gray, step)
        frame = self.frame
        self.frame += 1
        if not step.ok:
            self.seq += 1
            self._prev_ts = ts_ms
            return None
        if step.source == "reset" or (step.source == "global" and self.seq == 0):
            self.path.reset()
            self.seq = 0

        delta_ts = 0.0 if self._prev_ts is None else ts_ms - self._prev_ts
        self._prev_ts = ts_ms
        wall = ms_since_midnight() if wall_ms is None else wall_ms
        values, w_cam, w_lab, R_cam, R_lab = self._values(
            frame,
            self.seq,
            ts_ms,
            wall,
            step.w_win,
            step.R_win,
            step.cost,
            delta_ts,
            self.path,
            R_wc,
        )
        self.seq += 1
        return FrameResult(
            frame, int(values[22]), ts_ms, w_cam, w_lab, R_cam, R_lab, step, values
        )

    def refit_centre(self, new_centre) -> np.ndarray:
        """Point the tracking window at a new ball centre, keeping the surface map.

        Returns the rotation `Q` from the old window frame to the new one, which the
        caller must apply to any orientation it recorded in the old frame. The ignore
        polygons are image-fixed (the animal has not moved), so they are not translated.
        """
        centre = normalize(np.asarray(new_centre, dtype=np.float64))
        mask = source_mask(self.camera, centre, self.half_angle, self.cfg.roi_ignr)
        geometry = window_geometry(
            self.camera,
            centre,
            self.half_angle,
            self.cfg.window_size(),
            mask,
            prefilter=self.params.prefilter,
        )
        Q = geometry.to_camera.T @ self.R_wc
        self.engine.rebuild(geometry, Q)
        self.geometry = geometry
        self.R_wc = geometry.to_camera
        self.centre = centre
        self.geometry_version += 1
        self._moves.append(Q)
        self._prev_obs = None
        if self.scale_check is not None:
            from spintrack.autofit import ScaleCheck

            check = ScaleCheck(geometry, self.params, self.half_angle)
            if check.enabled:
                # The statistic is a ratio of projections within each frame, so rows taken
                # in the old window frame stay comparable with rows taken in the new one.
                check.rows = self.scale_check.rows
                self.scale_check = check
            else:
                self.scale_check = None
        return Q

    def _watch_centre(self, gray: np.ndarray, step) -> None:
        """Keep the tracking window on a ball that is moving in its holder."""
        if self.watch is None:
            return
        target = self.watch.update(self.frame, gray)
        if target is None:
            return
        cx, cy, _ = pixel_circle(self.camera, self.centre, self.half_angle)
        if float(np.hypot(target[0] - cx, target[1] - cy)) < FOLLOW_TOL_PX:
            return
        # The radius is left alone: a ball only changes apparent size by moving along the
        # optical axis, and translation on its own is much better conditioned. The target
        # is where the silhouette's circle should sit, which is not where the ball's
        # centre projects, so it is inverted rather than read as a direction.
        self.refit_centre(
            centre_from_pixel_circle(self.camera, target, self.half_angle, self.centre)
        )
        self._record_refit(target)

    def orientations_in_current_window(self, recorded) -> list:
        """Bring window-frame orientations recorded before a re-fit into the current frame.

        `recorded` is a sequence of `(R_win, geometry_version)`; every window move left a
        `Q` behind, and an orientation from version `v` needs the product of the moves
        since then applied to it.
        """
        out = []
        for R_win, version in recorded:
            if R_win is None:
                out.append(None)
                continue
            for Q in self._moves[version:]:
                R_win = Q @ R_win
            out.append(R_win)
        return out

    def _record_refit(self, target) -> None:
        """Fold this window move into the episode it belongs to, for the run summary."""
        from spintrack.refit import RefitEvent

        origin = tuple(self.watch.origin_px)
        last = self.refits[-1] if self.refits else None
        if last is not None and self.frame - last.end <= self.params.centre_watch_gap:
            last.extend(self.frame, (target[0], target[1]), self.watch.rim_fraction)
            return
        here = (float(target[0]), float(target[1]))
        event = RefitEvent(
            self.frame, self.frame, origin, here, here, self.watch.rim_fraction
        )
        self.refits.append(event)

    def _check_scale(self, step, r_prev, velocity) -> None:
        """Feed one frame-to-frame increment to the radius check, if one is due."""
        previous, self._prev_obs = self._prev_obs, self.engine.last_obs
        if self.scale_check is None or previous is None or not step.ok:
            return
        if self.frame % self.params.scale_check_stride:
            return
        self.scale_check.step(
            previous, self.engine.last_obs, r_prev, step.w_win, velocity
        )

    def _values(
        self, frame, seq, ts_ms, wall_ms, w_win, R_win, err, delta_ts, path, R_wc=None
    ):
        """The 25 FicTrac columns for one tracked frame (also advances `path`).

        `R_wc` is the window-to-camera transform that `w_win` and `R_win` are written
        in. It is not `self.R_wc` on a frame whose own window has just been re-fitted:
        pairing the new window with the old orientation rotates the reported camera-frame
        orientation by the window move.
        """
        R_wc = self.R_wc if R_wc is None else R_wc
        w_cam = R_wc @ w_win
        R_cam = R_wc @ R_win @ self.R_wc0.T
        w_lab = self.cam_to_lab @ w_cam
        R_lab = self.cam_to_lab @ R_cam @ self.cam_to_lab.T
        p = path.step(w_lab)
        values = np.empty(N_COLUMNS)
        values[0] = frame
        values[1:4] = w_cam
        values[4] = err if np.isfinite(err) else 0.0
        values[5:8] = w_lab
        values[8:11] = matrix_to_rotvec(R_cam)
        values[11:14] = matrix_to_rotvec(R_lab)
        values[14:17] = (p.pos_x, p.pos_y, p.heading)
        values[17:19] = (p.step_dir, p.step_mag)
        values[19:21] = (p.int_x, p.int_y)
        values[21] = ts_ms
        values[22] = seq
        values[23] = delta_ts
        values[24] = wall_ms
        return values, w_cam, w_lab, R_cam, R_lab

    def records_from_orientations(self, orientations, ts_list, wall_list, frames):
        """Records for a whole sequence of window-frame orientations (offline refinement).

        `orientations` may contain None for frames that stay untracked; the path integrator
        and sequence counter start fresh, as they would for a run from the first frame.
        """
        path = PathIntegrator()
        rows: list[np.ndarray] = []
        prev_R = None
        prev_ts = None
        seq = 0
        for frame, ts, wall, R_win in zip(
            frames, ts_list, wall_list, orientations, strict=True
        ):
            if R_win is None:
                seq += 1
                prev_ts = ts
                continue
            w_win = (
                np.zeros(3) if prev_R is None else matrix_to_rotvec(R_win @ prev_R.T)
            )
            delta_ts = 0.0 if prev_ts is None else ts - prev_ts
            values, *_ = self._values(
                frame, seq, ts, wall, w_win, R_win, 0.0, delta_ts, path
            )
            rows.append(values)
            prev_R, prev_ts = R_win, ts
            seq += 1
        return rows
