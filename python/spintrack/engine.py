"""Per-frame tracking state machine on top of the compiled solver.

`TrackEngine.step` takes the remapped grayscale window, normalizes it, aligns it against
the accumulated surface map (falling back to the pyramid, the previous frame's map and,
if enabled, a global search), updates the maps and returns the rotation increment in the
window frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spintrack._core import MAX_ITER, TOL, W_MIN
from spintrack._core import Engine as _Engine
from spintrack.camera import EquidistantCamera, pixel_centers
from spintrack.geometry import normalize, rotation_between
from spintrack.photometry import RESET_MOVE, TAU, Photometry
from spintrack.sphere import PREFILTER_SIGMA, WindowGeometry

# Gray-level floor added to the local standard deviation in `normalize`.
NORM_FLOOR = 2.0
# Map cells per window pixel along a side: `2 * (MAP_SCALE * window)^2` cells in all,
# FicTrac's map density.
MAP_SCALE = 1.5
# Loaded maps are priors: their weights are capped just above `W_MIN` so that the first
# fresh observation of a cell replaces what was loaded.
MAP_PRIOR_W_MAX = 0.15
# A solve is accepted when this much of the window overlaps seen map cells and this much
# of the overlap is inliers.
MIN_OVERLAP = 0.25
MIN_INLIER_FRAC = 0.5
# Weight of the newest increment in the constant-velocity prediction.
VELOCITY_SMOOTHING = 0.5
# Gauss-Newton budget of the first frame against a loaded map, in units of `MAX_ITER`:
# it may start a couple of degrees from where that map puts the ball.
FIRST_FRAME_ITERATIONS = 5
# A pixel's weight in the map goes as the cosine of its viewing angle to this power:
# foreshortened views near the limb are blurred and would wash out the map.
VIEW_WEIGHT_POWER = 2.0


@dataclass
class TrackParams:
    """The tracking options a config key, a CLI flag or a caller sets."""

    norm_win_pc: float = 0.25  # normalization window, fraction of the window size
    forget_outside_view: bool = False  # FicTrac fork's `accumulate_map: n`
    illum_bias: bool = True  # separate the rig's static illumination (`photometry`)
    global_search: bool = False  # relocalize when the local solves fail (opt_do_global)
    max_step: float = 0.5  # rad per frame; larger increments are rejected
    max_bad_frames: int = -1  # lost frames in a row before a reset (-1: never)
    # Solve on a spatial subsample of about this many window pixels (None = all); the
    # map is still updated from every pixel.
    max_pixels: int | None = 4000
    # Anti-aliasing blur of the source before the window remap, in units of the
    # decimation (0 disables).
    prefilter: float = PREFILTER_SIGMA
    # Follow a ball that moves in its holder (see `spintrack.refit`).
    center_watch: bool = True


@dataclass
class StepResult:
    ok: bool
    source: str  # "reset", "map", "prev", "global" or "lost"
    w_win: np.ndarray  # rotation increment, window frame (zeros when not ok)
    R_win: np.ndarray  # absolute ball orientation, window frame
    cost: float = float("nan")
    inlier_frac: float = float("nan")
    overlap: float = float("nan")
    iters: int = 0
    converged: bool = False


def map_face(window: int) -> int:
    """Cells along a cube face side for a `window`-pixel tracking window."""
    return max(round(MAP_SCALE * window / np.sqrt(3.0)), 1)


class TrackEngine:
    def __init__(self, geometry: WindowGeometry, params: TrackParams | None = None):
        self.geometry = geometry
        self.params = params or TrackParams()
        n = geometry.size
        self.face = map_face(n)
        self.map_shape = (6 * self.face, self.face)
        self.core = self._new_core(geometry)
        self._mask = geometry.mask.astype(np.float32)
        k = max(3, round(self.params.norm_win_pc * n) | 1)
        self._ksize = (k, k)
        self.photometry = Photometry(geometry.mask, self.params.illum_bias)
        # `_map_weight`'s view weight, and the geometry it was computed for.
        self._view_weight = None
        self._view_weight_for = None
        # Where the illumination field was learned: the window's optical axis in the
        # camera frame, averaged with the field's own memory (see
        # `_resample_photometry`).
        self._illum_axis = geometry.to_camera[:, 2].copy()
        self._illum_axis_frame = 0
        self.reset()

    # ----- state -----
    def reset(self) -> None:
        self.R = np.eye(3)
        self.velocity = np.zeros(3)
        self.n_bad = 0
        self.frames_tracked = 0
        self.core.reset()
        self._have_map = False
        self._frozen = False
        self._needs_localization = False
        self.last_obs: np.ndarray | None = None  # normalized window of the last step

    def _new_core(self, geometry: WindowGeometry) -> _Engine:
        """The Rust core for `geometry`; a window re-fit builds another the same way."""
        return _Engine(
            np.ascontiguousarray(geometry.surface, dtype=np.float32),
            np.ascontiguousarray(geometry.index, dtype=np.int64),
            geometry.size,
            self.face,
            self.params.max_pixels,
        )

    # ----- maps -----
    def load_map(
        self,
        mean: np.ndarray,
        weight: np.ndarray,
        frozen: bool = False,
        localize: bool = True,
    ) -> None:
        """Start from a saved map; the first frame is localized against it globally.

        `localize=False` says the caller already knows the map is in this engine's body
        frame - the map of an earlier pass over the same recording, say - so the first
        frame is solved from `R = I` like any other instead of searching all of SO(3).
        """
        if mean.shape != self.map_shape or weight.shape != self.map_shape:
            raise ValueError(f"map arrays must have shape {self.map_shape}")
        self.core.set_map(
            np.ascontiguousarray(mean, dtype=np.float32),
            np.ascontiguousarray(np.minimum(weight, MAP_PRIOR_W_MAX), dtype=np.float32),
        )
        self._have_map = True
        self._frozen = frozen
        self._needs_localization = localize
        self.R = np.eye(3)
        self.velocity = np.zeros(3)

    def export_map(self) -> tuple[np.ndarray, np.ndarray]:
        return self.core.map_mean(), self.core.map_weight()

    def rebuild(self, geometry: WindowGeometry, Q: np.ndarray) -> None:
        """Move to a new tracking window, carrying the map and the current orientation.

        The map lives in the window frame at `R = I`, which is the ball's body frame, so
        a new window only changes the coordinates the orientation is written in: `Q`
        maps the old window frame to the new one. The previous-frame map cannot be
        carried (the new core has never seen a frame), which costs one frame of the
        `prev` fallback. `last_obs` is kept: it is still the last tracked window.
        """
        if map_face(geometry.size) != self.face:
            raise ValueError("rebuild needs a window of the same size")
        mean, weight = self.export_map()
        old_geometry = self.geometry
        self.geometry = geometry
        self.core = self._new_core(geometry)
        self.core.set_map(mean, weight)
        self._mask = geometry.mask.astype(np.float32)
        self._resample_photometry(
            old_geometry.size,
            old_geometry.rad_per_pixel,
            Q,
            old_geometry.to_camera[:, 2],
        )
        self.R = Q @ self.R
        # A move's own apparent rotation must not seed the next solve: carried over, a
        # wrong move fed the prediction and the solver held the wrong increment.
        self.velocity = np.zeros(3)

    # ----- normalization -----
    def normalize(self, window: np.ndarray) -> np.ndarray:
        """Local zero-mean, unit-variance normalization over valid pixels (float32)."""
        img = window.astype(np.float32)
        m = self._mask
        s1 = cv2.boxFilter(img * m, -1, self._ksize, normalize=False)
        s2 = cv2.boxFilter(img * img * m, -1, self._ksize, normalize=False)
        cnt = cv2.boxFilter(m, -1, self._ksize, normalize=False)
        cnt = np.maximum(cnt, 1.0)
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean * mean, 0.0)
        valid = m > 0
        global_std = float(np.sqrt(var[valid].mean())) if valid.any() else 1.0
        floor = max(NORM_FLOOR, 0.05 * global_std)
        out = (img - mean) / (np.sqrt(var) + floor)
        out[~valid] = 0.0
        return np.ascontiguousarray(out, dtype=np.float32)

    # ----- per frame -----
    def _solve(self, obs, use_prev=False, levels=None, max_iter=MAX_ITER):
        w0 = [float(v) for v in self.velocity]
        return self.core.solve(obs, self.R, w0, use_prev, levels, max_iter)

    def _accept(self, res, max_step: float) -> bool:
        w = np.asarray(res.w)
        return bool(
            np.all(np.isfinite(w))
            and np.linalg.norm(w) <= max_step
            and res.overlap >= MIN_OVERLAP
            and res.inlier_frac >= MIN_INLIER_FRAC
            and np.isfinite(res.cost)
        )

    def _solve_local(self, obs: np.ndarray):
        """`(result, source)` of the first acceptable local solve, `(None, "lost")` if
        none is: the finest level, then the pyramid, then the previous frame's map."""
        max_step = self.params.max_step
        first = self.frames_tracked == 0
        max_iter = FIRST_FRAME_ITERATIONS * MAX_ITER if first else MAX_ITER
        res = self._solve(obs, levels=1, max_iter=max_iter)
        # A solve that stalled just above tolerance is still a good solution; only
        # escalate to the pyramid when the last step was clearly large.
        settled = res.converged or res.last_step < 5.0 * TOL
        if settled and self._accept(res, max_step):
            return res, "map"
        res = self._solve(obs, max_iter=max_iter)
        if self._accept(res, max_step):
            return res, "map"
        res = self._solve(obs, use_prev=True)
        if self._accept(res, max_step):
            return res, "prev"
        return None, "lost"

    def _global_search(self, obs: np.ndarray):
        """Relocalize against the whole map; None if nothing acceptable was found."""
        try:
            res = self.core.global_search(obs, MIN_OVERLAP)
        except ValueError:
            return None
        return res if self._accept(res, np.inf) else None

    def _map_weight(self) -> np.ndarray:
        """Each window pixel's weight in the map: the lighting gain's, times how
        squarely the pixel sees the surface."""
        g = self.geometry
        if self._view_weight_for is not g:
            w = np.zeros(g.size * g.size, np.float32)
            w[g.index] = np.clip(g.facing, 0.0, 1.0) ** VIEW_WEIGHT_POWER
            self._view_weight = w.reshape(g.size, g.size)
            self._view_weight_for = g
        gain = self.photometry.weights()
        if gain is None:
            return self._view_weight
        return np.ascontiguousarray(gain * self._view_weight, dtype=np.float32)

    def _update_maps(self, obs: np.ndarray) -> None:
        # Measure the static field against the map the tracker just used, before this
        # frame is folded into it: otherwise the residual is partly self-referential.
        self.photometry.observe(self.core, self.R, obs)
        self.core.update(
            obs,
            self.R,
            forget_outside=self.params.forget_outside_view and not self._frozen,
            update_main=not self._frozen,
            weight=self._map_weight(),
        )
        self._have_map = True

    def _resample_photometry(
        self, old_size: int, old_rad_per_pixel: float, Q: np.ndarray, old_axis=None
    ) -> None:
        """Carry the illumination field into the current window.

        Unlike the map, the field is fixed in the *window*, so a different window is a
        resampling, not a rotation: each new window pixel takes the value of whichever
        old window pixel looked in the same direction. `Q` maps the old window frame to
        the new one; `old_axis` is the old window's optical axis in the camera frame
        (None for a field loaded from another run, which is always resampled).
        """
        geometry = self.geometry
        n = geometry.size
        axis = geometry.to_camera[:, 2]
        if old_axis is not None:
            # A ball that moved in its holder is lit differently, so past a fraction of
            # its radius the field is dropped rather than carried. The move is measured
            # from where the field was learned (the axis averaged over the field's
            # memory), so that many small window moves add up.
            since = self.frames_tracked - self._illum_axis_frame
            decay = np.exp(-since / TAU)
            learned = normalize(decay * self._illum_axis + (1.0 - decay) * old_axis)
            self._illum_axis_frame = self.frames_tracked
            moved = float(np.arccos(np.clip(learned @ axis, -1.0, 1.0)))
            if moved > RESET_MOVE * 0.5 * n * geometry.rad_per_pixel:
                self.photometry.forget()
                self._illum_axis = axis.copy()
                return
            self._illum_axis = learned
        old_cam = EquidistantCamera(old_size, old_size, old_rad_per_pixel)
        new_cam = EquidistantCamera(n, n, geometry.rad_per_pixel)
        xs, ys = pixel_centers(n, n)
        dirs_old = new_cam.rays(xs, ys) @ Q  # Q.T applied row-wise
        x, y, _ = old_cam.project(dirs_old)
        self.photometry.resample(
            (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32), geometry.mask
        )

    def load_illumination(
        self,
        fields: dict,
        center: np.ndarray,
        half_angle: float,
        prior_frames: float = 0.0,
    ) -> None:
        """Start from an illumination field measured in another run on the same rig.

        It was measured in the window that run aimed at *its* ball center, so it is
        resampled into this one, using the center and angular radius stored beside it.
        """
        if not fields:
            return
        n = self.geometry.size
        old_cam = EquidistantCamera.from_extent(n, 2.0 * float(half_angle))
        old_to_camera = rotation_between(np.array([0.0, 0.0, 1.0]), normalize(center))
        self.photometry.load(**fields, prior_frames=prior_frames)
        self._resample_photometry(
            n, old_cam.rad_per_pixel, self.geometry.to_camera.T @ old_to_camera
        )

    def observation(self, window: np.ndarray) -> np.ndarray:
        """The normalized, illumination-corrected window the solver compares."""
        return self.photometry.correct(self.normalize(window))

    def _lost(self) -> StepResult:
        self.n_bad += 1
        if 0 <= self.params.max_bad_frames < self.n_bad:
            self.reset()
        return StepResult(False, "lost", np.zeros(3), self.R.copy())

    def step(self, window: np.ndarray) -> StepResult:
        """Track one remapped grayscale window (uint8, window_size x window_size)."""
        obs = self.observation(window)
        self.last_obs = obs
        if not self._have_map:
            self.R = np.eye(3)
            self.velocity = np.zeros(3)
            self._update_maps(obs)
            self.frames_tracked += 1
            return StepResult(True, "reset", np.zeros(3), self.R.copy())
        if self._needs_localization:
            res, source = self._global_search(obs), "global"
        else:
            res, source = self._solve_local(obs)
            if res is None and self.params.global_search and self.frames_tracked > 1:
                res, source = self._global_search(obs), "global"
        if res is None:
            return self._lost()
        self._needs_localization = False
        # A relocalization, or the first frame against a loaded map, fixes where the
        # ball is rather than measuring a rotation: report no motion, as FicTrac does.
        jump = source == "global" or self.frames_tracked == 0
        w = np.zeros(3) if jump else np.asarray(res.w, dtype=np.float64)
        self.R = np.asarray(res.r, dtype=np.float64)
        a = VELOCITY_SMOOTHING
        self.velocity = np.zeros(3) if jump else (1.0 - a) * self.velocity + a * w
        self._update_maps(obs)
        self.n_bad = 0
        self.frames_tracked += 1
        return StepResult(
            True, source, w, self.R.copy(), res.cost, res.inlier_frac, res.overlap,
            res.iters, res.converged,
        )  # fmt: skip

    # ----- inspection -----
    def map_coverage(self) -> float:
        """Fraction of map cells seen, which is the fraction of the surface to within
        the 1.41:1 spread of the cube's cell solid angles."""
        return float(np.mean(self.core.map_weight() >= W_MIN))

    def illumination_image(self) -> np.ndarray | None:
        """uint8 rendering of the lighting gain field: 170 where a pixel delivers the
        window's full texture contrast, darker in a shadow."""
        photo = self.photometry
        if not photo.enabled:
            return None
        img = np.clip(170.0 * photo.gain, 0, 255).astype(np.uint8)
        img[~photo.mask] = 0
        return img

    def map_image(self, layout: str = "grid") -> np.ndarray:
        """uint8 rendering of the accumulated map (unseen cells mid-gray)."""
        from spintrack.maps import render_map

        return render_map(*self.export_map(), W_MIN, layout)
