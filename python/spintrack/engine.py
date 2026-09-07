"""Per-frame tracking state machine on top of the compiled solver.

`TrackEngine.step` takes the remapped grayscale window, normalizes it, aligns it against
the accumulated surface map (falling back to the previous frame's map), updates the maps
and returns the rotation increment in the window frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from spintrack._core import Engine as _Engine
from spintrack.camera import EquidistantCamera, pixel_centres
from spintrack.geometry import (
    matrix_to_rotvec,
    normalize,
    rotation_between,
    rotvec_to_matrix,
)
from spintrack.photometry import Photometry
from spintrack.sphere import WindowGeometry


@dataclass
class TrackParams:
    """Tunable solver and map parameters (defaults are the benchmark-tuned values)."""

    levels: int = 3
    max_iter: int = 10
    tol: float = 3e-4  # rad; GN stops when the step is smaller
    huber: float = 1.345  # Huber constant, in units of the robust residual scale
    tukey: float = 4.685  # Tukey constant, same units (0 disables)
    w_min: float = 0.1  # map weight below which a cell counts as unseen
    w_sat: float = 3.0
    damping: float = 1e-6
    # Anti-aliasing blur of the source before the window remap, in units of the decimation
    # (0 disables; see `sphere.PREFILTER_SIGMA`).
    prefilter: float = 0.5
    norm_win_pc: float = 0.25  # normalization window, fraction of the window size
    norm_floor: float = 2.0  # gray-level floor added to the local std
    # `cube` is an equi-angular cubemap; `equal_area` is the Lambert cylindrical grid
    # FicTrac uses, kept because a comparison against FicTrac wants the same tessellation.
    # The cube trades a little resolution at the equator for the 8.6-degree cells the
    # cylindrical grid puts at its poles, and is the default because it measures better:
    # see `rust/src/map.rs` and `benchmarks/spintrack_bench/map_grid_sweep.py`.
    map_projection: str = "cube"
    map_scale: float = 1.5  # cells = 2 * (map_scale * window size)^2, either projection
    map_lambda: float = 1.0  # forgetting factor of the accumulated map (1 = none)
    # Body frame the map is stored in, as a rotation vector from the initial window frame.
    # The identity puts the grid's poles - its one badly shaped region, see `map.rs` - at
    # the top and bottom of the first frame's view. Nothing depends on the choice, which
    # is exactly what makes it the control for `spintrack_bench.map_grid_sweep poles`.
    map_frame: tuple[float, float, float] = (0.0, 0.0, 0.0)
    map_w_max: float = 50.0
    forget_outside_view: bool = False  # FicTrac fork's `accumulate_map: n`
    forget_margin: int = 1
    # Static camera-frame illumination (see `photometry.py`). The arms are independent,
    # and `benchmarks/spintrack_bench/` (`photometry_sweep` on synthetic scenes with
    # ground truth, `illumination_real` on lab recordings) measures them separately and in
    # combination; only the bias field earned its default.
    illum_bias: bool = True  # additive field, estimated from the map residual
    # Multiplicative field scaling the model. Worth 5% of median tracking error on the
    # synthetic shadow scenes, but per-pixel gain and map amplitude are only separable
    # when the ball turns enough to mix the two, and on the lab recordings it does not:
    # over 4000 frames the field spreads without converging (p95 1.13 -> 1.35, max -> 2.4)
    # instead of settling. Off until it is conditioned on something.
    illum_gain: bool = False
    # Per-pixel inverse-noise-variance weight. It lowers the reported cost, but that is
    # partly circular - the cost is weighted by the same weights - and on ground truth it
    # is not an improvement.
    illum_weight: bool = False
    # Temporal flat field of the raw window, applied before the local normalization. It
    # flattens the shading further than the bias field does, but at ~3 effectively
    # independent samples per pixel its estimate contains texture, and subtracting that
    # raises the photometric residual by up to 60% on the lab recordings.
    illum_flat: bool = False
    # Accumulate the fields for reporting without applying any of them.
    illum_measure: bool = False
    illum_tau: float = 500.0  # frames; memory of the fields and of their accumulators
    illum_update_every: int = 25  # frames between field refreshes
    # Fold in every n-th accepted frame. The field has a time constant of hundreds
    # of frames, so sampling costs it nothing and keeps the estimator off the hot
    # path (every frame is ~0.6 ms at q_factor 12, every fourth is ~0.15 ms).
    illum_stride: int = 4
    # Forget the fields when a window re-fit moves them this far, as a fraction of
    # the ball's radius: past that the ball has moved enough to be lit differently.
    illum_reset_move: float = 0.05
    illum_warmup: int = 100  # frames before the first refresh
    illum_smooth: float = 0.0  # window px; spatial smoothing of the fields (0 = none)
    illum_gain_damping: float = 0.25
    illum_gain_min: float = 0.4
    illum_gain_max: float = 2.5
    illum_weight_min: float = 0.05
    illum_weight_max: float = 4.0
    illum_flat_tau: float = 2000.0
    illum_flat_sigma: float = 4.0  # window px; smoothing of the raw flat field
    min_overlap: float = 0.25
    min_inlier_frac: float = 0.5
    max_cost: float = float("inf")
    # Reject a solve whose cost exceeds this multiple of the 90th percentile of the last
    # `cost_gate_history` accepted map solves, so that a wrong local minimum after a jump
    # the local solve cannot follow falls through to the previous-frame solve or the
    # global search. Off by default (0): with the pre-filtered window the cost is bimodal
    # - a still frame's is a tenth of a moving one's (trial 003: median 0.006, p90 0.072)
    # - so any level learned while the animal stands rejects the frames where it walks. A
    # running mean rejected 700 of 4000 frames on 003 (cross-check correlation 0.999 ->
    # 0.95); the percentile still sent 277 to the previous-frame solve there and lost 54
    # frames of 004's drop, while on the 21 synthetic scenes the gate never fires at all.
    # The failure it guards against is a rotation of tens of degrees within one frame.
    cost_gate: float = 0.0
    cost_gate_warmup: int = 10
    cost_gate_history: int = 200
    max_step: float = 0.5  # rad per frame; larger increments are rejected
    velocity_smoothing: float = 0.5  # 0 disables the constant-velocity prediction
    # Solve on a spatial subsample of about this many window pixels (None = all). The map
    # is still updated from every pixel. 4000 keeps q_factor 12 under 1 ms/frame with a
    # small precision cost; set None for maximum precision.
    max_pixels: int | None = 4000
    fine_first: bool = True  # solve at full resolution first, pyramid only on failure
    level_tol_factor: float = 4.0  # tolerance relaxation per coarser pyramid level
    coarse_max_iter: int = 3
    # Iterations per level during which robust weights are recomputed (then frozen). With
    # residual-scaled thresholds, always reweighting converges best; keep it large.
    reweight_iters: int = 1000
    max_bad_frames: int = -1
    # Re-solve every n-th frame on an inner disc and an outer annulus of the window, as an
    # independent check that the ball's assumed radius is right (0 disables). It reads the
    # engine's state and never writes to it, so tracking output is bit-for-bit unchanged.
    scale_check_stride: int = 10
    # Measure the ball's silhouette on every frame and re-fit the tracking window onto a
    # ball that moves in its holder (see `spintrack.refit`).
    centre_watch: bool = True
    centre_watch_gap: int = (
        100  # frames of stillness that end a "the ball moved" episode
    )
    # Relocalise against the map when both solves fail (FicTrac's opt_do_global).
    global_search: bool = False
    global_candidates: int = 2000
    extra: dict = field(default_factory=dict)


@dataclass
class StepResult:
    ok: bool
    source: str  # "reset", "map", "prev" or "lost"
    w_win: np.ndarray  # rotation increment, window frame (zeros when not ok)
    R_win: np.ndarray  # absolute ball orientation, window frame
    cost: float = float("nan")
    rms: float = float("nan")
    inlier_frac: float = float("nan")
    overlap: float = float("nan")
    iters: int = 0
    converged: bool = False
    hessian: np.ndarray | None = None


def map_shape(projection: str, scale: float, window: int) -> tuple[int, int]:
    """`(h, w)` of the surface map, with the same cell count either way.

    `2 * (scale * window)^2` cells: a `scale * window` by `2 * scale * window` rectangle,
    or six faces of `scale * window / sqrt(3)` a side stacked into `(6 face, face)`.
    """
    if projection == "cube":
        face = max(round(scale * window / np.sqrt(3.0)), 1)
        return 6 * face, face
    if projection != "equal_area":
        raise ValueError(f"unknown map projection {projection!r}")
    h = round(scale * window)
    return h, 2 * h


class TrackEngine:
    def __init__(self, geometry: WindowGeometry, params: TrackParams | None = None):
        self.geometry = geometry
        self.params = params or TrackParams()
        p = self.params
        n = geometry.size
        self.map_shape = map_shape(p.map_projection, p.map_scale, n)
        self.core = self._new_core(geometry)
        self._mask = geometry.mask.astype(np.float32)
        k = max(3, round(self.params.norm_win_pc * n) | 1)
        self._ksize = (k, k)
        self.photometry = Photometry(geometry.mask, self.params)
        # Where the illumination fields were learned: the window's optical axis in the
        # camera frame, averaged with the fields' own memory (see `_resample_photometry`).
        self._illum_axis = geometry.to_camera[:, 2].copy()
        self._illum_axis_frame = 0
        self.reset()

    # ----- state -----
    def reset(self) -> None:
        self.R = rotvec_to_matrix(self.params.map_frame)
        self.velocity = np.zeros(3)
        self.n_bad = 0
        self.frames_tracked = 0
        self.core.reset()
        self._have_map = False
        self._frozen = False
        self._needs_localisation = False
        self.last_obs: np.ndarray | None = None  # normalized window of the last step
        self._costs: deque[float] = deque(maxlen=self.params.cost_gate_history)

    def _new_core(self, geometry: WindowGeometry) -> _Engine:
        """Build the Rust core for `geometry`. The one place its arguments are chosen: a
        window re-fit builds a second one, and the two must not drift apart."""
        map_h, map_w = self.map_shape
        return _Engine(
            np.ascontiguousarray(geometry.surface, dtype=np.float32),
            np.ascontiguousarray(geometry.index, dtype=np.int64),
            geometry.size,
            map_w,
            map_h,
            self.params.levels,
            self.params.max_pixels,
            self.params.map_projection,
        )

    # ----- maps -----
    def load_map(
        self,
        mean: np.ndarray,
        weight: np.ndarray,
        frozen: bool = False,
        localise: bool = True,
    ) -> None:
        """Start from a saved map; the first frame is localised against it globally.

        `localise=False` says the caller already knows the map is in this engine's body
        frame - the map of an earlier pass over the same recording, say - so the first
        frame is solved from `R = I` like any other instead of searching all of SO(3).
        """
        if mean.shape != self.map_shape or weight.shape != self.map_shape:
            raise ValueError(f"map arrays must have shape {self.map_shape}")
        self.core.set_map(
            np.ascontiguousarray(mean, dtype=np.float32),
            np.ascontiguousarray(weight, dtype=np.float32),
        )
        self._have_map = True
        self._frozen = frozen
        self._needs_localisation = localise
        self.R = rotvec_to_matrix(self.params.map_frame)
        self.velocity = np.zeros(3)

    def export_map(self) -> tuple[np.ndarray, np.ndarray]:
        return self.core.map_mean(), self.core.map_weight()

    def rebuild(self, geometry: WindowGeometry, Q: np.ndarray) -> None:
        """Move to a new tracking window, carrying the map and the current orientation.

        The map lives in the window frame at `R = I`, which is the ball's body frame, so a
        new window only changes the coordinates the orientation is written in: `Q` maps
        the old window frame to the new one. The previous-frame map cannot be carried (the
        new core has never seen a frame), which costs one frame of the `prev` fallback.

        `last_obs` is kept: the frame it came from was tracked, and it is a valid
        observation in the old window frame, exactly like the orientation `Q` is applied
        to. Its consumers pair it with the version the frame was tracked at
        (`Tracker.tracked_version`); dropping it here instead cost the refine buffer and
        the debug video every frame whose window moved.
        """
        mean, weight = self.export_map()
        old_geometry = self.geometry
        p = self.params
        if map_shape(p.map_projection, p.map_scale, geometry.size) != self.map_shape:
            raise ValueError("rebuild needs a window of the same size")
        self.geometry = geometry
        self.core = self._new_core(geometry)
        self.core.set_map(
            np.ascontiguousarray(mean, dtype=np.float32),
            np.ascontiguousarray(weight, dtype=np.float32),
        )
        self._mask = geometry.mask.astype(np.float32)
        self._resample_photometry(
            old_geometry.size,
            old_geometry.rad_per_pixel,
            Q,
            old_geometry.to_camera[:, 2],
        )
        self.R = Q @ self.R
        self.velocity = Q @ self.velocity

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
        floor = max(self.params.norm_floor, 0.05 * global_std)
        out = (img - mean) / (np.sqrt(var) + floor)
        out[~valid] = 0.0
        return np.ascontiguousarray(out, dtype=np.float32)

    # ----- per frame -----
    def _solve(self, obs, w0, use_prev, levels=None):
        p = self.params
        return self.core.solve(
            obs,
            self.R,
            [float(v) for v in w0],
            use_prev=use_prev,
            levels=levels,
            level_tol_factor=p.level_tol_factor,
            coarse_max_iter=p.coarse_max_iter,
            reweight_iters=p.reweight_iters,
            max_iter=p.max_iter,
            tol=p.tol,
            huber=p.huber,
            tukey=p.tukey,
            w_min=p.w_min,
            w_sat=p.w_sat,
            damping=p.damping,
        )

    def _cost_ok(self, cost: float) -> bool:
        p = self.params
        if not np.isfinite(cost) or cost > p.max_cost:
            return False
        if p.cost_gate > 0 and len(self._costs) >= max(p.cost_gate_warmup, 1):
            return cost <= p.cost_gate * float(np.percentile(self._costs, 90))
        return True

    def _note_cost(self, cost: float) -> None:
        if np.isfinite(cost):
            self._costs.append(float(cost))

    def _accept(self, res) -> bool:
        p = self.params
        w = np.asarray(res.w)
        return bool(
            np.all(np.isfinite(w))
            and np.linalg.norm(w) <= p.max_step
            and res.overlap >= p.min_overlap
            and res.inlier_frac >= p.min_inlier_frac
            and self._cost_ok(res.cost)
        )

    def _accept_global(self, res) -> bool:
        p = self.params
        return bool(
            np.all(np.isfinite(res.w))
            and res.overlap >= p.min_overlap
            and res.inlier_frac >= p.min_inlier_frac
            and self._cost_ok(res.cost)
        )

    def _update_maps(self, obs: np.ndarray) -> None:
        p = self.params
        # Measure the static field against the map the tracker just used, before this
        # frame is folded into it: otherwise the residual is partly self-referential.
        self.photometry.observe(self.core, self.R, obs)
        self.core.update(
            obs,
            self.R,
            lambda_=p.map_lambda,
            w_max=p.map_w_max,
            forget_outside=p.forget_outside_view and not self._frozen,
            margin=p.forget_margin,
            update_main=not self._frozen,
        )
        self._have_map = True

    def _localise(self, obs: np.ndarray) -> StepResult:
        """First frame against a loaded map: find the absolute orientation globally."""
        p = self.params
        try:
            res = self.core.global_search(
                obs,
                n_candidates=p.global_candidates,
                max_iter=p.max_iter,
                tol=p.tol,
                huber=p.huber,
                tukey=p.tukey,
                w_min=p.w_min,
                w_sat=p.w_sat,
                min_overlap=p.min_overlap,
            )
        except ValueError:
            res = None
        if res is None or not self._accept_global(res):
            self.n_bad += 1
            if p.max_bad_frames >= 0 and self.n_bad > p.max_bad_frames:
                self.reset()
            return StepResult(False, "lost", np.zeros(3), self.R.copy())
        self.R = np.asarray(res.r, dtype=np.float64)
        self.velocity = np.zeros(3)
        self._needs_localisation = False
        self._update_maps(obs)
        self._note_cost(res.cost)
        self.n_bad = 0
        self.frames_tracked += 1
        return StepResult(
            True, "global", np.zeros(3), self.R.copy(), res.cost, res.rms, res.inlier_frac,
            res.overlap, res.iters, res.converged, np.asarray(res.hessian),
        )  # fmt: skip

    def _resample_photometry(
        self, old_size: int, old_rad_per_pixel: float, Q: np.ndarray, old_axis=None
    ) -> None:
        """Carry the illumination fields into the current window.

        Unlike the map, these fields are fixed in the *window*, so a different window is a
        resampling, not a rotation: each new window pixel takes the value of whichever old
        window pixel looked in the same direction. `Q` maps the old window frame to the
        new one; `old_axis` is the old window's optical axis in the camera frame (None for
        fields loaded from another run, which are always resampled).
        """
        geometry = self.geometry
        n = geometry.size
        axis = geometry.to_camera[:, 2]
        if old_axis is not None:
            # A ball that has moved in its holder is lit differently - shading follows the
            # surface normal, and translating the ball changes it - so a field measured
            # before the move is not merely displaced, it is wrong. Past a fraction of a
            # ball radius, drop it and re-learn rather than carry a shadow to where there
            # is none. The distance is measured from where the field was *learned*: the
            # window's axis averaged with the field's own memory, so that a follower moving
            # the window a pixel per frame still trips the reset once the ball has gone
            # far enough within a time constant, while a drift the field has had time to
            # absorb does not.
            since = self.frames_tracked - self._illum_axis_frame
            decay = np.exp(-since / max(self.params.illum_tau, 1.0))
            learned = normalize(decay * self._illum_axis + (1.0 - decay) * old_axis)
            self._illum_axis_frame = self.frames_tracked
            moved = float(np.arccos(np.clip(learned @ axis, -1.0, 1.0)))
            if moved > self.params.illum_reset_move * 0.5 * n * geometry.rad_per_pixel:
                self.photometry.forget()
                self.photometry.push(self.core)
                self._illum_axis = axis.copy()
                return
            self._illum_axis = learned
        old_cam = EquidistantCamera(old_size, old_size, old_rad_per_pixel)
        new_cam = EquidistantCamera(n, n, geometry.rad_per_pixel)
        xs, ys = pixel_centres(n, n)
        dirs_old = new_cam.rays(xs, ys) @ Q  # Q.T applied row-wise
        x, y, _ = old_cam.project(dirs_old)
        self.photometry.resample(
            (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32), geometry.mask
        )
        self.photometry.push(self.core)

    def load_illumination(
        self, fields: dict, centre: np.ndarray, half_angle: float
    ) -> None:
        """Start from illumination fields measured in another run on the same rig.

        They were measured in the window that run aimed at *its* ball centre, so they are
        resampled into this one. Which is the whole reason the centre and angular radius
        are stored beside them: applied as they stand, a field measured half a ball radius
        away would put the holder's shadow somewhere the holder is not.
        """
        if not fields:
            return
        n = self.geometry.size
        old_cam = EquidistantCamera.from_extent(n, 2.0 * float(half_angle))
        old_to_camera = rotation_between(np.array([0.0, 0.0, 1.0]), normalize(centre))
        self.photometry.load(**fields)
        self._resample_photometry(
            n, old_cam.rad_per_pixel, self.geometry.to_camera.T @ old_to_camera
        )

    def step(self, window: np.ndarray) -> StepResult:
        """Track one remapped grayscale window (uint8, window_size x window_size)."""
        photo = self.photometry
        if photo.flat is not None:
            window = photo.flat.apply(window)
        obs = photo.correct(self.normalize(window))
        self.last_obs = obs
        if self._needs_localisation:
            return self._localise(obs)
        if not self._have_map:
            self.R = rotvec_to_matrix(self.params.map_frame)
            self.velocity = np.zeros(3)
            self._update_maps(obs)
            self.frames_tracked += 1
            return StepResult(True, "reset", np.zeros(3), self.R.copy())

        p = self.params
        w0 = self.velocity if p.velocity_smoothing > 0 else np.zeros(3)
        source = "map"
        if p.fine_first:
            res = self._solve(obs, w0, use_prev=False, levels=1)
            # A solve that stalled just above tolerance is still a good solution; only
            # escalate to the pyramid when the last step was clearly large.
            settled = res.converged or (
                len(res.steps) > 0 and res.steps[-1] < 5.0 * p.tol
            )
            ok = settled and self._accept(res)
            if not ok and p.levels > 1:
                res = self._solve(obs, w0, use_prev=False)
                ok = self._accept(res)
        else:
            res = self._solve(obs, w0, use_prev=False)
            ok = self._accept(res)
        if not ok:
            alt = self._solve(obs, w0, use_prev=True)
            if self._accept(alt):
                res, ok, source = alt, True, "prev"
        if not ok and p.global_search and self.frames_tracked > 1:
            try:
                alt = self.core.global_search(
                    obs,
                    n_candidates=p.global_candidates,
                    max_iter=p.max_iter,
                    tol=p.tol,
                    huber=p.huber,
                    tukey=p.tukey,
                    w_min=p.w_min,
                    w_sat=p.w_sat,
                    min_overlap=p.min_overlap,
                )
            except ValueError:
                alt = None
            if alt is not None and self._accept_global(alt):
                res, ok, source = alt, True, "global"
        if not ok:
            self.n_bad += 1
            if p.max_bad_frames >= 0 and self.n_bad > p.max_bad_frames:
                self.reset()
            return StepResult(False, "lost", np.zeros(3), self.R.copy(), res.cost, res.rms,
                              res.inlier_frac, res.overlap, res.iters, res.converged)  # fmt: skip

        R_new = np.asarray(res.r, dtype=np.float64)
        # Increment relative to the previous orientation (global relocalisation may jump).
        w = (
            matrix_to_rotvec(R_new @ self.R.T)
            if source == "global"
            else np.asarray(res.w)
        )
        w = np.asarray(w, dtype=np.float64)
        self.R = R_new
        a = p.velocity_smoothing
        if source == "global":
            self.velocity = np.zeros(3)
        else:
            self.velocity = (1.0 - a) * self.velocity + a * w if a > 0 else np.zeros(3)
        self._update_maps(obs)
        if source == "map":
            # A solve against the previous frame's map scores lower than one against the
            # accumulated map; letting it into the level ratchets the gate shut.
            self._note_cost(res.cost)
        self.n_bad = 0
        self.frames_tracked += 1
        return StepResult(
            True,
            source,
            w,
            self.R.copy(),
            res.cost,
            res.rms,
            res.inlier_frac,
            res.overlap,
            res.iters,
            res.converged,
            np.asarray(res.hessian, dtype=np.float64),
        )

    # ----- inspection -----
    def map_coverage(self) -> float:
        """Fraction of map cells seen at least `w_min`.

        Read as a fraction of the surface unweighted, which the equal-area grid makes exact
        and the cubemap makes true to within the 1.41:1 spread of its cell solid angles.
        """
        return float(np.mean(self.core.map_weight() >= self.params.w_min))

    def illumination_image(self) -> np.ndarray | None:
        """uint8 rendering of the static bias field, on the same scale as the map."""
        photo = self.photometry
        if not photo.active:
            return None
        field = photo.bias if self.params.illum_bias else photo.residual_field()
        from spintrack.maps import CONTRAST, UNSEEN

        img = np.clip(UNSEEN + CONTRAST * field, 0, 255).astype(np.uint8)
        img[~photo.mask] = UNSEEN
        return img

    def map_image(self, layout: str = "grid") -> np.ndarray:
        """uint8 rendering of the accumulated map (unseen cells mid-gray)."""
        from spintrack.maps import render_map

        return render_map(*self.export_map(), self.params.w_min, layout)
