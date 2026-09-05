"""Per-frame tracking state machine on top of the compiled solver.

`TrackEngine.step` takes the remapped grayscale window, normalizes it, aligns it against
the accumulated surface map (falling back to the previous frame's map), updates the maps
and returns the rotation increment in the window frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from spintrack._core import Engine as _Engine
from spintrack.geometry import matrix_to_rotvec
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
    norm_win_pc: float = 0.25  # normalization window, fraction of the window size
    norm_floor: float = 2.0  # gray-level floor added to the local std
    map_scale: float = 1.5  # map height = map_scale * window size; width = 2 * height
    map_lambda: float = 1.0  # forgetting factor of the accumulated map (1 = none)
    map_w_max: float = 50.0
    forget_outside_view: bool = False  # FicTrac fork's `accumulate_map: n`
    forget_margin: int = 1
    min_overlap: float = 0.25
    min_inlier_frac: float = 0.5
    max_cost: float = float("inf")
    # Reject a solve whose cost exceeds this multiple of the running cost of accepted
    # frames (catches wrong local minima on repetitive textures); 0 disables.
    cost_gate: float = 3.0
    cost_gate_warmup: int = 10
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


class TrackEngine:
    def __init__(self, geometry: WindowGeometry, params: TrackParams | None = None):
        self.geometry = geometry
        self.params = params or TrackParams()
        n = geometry.size
        map_h = round(self.params.map_scale * n)
        map_w = 2 * map_h
        self.map_shape = (map_h, map_w)
        self.core = _Engine(
            np.ascontiguousarray(geometry.surface, dtype=np.float32),
            np.ascontiguousarray(geometry.index, dtype=np.int64),
            n,
            map_w,
            map_h,
            self.params.levels,
            self.params.max_pixels,
        )
        self._mask = geometry.mask.astype(np.float32)
        k = max(3, round(self.params.norm_win_pc * n) | 1)
        self._ksize = (k, k)
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
        self._needs_localisation = False
        self.last_obs: np.ndarray | None = None  # normalized window of the last step
        self._cost_level: float | None = None  # running mean cost of accepted frames

    # ----- maps -----
    def load_map(
        self, mean: np.ndarray, weight: np.ndarray, frozen: bool = False
    ) -> None:
        """Start from a saved map; the first frame is localised against it globally."""
        if mean.shape != self.map_shape or weight.shape != self.map_shape:
            raise ValueError(f"map arrays must have shape {self.map_shape}")
        self.core.set_map(
            np.ascontiguousarray(mean, dtype=np.float32),
            np.ascontiguousarray(weight, dtype=np.float32),
        )
        self._have_map = True
        self._frozen = frozen
        self._needs_localisation = True
        self.R = np.eye(3)
        self.velocity = np.zeros(3)

    def export_map(self) -> tuple[np.ndarray, np.ndarray]:
        return self.core.map_mean(), self.core.map_weight()

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
        gated = p.cost_gate > 0 and self._cost_level is not None
        if gated and self.frames_tracked >= p.cost_gate_warmup:
            return cost <= p.cost_gate * self._cost_level
        return True

    def _note_cost(self, cost: float) -> None:
        if np.isfinite(cost):
            if self._cost_level is None:
                self._cost_level = cost
            else:
                self._cost_level = 0.9 * self._cost_level + 0.1 * cost

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

    def step(self, window: np.ndarray) -> StepResult:
        """Track one remapped grayscale window (uint8, window_size x window_size)."""
        obs = self.normalize(window)
        self.last_obs = obs
        if self._needs_localisation:
            return self._localise(obs)
        if not self._have_map:
            self.R = np.eye(3)
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
    def map_image(self) -> np.ndarray:
        """uint8 rendering of the accumulated map (unseen cells mid-gray)."""
        mean = self.core.map_mean()
        weight = self.core.map_weight()
        img = np.clip(128 + 40 * mean, 0, 255).astype(np.uint8)
        img[weight < self.params.w_min] = 128
        return img
