"""Offline refinement: rebuild the map from every frame, then re-solve every frame.

Online tracking builds the map incrementally, so early errors are baked into it and
frame-to-frame modes drift. Given all normalized windows and the online orientations, each
sweep splats every frame into a fresh map at its current orientation and re-aligns every
frame against that complete map. Two or three sweeps are usually enough.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spintrack.engine import TrackEngine
from spintrack.geometry import matrix_to_rotvec


def refine_orientations(
    engine: TrackEngine,
    windows: Sequence[np.ndarray],
    orientations: Sequence[np.ndarray | None],
    sweeps: int = 2,
    progress=None,
) -> tuple[list[np.ndarray | None], dict]:
    """Return refined window-frame orientations (None where a frame stays untrackable).

    `windows` are normalized windows (as produced by `TrackEngine.normalize`, any float dtype);
    `orientations` are the online estimates, with None for dropped frames (they are seeded
    from the nearest earlier estimate and solved against the complete map).
    """
    p = engine.params
    core = engine.core
    n = len(windows)
    R = list(orientations)
    stats = {"sweeps": sweeps, "recovered": 0, "rejected": 0}
    for sweep in range(sweeps):
        core.reset()
        last = np.eye(3)
        seeds: list[np.ndarray] = []
        for i in range(n):
            if R[i] is not None:
                last = R[i]
                core.update(
                    np.ascontiguousarray(windows[i], dtype=np.float32),
                    last,
                    lambda_=1.0,
                    w_max=1e6,
                    update_main=True,
                )
            seeds.append(last)
        new_R: list[np.ndarray | None] = []
        cost_of: list[float | None] = []
        costs = []
        for i in range(n):
            obs = np.ascontiguousarray(windows[i], dtype=np.float32)
            res = core.solve(
                obs,
                seeds[i],
                [0.0, 0.0, 0.0],
                levels=1 if R[i] is not None else None,
                max_iter=p.max_iter,
                tol=p.tol,
                huber=p.huber,
                tukey=p.tukey,
                w_min=p.w_min,
                w_sat=p.w_sat,
                reweight_iters=p.reweight_iters,
            )
            ok = (
                np.all(np.isfinite(res.w))
                and res.overlap >= p.min_overlap
                and res.inlier_frac >= p.min_inlier_frac
                and np.isfinite(res.cost)
            )
            if ok:
                new_R.append(np.asarray(res.r, dtype=np.float64))
                costs.append(res.cost)
                cost_of.append(res.cost)
            else:
                new_R.append(R[i])
                cost_of.append(None)
            if progress is not None and i % 1000 == 0:
                progress(sweep, i, n)
        # Cost gate against wrong minima: solutions far above the typical cost keep their
        # previous orientation (or stay untracked).
        if costs:
            level = float(np.median(costs))
            for i in range(n):
                if (
                    new_R[i] is not None
                    and cost_of[i] is not None
                    and cost_of[i] > 3.0 * level
                ):
                    new_R[i] = R[i]
        stats[f"median_cost_sweep{sweep}"] = (
            float(np.median(costs)) if costs else float("nan")
        )
        R = new_R
    stats["recovered"] = sum(
        1 for a, b in zip(orientations, R, strict=True) if a is None and b is not None
    )
    stats["rejected"] = sum(1 for r in R if r is None)
    return R, stats


def increments(orientations: Sequence[np.ndarray | None]) -> list[np.ndarray | None]:
    """Per-frame rotation vectors `log(R_t R_prev^T)` relative to the previous tracked frame."""
    out: list[np.ndarray | None] = []
    prev = None
    for R in orientations:
        if R is None:
            out.append(None)
            continue
        out.append(np.zeros(3) if prev is None else matrix_to_rotvec(R @ prev.T))
        prev = R
    return out
