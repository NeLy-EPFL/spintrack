"""Offline refinement: rebuild the map from every frame, then re-solve every frame.

Online tracking builds the map incrementally, so early errors are baked into it and
frame-to-frame modes drift. Given all normalized windows and the online orientations, each
sweep splats every frame into a map at its current orientation and re-aligns every frame
against it. Two or three sweeps are usually enough.

The map a frame is aligned against is *temporally local*: an exponential window of
`LOCAL_TAU` frames on either side of it, the frame itself left out. A map of the whole
recording is sharper in principle and blurrier in practice - the lighting, the shading and
the animal's shadow change over a recording, and a mean over all of it no longer looks like
the texture any one frame sees. Judged by an optical-flow cross-check on trial 003, refining
against the whole recording made the per-frame increments worse than the online ones (0.59
px rms against 0.43) and the local map makes them better (0.41 px); on `ball_drop`, with
exact truth, the episode error goes from 0.053 deg online to 0.049 (whole recording:
0.053). Fifty frames measures best of 50, 150 and 500. The two sides of the window are two
passes: forward, where the map of the frames before each one is checkpointed every
`CHECKPOINT_EVERY` frames and replayed by block, and backward, where each frame is solved
and then folded into the map of the frames after it. The memory is a block of maps, not one
per frame.

Windows recorded before the tracking window moved (see `spintrack.refit`) are in the
window frame they were tracked in, and so must their orientations be: a window's pixels
map to the same surface directions wherever the window sits, so a window paired with an
orientation expressed in a later window frame is splatted rotated by the move. On
`ball_drop` that pairing cost the refined episode 0.088 deg per frame against 0.070 with
each frame kept in its own frame. `versions` and `moves` carry the frames, and the refined
orientations come back in the same per-frame frames.

A re-solve can land in a wrong minimum several degrees away and still fit well enough to be
accepted: on trial 003 seventeen frames did, and the refined output carried out-and-back
spikes of 9-12 degrees that the online run never had. A frame whose orientation is far from
both of its neighbors while the neighbors agree with each other is such a spike - the ball
cannot go there and back within a frame - and it keeps its pre-sweep orientation instead.
There is no cost gate: the pre-filtered window makes a still frame's cost a tenth of a
moving one's, so a multiple of the median rejects the frames where the animal walks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spintrack.engine import TrackEngine
from spintrack.geometry import matrix_to_rotvec, rotation_angle

LOCAL_TAU = 50.0  # frames; the e-folding of the map's window on either side of a frame
CHECKPOINT_EVERY = 64  # frames between the forward maps kept for the backward pass
# A spike is a jump into and out of one frame larger than this multiple of the median
# frame-to-frame increment (with a floor), while the two neighbors agree to within this
# fraction of the smaller of the two jumps.
SPIKE_FACTOR = 5.0
SPIKE_MIN_DEG = 1.0
SPIKE_AGREEMENT = 0.5


def bring_forward(
    R: np.ndarray, from_version: int, to_version: int, moves
) -> np.ndarray:
    """Re-express a window-frame orientation across the window moves between two versions."""
    for Q in moves[from_version:to_version]:
        R = Q @ R
    return R


def refine_orientations(
    engine: TrackEngine,
    windows: Sequence[np.ndarray],
    orientations: Sequence[np.ndarray | None],
    sweeps: int = 2,
    progress=None,
    versions: Sequence[int] | None = None,
    moves: Sequence[np.ndarray] | None = None,
    tau: float | None = LOCAL_TAU,
) -> tuple[list[np.ndarray | None], dict]:
    """Return refined window-frame orientations (None where a frame stays untrackable).

    `windows` are normalized windows (as produced by `TrackEngine.normalize`, any float dtype);
    `orientations` are the online estimates, with None for dropped frames (they are seeded
    from the nearest earlier estimate and solved against the map). Each is in the window
    frame its frame was tracked in: `versions[i]` names that frame and `moves[v]` is the
    rotation from window frame `v` to `v + 1` (`Tracker._moves`), both optional when the
    window never moved. The result is in the same per-frame frames. `tau` is the map's
    temporal window in frames; None uses every other frame of the recording.
    """
    n = len(windows)
    R = list(orientations)
    versions = [0] * n if versions is None else list(versions)
    moves = [] if moves is None else list(moves)
    decay = float(np.exp(-1.0 / tau)) if tau and np.isfinite(tau) else 1.0
    obs = [np.ascontiguousarray(w, dtype=np.float32) for w in windows]
    stats = {"sweeps": sweeps, "recovered": 0, "rejected": 0, "spikes": 0}
    for sweep in range(sweeps):
        seeds, checkpoints = _forward_pass(engine.core, obs, R, versions, moves, decay)
        new_R, costs = _backward_pass(
            engine, obs, R, seeds, checkpoints, decay, progress, sweep
        )
        stats["spikes"] += revert_spikes(new_R, R, versions, moves)
        stats[f"median_cost_sweep{sweep}"] = (
            float(np.median(costs)) if costs else float("nan")
        )
        R = new_R
    stats["recovered"] = sum(
        1 for a, b in zip(orientations, R, strict=True) if a is None and b is not None
    )
    stats["rejected"] = sum(1 for r in R if r is None)
    return R, stats


def _export(core) -> tuple[np.ndarray, np.ndarray]:
    return core.map_mean().copy(), core.map_weight().copy()


def _splat(core, window: np.ndarray, R: np.ndarray, decay: float) -> None:
    """Fold one frame into the core's map, forgetting the rest by `decay` first."""
    if decay < 1.0:
        core.set_map(core.map_mean(), core.map_weight() * np.float32(decay))
    core.update(window, R, lambda_=1.0, w_max=1e6, update_main=True)


def _forward_pass(core, obs, R, versions, moves, decay):
    """Seeds for every frame, and the map of the frames before each checkpoint."""
    core.reset()
    last, last_version = np.eye(3), 0
    seeds: list[np.ndarray] = []
    checkpoints: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(obs)):
        if i % CHECKPOINT_EVERY == 0:
            checkpoints.append(_export(core))
        if R[i] is not None:
            last, last_version = R[i], versions[i]
            _splat(core, obs[i], last, decay)
        seeds.append(bring_forward(last, last_version, versions[i], moves))
    return seeds, checkpoints


def _backward_pass(engine, obs, R, seeds, checkpoints, decay, progress, sweep):
    """Solve every frame against the frames before it and the frames after it."""
    p = engine.params
    core = engine.core
    n = len(obs)
    new_R: list[np.ndarray | None] = [None] * n
    costs: list[float] = []
    after_mean, after_weight = (np.zeros_like(a) for a in checkpoints[0])
    for block in reversed(range(len(checkpoints))):
        start = block * CHECKPOINT_EVERY
        stop = min(start + CHECKPOINT_EVERY, n)
        # The maps of the frames before each frame of the block, replayed from the
        # checkpoint with the pre-sweep orientations, as the forward pass built them.
        core.set_map(*checkpoints[block])
        before = []
        for i in range(start, stop):
            before.append(_export(core))
            if R[i] is not None:
                _splat(core, obs[i], R[i], decay)
        for i in reversed(range(start, stop)):
            before_mean, before_weight = before[i - start]
            weight = before_weight + after_weight
            mean = np.where(
                weight > 0.0,
                (before_mean * before_weight + after_mean * after_weight)
                / np.maximum(weight, 1e-12),
                0.0,
            ).astype(np.float32)
            core.set_map(mean, weight)
            res = core.solve(
                obs[i],
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
                new_R[i] = np.asarray(res.r, dtype=np.float64)
                costs.append(res.cost)
            else:
                new_R[i] = R[i]
            if new_R[i] is not None:
                # This frame now belongs to the map of the frames after the next one.
                core.set_map(after_mean, after_weight * np.float32(decay))
                core.update(obs[i], new_R[i], lambda_=1.0, w_max=1e6, update_main=True)
                after_mean, after_weight = _export(core)
            if progress is not None and i % 1000 == 0:
                progress(sweep, i, n)
    return new_R, costs


def revert_spikes(new_R, old_R, versions, moves) -> int:
    """Put back the pre-sweep orientation of every frame that is a one-frame excursion.

    Returns how many were reverted. `new_R` is edited in place.
    """
    tracked = [i for i in range(len(new_R)) if new_R[i] is not None]
    if len(tracked) < 3:
        return 0

    def jump(a: int, b: int) -> float:
        R_a = bring_forward(new_R[a], versions[a], versions[b], moves)
        return rotation_angle(new_R[b] @ R_a.T)

    jumps = np.array([jump(tracked[k - 1], tracked[k]) for k in range(1, len(tracked))])
    limit = max(np.radians(SPIKE_MIN_DEG), SPIKE_FACTOR * float(np.median(jumps)))
    reverted = 0
    for k in range(1, len(tracked) - 1):
        into, out = jumps[k - 1], jumps[k]
        if into <= limit or out <= limit:
            continue
        if jump(tracked[k - 1], tracked[k + 1]) < SPIKE_AGREEMENT * min(into, out):
            new_R[tracked[k]] = old_R[tracked[k]]
            reverted += 1
    return reverted


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
