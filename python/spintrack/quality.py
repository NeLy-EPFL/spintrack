"""Run quality: what was tracked, and where tracking was harder than elsewhere in it.

An *episode* is a stretch where both the photometric cost and the solver's iteration
count exceed their run baselines. The cost alone also rises when the ball turns fast or
shows surface the map has not seen; the iteration count rises only when the model stops
fitting.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    """A contiguous stretch of frames where tracking was harder than the baseline."""

    start: int  # frame numbers, inclusive
    end: int
    t0_s: float
    t1_s: float
    n: int
    median_cost: float
    ratio: float  # median_cost / cost_median


@dataclass
class RunQuality:
    n_frames: int
    n_tracked: int
    n_dropped: int
    cost_median: float  # over accepted frames; the episodes' baseline
    cost_p90: float
    cost_p99: float
    iters_median: float | None
    iters_p95: float | None
    sources: dict[str, int]  # map / prev / global / reset / lost counts
    turned_deg: float  # sum of |w_cam| in degrees over tracked frames
    map_coverage: float | None  # fraction of map cells with weight >= w_min
    illumination: dict | None = None  # what the static illumination field is doing
    episodes: list[Episode] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


# The `checks` entries the terminal block shows, in order; the sidecar keeps them all.
SUMMARY_CHECKS = ("ball", "vfov", "ball moved", "radius")


def running_median(x, k: int) -> np.ndarray:
    """Centered running median of a 1-D array; the edges repeat the end values."""
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or x.size == 0:
        return x.copy()
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * pad + 1)
    return np.median(windows, axis=-1)


def _runs(mask: np.ndarray, min_len: int, close_gap: int) -> list[tuple[int, int]]:
    """Index ranges of True runs in `mask`, gaps up to `close_gap` closed first."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        if i - prev - 1 > close_gap:
            out.append((start, prev))
            start = int(i)
        prev = int(i)
    out.append((start, prev))
    return [r for r in out if r[1] - r[0] + 1 >= min_len]


def _ratio_to_baseline(values, ok, smooth: int) -> np.ndarray | None:
    """Smoothed `values / median(values)` over the accepted frames.

    Frames that are not `ok`, or whose value is not finite, take the baseline value so
    that they neither raise nor lower the running median; the caller marks them
    elsewhere.
    """
    values = np.asarray(values, dtype=np.float64)
    good = ok & np.isfinite(values) & (values > 0)
    if good.sum() < 3:
        return None
    baseline = float(np.median(values[good]))
    if baseline <= 0:
        return None
    return running_median(np.where(good, values, baseline), smooth) / baseline


def episodes_above_baseline(
    cost,
    ok,
    iters=None,
    *,
    factor: float = 2.0,
    smooth: int = 11,
    min_len: int = 10,
    close_gap: int = 50,
) -> list[tuple[int, int]]:
    """Index ranges where cost (and solver effort, when known) exceed the run baseline.

    Both series are divided by their own median over accepted frames and smoothed with a
    `smooth`-frame running median; a frame is elevated when both ratios exceed `factor`,
    so the cost and the solver have to agree. Dropped frames are elevated by definition.
    Runs are merged across gaps of up to `close_gap` frames and those shorter than
    `min_len` are discarded. The defaults find no episode on still or steadily tracked
    recordings and one over a ball that drops in its holder.
    """
    cost = np.asarray(cost, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    ratio = _ratio_to_baseline(cost, ok, smooth)
    if ratio is None:
        return []
    high = ratio > factor
    if iters is not None:
        r_it = _ratio_to_baseline(iters, ok, smooth)
        if r_it is not None:
            high &= r_it > factor
    return _runs(high | ~ok, min_len, close_gap)


def _seconds(frames, ts_ms, fps: float | None) -> np.ndarray:
    """Per-frame time in seconds, from `ts_ms` when it is usable, else `frame / fps`."""
    frames = np.asarray(frames, dtype=np.float64)
    ts = np.asarray(ts_ms, dtype=np.float64) if ts_ms is not None else None
    usable = (
        ts is not None
        and ts.size == frames.size
        and np.all(np.isfinite(ts))
        and np.any(ts > 0)
        and np.all(np.diff(ts) >= 0)
    )
    if usable:
        return ts / 1e3
    return frames / float(fps) if fps and fps > 0 else np.full(frames.shape, np.nan)


def summarize_run(
    frames,
    ts_ms,
    ok,
    cost,
    iters,
    source,
    w_cam,
    fps: float | None = None,
    *,
    map_coverage: float | None = None,
    illumination: dict | None = None,
) -> RunQuality:
    """Build a `RunQuality` from the per-frame series of one run.

    `frames` must be consecutive frame numbers covering the whole recording (dropped
    frames included, with `ok` False); the other series are aligned with it. `iters` may
    be None. `w_cam` is the per-frame rotation increment in the camera frame (radians).
    """
    frames = np.asarray(frames, dtype=np.int64)
    ok = np.asarray(ok, dtype=bool)
    cost = np.asarray(cost, dtype=np.float64)
    good = ok & np.isfinite(cost)
    accepted = cost[good]
    if accepted.size == 0:
        accepted = np.zeros(1)
    it_median = it_p95 = None
    if iters is not None:
        it = np.asarray(iters, dtype=np.float64)[ok]
        it = it[np.isfinite(it) & (it > 0)]
        if it.size:
            it_median = float(np.median(it))
            it_p95 = float(np.percentile(it, 95))
    counts = Counter(s for s in source if s) if source is not None else {}
    w = np.asarray(w_cam, dtype=np.float64).reshape(-1, 3)[ok]
    turned = float(np.degrees(np.linalg.norm(w, axis=1)).sum()) if w.size else 0.0

    seconds = _seconds(frames, ts_ms, fps)
    baseline = float(np.median(accepted))
    episodes = []
    for a, b in episodes_above_baseline(cost, ok, iters):
        block = cost[a : b + 1][good[a : b + 1]]
        median = float(np.median(block)) if block.size else float("nan")
        episodes.append(
            Episode(
                start=int(frames[a]),
                end=int(frames[b]),
                t0_s=float(seconds[a]),
                t1_s=float(seconds[b]),
                n=b - a + 1,
                median_cost=median,
                ratio=median / baseline if baseline > 0 else float("nan"),
            )
        )
    return RunQuality(
        n_frames=int(frames.size),
        n_tracked=int(ok.sum()),
        n_dropped=int((~ok).sum()),
        cost_median=baseline,
        cost_p90=float(np.percentile(accepted, 90)),
        cost_p99=float(np.percentile(accepted, 99)),
        iters_median=it_median,
        iters_p95=it_p95,
        sources=dict(counts),
        turned_deg=turned,
        map_coverage=map_coverage,
        illumination=illumination,
        episodes=episodes,
    )


def format_summary(q: RunQuality) -> str:
    """The terminal block: frames, the ball's checks, the hard-tracking episodes."""
    lines = [
        f"run quality: {q.n_frames} frames, {q.n_tracked} tracked, "
        + f"{q.n_dropped} dropped"
    ]
    for key in SUMMARY_CHECKS:
        if isinstance(q.checks.get(key), str):
            lines.append(f"{key}: {q.checks[key]}")
    if not q.episodes:
        lines.append("hard tracking: none")
    else:
        parts = []
        for ep in q.episodes[:3]:
            when = f", {ep.t0_s:.1f}-{ep.t1_s:.1f} s" if np.isfinite(ep.t0_s) else ""
            parts.append(
                f"frames {ep.start}-{ep.end}{when} ({ep.ratio:.1f}x the median cost)"
            )
        more = len(q.episodes) - 3
        lines.append(
            "hard tracking: "
            + "; ".join(parts)
            + (f"; {more} more" if more > 0 else "")
        )
    return "\n".join(lines)


def write_sidecar(path: str | Path, q: RunQuality, provenance: dict) -> Path:
    """Write the run's JSON summary to `path`; returns the path written."""
    from spintrack import __version__

    path = Path(path)
    payload = {
        "spintrack": __version__,
        "provenance": provenance,
        "quality": asdict(q),
    }
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path
