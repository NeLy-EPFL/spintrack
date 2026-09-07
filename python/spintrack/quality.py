"""Run quality summary: cost statistics, map coverage and elevated-cost episodes.

An *episode* is a contiguous stretch where tracking was measurably harder than in the rest
of the same run. Two per-frame signals go into it, both expressed as a ratio to the run's
own robust baseline: the photometric `cost` and the number of Gauss-Newton iterations the
solver needed. A frame counts as elevated only when **both** are, because the cost on its
own also rises whenever the ball simply turns fast or shows surface the map has not seen
yet - on real trials those bouts reach 4-6x the run baseline, as high as an actual
geometry failure. The iteration count does not move for them (3-4 either way) and jumps to
8-10 when the model no longer fits. See `episodes_above_baseline` for the measured
constants.

`.dat` files carry no iteration count, so `summary_from_dat` falls back to the cost alone
and says so in `RunQuality.notes`; episodes found that way are far less specific.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spintrack.io.dat import read_dat


@dataclass
class Episode:
    """A contiguous stretch of frames where tracking was harder than the run baseline."""

    start: int  # frame numbers, inclusive
    end: int
    t0_s: float
    t1_s: float
    n: int
    median_cost: float
    ratio: float  # median_cost / cost_baseline


@dataclass
class RunQuality:
    n_frames: int
    n_tracked: int
    n_dropped: int
    cost_baseline: float  # median cost over accepted frames
    cost_median: float
    cost_p90: float
    cost_p99: float
    iters_median: float | None  # None when unavailable (summarize from .dat)
    iters_p95: float | None
    sources: dict[str, int]  # map / prev / global / reset / lost counts
    turned_deg: float  # sum of |w_cam| in degrees over tracked frames
    map_coverage: float | None  # fraction of map cells with weight >= w_min
    # Peak of the static illumination field, in normalized-intensity units, and the
    # fraction of the window it dims by more than half. None when the correction is off.
    illum_peak: float | None = None
    illum_dim_frac: float | None = None
    episodes: list[Episode] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


COVERAGE_NOTE = (
    "Unseen surface is a cap around the axis the ball turned least about; it shrinks "
    "as the ball turns further and does not indicate bad tracking."
)
COST_ONLY_NOTE = (
    "No solver iteration counts (summarised from a .dat): episodes come from the cost "
    "alone and also fire on fast turning and on map warm-up."
)


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
    that they neither raise nor lower the running median; the caller marks them elsewhere.
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
    `min_len` are discarded.

    The defaults were tuned on the three `AN07B017_260414_Fly4` trials and on the 17
    synthetic benchmark scenes: no episode on any synthetic scene nor on trials 003 and
    005, and 004's ball drop comes out as frames 1134-1643 at 4.7x the run's cost
    baseline. Dropping the iteration term instead makes 003 report six episodes of the
    same apparent severity, and thresholding the cost on its own spread (a MAD term) only
    truncates 004's episode, because that spread is what the episode itself creates.
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
    notes = []
    if map_coverage is not None:
        notes.append(COVERAGE_NOTE)
    if iters is None:
        notes.append(COST_ONLY_NOTE)
    return RunQuality(
        n_frames=int(frames.size),
        n_tracked=int(ok.sum()),
        n_dropped=int((~ok).sum()),
        cost_baseline=baseline,
        cost_median=baseline,
        cost_p90=float(np.percentile(accepted, 90)),
        cost_p99=float(np.percentile(accepted, 99)),
        iters_median=it_median,
        iters_p95=it_p95,
        sources=dict(counts),
        turned_deg=turned,
        map_coverage=map_coverage,
        episodes=episodes,
        notes=notes,
        **(illumination or {}),
    )


def summary_from_dat(path: str | Path, fps: float | None = None) -> RunQuality:
    """Summarise an existing `.dat`. Gaps in the frame column count as dropped frames."""
    data = read_dat(path)
    if data.shape[0] == 0:
        raise ValueError(f"{path}: no records")
    written = data[:, 0].astype(np.int64)
    frames = np.arange(written[0], written[-1] + 1)
    at = np.searchsorted(frames, written)
    ok = np.zeros(frames.size, dtype=bool)
    ok[at] = True
    cost = np.full(frames.size, np.nan)
    cost[at] = data[:, 4]
    ts = np.interp(frames, written, data[:, 21])
    w_cam = np.zeros((frames.size, 3))
    w_cam[at] = data[:, 1:4]
    # The .dat does not record which solve produced a row; only sequence resets show.
    source = np.full(frames.size, "", dtype=object)
    source[at[np.flatnonzero(np.diff(data[:, 22]) < 0) + 1]] = "reset"
    return summarize_run(frames, ts, ok, cost, None, source, w_cam, fps)


def format_summary(q: RunQuality) -> str:
    """The terminal block (at most ~15 lines)."""
    lines = [
        f"run quality: {q.n_frames} frames, {q.n_tracked} tracked, {q.n_dropped} dropped",
        f"cost: median {q.cost_median:.4g}, p90 {q.cost_p90:.4g}, p99 {q.cost_p99:.4g}",
    ]
    if q.iters_median is not None:
        lines.append(
            f"solver: {q.iters_median:.0f} iterations median, {q.iters_p95:.0f} at p95"
        )
    if q.sources:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(q.sources.items()))
        lines.append(f"sources: {parts}")
    turned = f"ball turned {q.turned_deg:.0f} deg"
    if q.map_coverage is not None:
        lines.append(
            f"map coverage: {100 * q.map_coverage:.1f}% of the sphere, {turned}"
        )
    else:
        lines.append(turned.capitalize())
    if q.illum_peak is not None:
        dimmed = (
            f", {100 * q.illum_dim_frac:.1f}% of the window down-weighted below half"
            if q.illum_dim_frac is not None
            else ""
        )
        lines.append(
            f"illumination: static field peaks at {q.illum_peak:.2f} of a "
            f"normalized-intensity unit{dimmed}"
        )
    for key, value in q.checks.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
    if not q.episodes:
        lines.append("no episodes above the run baseline")
    else:
        lines.append(f"episodes above the run baseline: {len(q.episodes)}")
        for ep in q.episodes[:5]:
            span = (
                f"{ep.t0_s:.1f}-{ep.t1_s:.1f} s" if np.isfinite(ep.t0_s) else "no times"
            )
            lines.append(
                f"  frames {ep.start}-{ep.end} ({span}, {ep.n} frames): "
                f"cost {ep.ratio:.1f}x baseline"
            )
        if len(q.episodes) > 5:
            lines.append(f"  ... and {len(q.episodes) - 5} more (see the sidecar)")
    lines.extend(f"note: {n}" for n in q.notes)
    return "\n".join(lines)


def write_sidecar(path: str | Path, q: RunQuality, provenance: dict) -> Path:
    """Write the JSON sidecar next to the `.dat`; returns the path written."""
    from spintrack import __version__

    path = Path(path)
    payload = {
        "spintrack": __version__,
        "provenance": provenance,
        "quality": asdict(q),
    }
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path
