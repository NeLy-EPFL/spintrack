"""Run spintrack on a real recording (video + FicTrac config) and return its records.

The lab's recordings come with FicTrac configs, which are translated on the fly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from spintrack.engine import TrackParams
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker
from spintrack_bench.fictrac_config import read_fictrac_config, to_spintrack
from spintrack_bench.runners.spintrack_runner import apply_overrides


def track_video(
    config_path: Path,
    video_path: Path | None = None,
    overrides: dict | None = None,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict, Tracker]:
    """Returns (records (M, 25), timing dict, the tracker)."""
    config_path = Path(config_path)
    src_fn = read_fictrac_config(config_path)["src_fn"]
    src = VideoSource(Path(video_path) if video_path else config_path.parent / src_fn)
    cfg = to_spintrack(config_path, (src.width, src.height))
    params = TrackParams()
    apply_overrides(cfg, params, overrides or {})
    tracker = Tracker(cfg, src.width, src.height, params)
    rows = []
    t_track = 0.0
    t0 = time.perf_counter()
    n = 0
    lost = 0
    for frame in src:
        t1 = time.perf_counter()
        res = tracker.process_frame(frame.image, frame.ts_ms, frame.wall_ms)
        t_track += time.perf_counter() - t1
        n += 1
        if res is None:
            lost += 1
        else:
            rows.append(res.values)
        if max_frames is not None and n >= max_frames:
            break
    wall = time.perf_counter() - t0
    src.close()
    timing = {
        "frames": n,
        "lost": lost,
        "wall_s": wall,
        "fps_total": n / wall if wall > 0 else np.nan,
        "tracking_ms_per_frame": 1e3 * t_track / max(n, 1),
        # A ball that moved in its holder: FicTrac's fixed window reads that as
        # rotation.
        "ball_moves": len(getattr(tracker.watch, "episodes", [])),
    }
    return np.array(rows), timing, tracker
