"""Run spintrack on a real recording (video + FicTrac config) and return `.dat` rows."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker


def track_video(
    config_path: Path,
    video_path: Path | None = None,
    overrides: dict | None = None,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Returns (dat rows (M, 25), timing dict)."""
    config_path = Path(config_path)
    cfg = Config.load(config_path)
    params = TrackParams()
    names = {f.name for f in fields(TrackParams)}
    for key, value in (overrides or {}).items():
        if key in names:
            setattr(params, key, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    video = Path(video_path) if video_path else (config_path.parent / cfg.src_fn)
    src = VideoSource(video)
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
    }
    return np.array(rows), timing
