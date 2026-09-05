"""Run spintrack on a dataset directory."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker
from spintrack_bench.synth.dataset import load_truth


def run_spintrack(
    dataset: Path, overrides: dict | None = None
) -> tuple[np.ndarray, dict]:
    """Track `dataset/video.mp4`; return (est_cam (N,3) with NaN for dropped frames, timing)."""
    dataset = Path(dataset)
    cfg = Config.load(dataset / "config.txt")
    params = TrackParams()
    param_names = {f.name for f in fields(TrackParams)}
    for key, value in (overrides or {}).items():
        if key in param_names:
            setattr(params, key, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    truth = load_truth(dataset)
    n = len(truth["w_cam"])
    src = VideoSource(dataset / cfg.src_fn)
    tracker = Tracker(cfg, src.width, src.height, params)
    est = np.full((n, 3), np.nan)
    t_track = 0.0
    t0 = time.perf_counter()
    iters = []
    sources = {"map": 0, "prev": 0, "lost": 0, "reset": 0}
    for frame in src:
        t1 = time.perf_counter()
        res = tracker.process_frame(frame.image, frame.ts_ms)
        t_track += time.perf_counter() - t1
        if res is not None:
            if frame.index < n:
                est[frame.index] = res.w_cam
            iters.append(res.step.iters)
            sources[res.step.source] += 1
        else:
            sources["lost"] += 1
    wall = time.perf_counter() - t0
    src.close()
    n_frames = max(1, tracker.frame)
    timing = {
        "wall_s": wall,
        "fps_reported": n_frames / wall,
        "evals_per_frame": float(np.mean(iters)) if iters else np.nan,
        "tracking_ms_per_frame": 1e3 * t_track / n_frames,
        "frames_prev_fallback": sources["prev"],
        "frames_lost": sources["lost"],
    }
    return est, timing
