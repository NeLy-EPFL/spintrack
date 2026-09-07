"""Run spintrack on a dataset directory."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.sources import VideoSource
from spintrack.maps import map_directions
from spintrack.tracker import Tracker
from spintrack_bench.synth.dataset import SceneSpec, load_truth


def map_fidelity(dataset: Path, tracker, truth: dict) -> float:
    """Correlation between the recovered surface map and the true albedo.

    The map is built in the window frame the tracker started in, so the ground truth has
    to be rotated into it: window -> camera by `R_wc0`, camera -> body by the true
    orientation of the first frame. Correlation, not error, because the map holds locally
    normalized intensity rather than albedo.
    """
    spec = SceneSpec.from_json((Path(dataset) / "scene.json").read_text())
    texture = spec.texture.build(np.random.default_rng(spec.seed))
    mean, weight = tracker.engine.export_map()
    dirs = map_directions(mean.shape)
    to_body = truth["R"][0].T @ tracker.R_wc0
    gt = texture.sample((dirs.reshape(-1, 3) @ to_body.T).astype(np.float64))
    seen = (weight > 1.0).ravel()
    if seen.sum() < 100:
        return float("nan")
    return float(np.corrcoef(mean.ravel()[seen], gt.ravel()[seen])[0, 1])


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
        "n_refits": len(tracker.refits),
    }
    timing["map_corr"] = map_fidelity(dataset, tracker, truth)
    photo = tracker.engine.photometry
    if photo.active:
        field = photo.residual_field()
        seen = photo.mask & (photo.acc_n > 1.0)
        if seen.any():
            # What is left in the camera frame that the ball-fixed map cannot explain.
            timing["static_bias_rms"] = float(np.sqrt((field[seen] ** 2).mean()))
            timing["static_bias_p99"] = float(np.percentile(np.abs(field[seen]), 99))
    return est, timing
