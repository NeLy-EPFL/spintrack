"""Where does the photometric cost put the ball's radius, and where does truth put it?

Scales a scene's `roi_circ` about its own centre and reports, per scale, the median cost
and the rotation gain against ground truth. The question it answers is whether the cost
minimum can be used as evidence about the radius at all: if it sits at the true radius on
scenes where the truth is known, then a disagreement on real data is about the real data.

    uv run --group bench python -m spintrack_bench.geometry_sweep clean_fly offaxis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker
from spintrack_bench.metrics import frame_errors_deg

DATA = Path("benchmarks/data")
SCALES = tuple(np.round(np.arange(0.90, 1.101, 0.02), 3))
DEFAULT_SCENES = ("clean_fly", "offaxis", "lab_small_ball", "occluded", "sparse")


def scaled_points(points: np.ndarray, scale: float) -> list[int]:
    """`roi_circ` scaled about the circle its points describe."""
    a = np.stack([2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))], 1)
    sol, *_ = np.linalg.lstsq(a, (points**2).sum(axis=1), rcond=None)
    centre = np.array([sol[0], sol[1]])
    return [round(float(v)) for xy in centre + (points - centre) * scale for v in xy]


def track(scene: str, points: list[int], n_frames: int):
    cfg = Config.load(DATA / scene / "config.txt")
    cfg.roi_circ, cfg.roi_c, cfg.roi_r = points, None, None
    source = VideoSource(DATA / scene / "video.mp4")
    tracker = Tracker(cfg, source.width, source.height)
    estimates, costs = [], []
    for _ in range(n_frames):
        frame = source.read()
        if frame is None:
            break
        result = tracker.process_frame(frame.image, frame.ts_ms)
        estimates.append(result.w_cam if result is not None else [np.nan] * 3)
        if result is not None:
            costs.append(result.step.cost)
    source.close()
    return np.array(estimates), float(np.nanmedian(costs))


def parabolic_minimum(x: np.ndarray, y: np.ndarray) -> float:
    """Location of the minimum of the parabola through the three lowest samples."""
    best = int(np.argmin(y))
    if best in (0, len(y) - 1):
        return float(x[best])
    i, j, k = best - 1, best, best + 1
    denom = y[i] - 2.0 * y[j] + y[k]
    if abs(denom) < 1e-15:
        return float(x[j])
    return float(x[j] - 0.5 * (x[j + 1] - x[j]) * (y[k] - y[i]) / denom)


def sweep(scene: str, n_frames: int) -> None:
    cfg = Config.load(DATA / scene / "config.txt")
    points = np.asarray(cfg.roi_circ, dtype=np.float64).reshape(-1, 2)
    truth = np.load(DATA / scene / "truth.npz")
    w_true = truth["w_cam"]
    print(f"\n=== {scene} ({n_frames} frames) ===")
    print(f"{'radius':>8} {'median cost':>12} {'rotation gain':>14} {'median err':>11}")
    costs = []
    for scale in SCALES:
        estimates, cost = track(scene, scaled_points(points, float(scale)), n_frames)
        true = w_true[: len(estimates)]
        ok = np.isfinite(estimates).all(axis=1)
        magnitude_true = np.linalg.norm(true[ok], axis=1)
        magnitude_est = np.linalg.norm(estimates[ok], axis=1)
        gain = float(magnitude_est @ magnitude_true / (magnitude_true @ magnitude_true))
        error = float(np.nanmedian(frame_errors_deg(estimates, true)))
        costs.append(cost)
        print(f"{100 * (scale - 1):+7.0f}% {cost:12.5f} {gain:14.4f} {error:11.4f}")
    best = parabolic_minimum(np.array(SCALES), np.array(costs))
    print(f"  cost minimum at {100 * (best - 1):+.1f}% of the true radius")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args(argv)
    for scene in args.scenes or DEFAULT_SCENES:
        sweep(scene, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
