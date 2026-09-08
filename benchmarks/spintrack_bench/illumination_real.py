"""What each illumination correction does on the real recordings, where there is no truth.

Ground truth exists only on the synthetic scenes (`spintrack_bench.photometry_sweep`), so
on lab video the readouts are:

`shading` is the temporal mean of the corrected window over the whole recording, per
window pixel. Whatever the tracker still sees that does not rotate lands here, and its
profile down the window is the shape the eye picks out of the video: on these recordings
the bottom band reads about -0.9 in normalized-intensity units before correction. A few
percent of it is texture that never averaged away, the ball turning as little as it does,
but that part is common to every arm, so the comparison between them is fair.

`static` is the same quantity taken against the map rather than against zero, so it counts
only what the ball-fixed map could not absorb. It reads much lower, because a map free to
smear the shadow over the surface absorbs most of it - which is the problem, not the fix.
It is only evidence with `--holdout`: while the bias field is live it is the running
mean that this residual is measured against, so the column reads ~0 for every arm with
`A` in it, whatever the field contains (see `Photometry.residual_field`).

`cost` is the photometric residual the solver reports and `map_std` the map's own
contrast; both fall when the map stops carrying illumination it never should have had.

`--holdout` is the honest test of whether an arm has separated lighting from texture or
has merely absorbed texture: the fields are estimated on the first half, frozen, and the
numbers are reported on the second half alone. An arm that has learned real texture cannot
transfer.

    uv run --group bench python -m spintrack_bench.illumination_real
    uv run --group bench python -m spintrack_bench.illumination_real --holdout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spintrack.autofit import prepare_config
from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker

EXAMPLES = Path("examples")
# Every arm off, whatever the shipped defaults are.
OFF = {
    "illum_bias": False,
    "illum_gain": False,
    "illum_weight": False,
    "illum_flat": False,
}

ARMS: dict[str, dict] = {
    "baseline": {},
    "A bias": {"illum_bias": True},
    "B weight": {"illum_weight": True},
    "C gain": {"illum_gain": True},
    "D flat": {"illum_flat": True},
    "A+B": {"illum_bias": True, "illum_weight": True},
    "A+C": {"illum_bias": True, "illum_gain": True},
    "A+D": {"illum_bias": True, "illum_flat": True},
    "A+B+D": {"illum_bias": True, "illum_weight": True, "illum_flat": True},
    "A+B+C": {"illum_bias": True, "illum_weight": True, "illum_gain": True},
}


def row_profile(field: np.ndarray, mask: np.ndarray, bands: int = 12) -> list[float]:
    """Mean of `field` per horizontal band of the window, top to bottom."""
    n = field.shape[0]
    step = max(1, n // bands)
    out = []
    for i in range(0, n, step):
        sel = mask[i : i + step]
        out.append(
            float(field[i : i + step][sel].mean()) if sel.sum() >= 30 else np.nan
        )
    return out


def track(session: Path, overrides: dict, holdout: bool, max_frames: int | None):
    cfg = Config.load(session / "camera_H.txt")
    if not cfg.has_ball():
        # As `spintrack run` does: a config that names no ball leaves it to the detector.
        prepare_config(cfg, cfg.src_fn)
    params = TrackParams(illum_measure=True, **{**OFF, **overrides})
    src = VideoSource(cfg.src_fn)
    tracker = Tracker(cfg, src.width, src.height, params)
    photo = tracker.engine.photometry
    shade_sum = np.zeros((tracker.geometry.size,) * 2)
    shade_n = 0
    costs, split = [], None
    n_total = max_frames or src.n_frames
    for frame in src:
        if max_frames and frame.index >= max_frames:
            break
        if holdout and split is None and frame.index >= n_total // 2:
            split = frame.index
            photo.freeze()
            costs, shade_sum, shade_n = [], np.zeros_like(shade_sum), 0
        res = tracker.process_frame(frame.image, frame.ts_ms)
        if res is not None and np.isfinite(res.step.cost):
            costs.append(res.step.cost)
            shade_sum += tracker.engine.last_obs
            shade_n += 1
    src.close()
    mean, weight = tracker.engine.export_map()
    seen = photo.mask & (photo.acc_n > 1.0)
    field = photo.residual_field()
    shading = np.where(photo.mask, shade_sum / max(shade_n, 1), 0.0)
    return {
        "shading_rms": float(np.sqrt((shading[photo.mask] ** 2).mean())),
        "static_rms": float(np.sqrt((field[seen] ** 2).mean())) if seen.any() else np.nan,
        "cost": float(np.median(costs)) if costs else np.nan,
        "map_std": float(mean[weight > 1.0].std()),
        "coverage": float((weight >= tracker.engine.params.w_min).mean()),
        "profile": row_profile(shading, photo.mask),
        "frames": tracker.frame,
    }  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    sessions = [EXAMPLES / s for s in args.sessions] if args.sessions else sorted(
        p for p in EXAMPLES.iterdir() if (p / "camera_H.txt").exists()
    )  # fmt: skip
    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}
    for session in sessions:
        print(
            f"\n=== {session.name}{' (held out on the second half)' if args.holdout else ''}"
        )
        print(
            "  " + "arm".ljust(10) + " shading_rms  static_rms        cost     map_std"
        )
        try:
            rows = [
                (name, track(session, ov, args.holdout, args.max_frames))
                for name, ov in arms.items()
            ]
        except (ValueError, OSError) as exc:  # e.g. a config without a ball ROI
            print(f"  skipped: {exc}")
            continue
        for name, r in rows:
            print(
                f"  {name.ljust(10)}  {r['shading_rms']:10.4f}  {r['static_rms']:10.4f}"
                f"  {r['cost']:10.5f}  {r['map_std']:10.4f}",
                flush=True,
            )
            print("    shading " + " ".join(f"{v:+5.2f}" for v in r["profile"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
