"""Does the surface map's grid cost accuracy, and would a different tessellation help?

The map is a Lambert cylindrical equal-area grid. Every cell covers the same solid angle,
but not the same shape: cell extents are `(2pi/W) cos(lat)` east-west and `(2/H) / cos(lat)`
north-south, so at the lab default (180x360) an equatorial cell is 0.64 x 1.0 degrees while
the polar row is 8.6 x 0.1 - a sliver. `Map::projection_jacobian` inherits the same
asymmetry: `du/dp` grows as `1/cos^2(lat)`, so whatever noise the map holds near a pole is
amplified on its way into the normal equations. An equi-angular cubemap of the same texel
budget would be 0.87 degrees isotropic everywhere.

Whether any of that is measurable is a separate question, and this script is how to answer
it before rewriting `map.rs`. Two arms:

`resolution` sweeps `map_scale`. If accuracy is flat in it, the grid is not the binding
constraint anywhere and the shape of its cells cannot matter either.

`poles` moves only the grid. Every run tracks the same recording from scratch and builds
its own map; the single difference is `map_frame`, the body frame the map is stored in,
which decides where on the ball the grid's two polar rows fall. No map is resampled or
handed over, so there is nothing else for a difference to come from. If the poles cost
anything, the error varies with where they are pointed.

An earlier version of this arm rotated a finished map instead. It does not work: the
identity rotation reads cell centers exactly and every other one blurs the map by about a
cell, which swamps the effect being measured.

`projection` is the head-to-head that follows from the other two: the same cell count as
an equal-area rectangle or as an equi-angular cube, on the same scenes, at the pole
placement each is stuck with.

    uv run --group bench python -m spintrack_bench.map_grid_sweep resolution
    uv run --group bench python -m spintrack_bench.map_grid_sweep poles --frames 2000
    uv run --group bench python -m spintrack_bench.map_grid_sweep projection

Measured, 600 frames, `q_factor 6` (window 60), median per-frame error in degrees.

`resolution`, and the answer is yes - the grid is the binding constraint, right up to the
point where it stops working:

    map_scale     cells   clean_fly  random_walk
    0.75          45x90      0.0616       0.0678
    1.0          60x120      0.0526       0.0586
    1.5 (default)90x180      0.0435       0.0473
    2.25        135x270      0.0345       0.0363
    3.0         180x360     every frame dropped

At 3.0 there are 64800 cells against ~2800 window pixels, so a single frame leaves almost
every cell below `w_min` and `sample` rejects all four corners everywhere. The thresholds
are calibrated for cells that several window pixels land in.

`poles`, and the answer is also yes:

    pole from +y   clean_fly  random_walk  saccades
    0 (default)       0.0435       0.0473    0.0356
    63-95             0.0387       0.0444    0.0308
    180               0.0433       0.0474    0.0356
    peak-to-peak       18.6%         6.5%     16.5%

The shape is the same on all three scenes and symmetric about 90 degrees, so it is not
noise: the default placement, poles at the top and bottom of the first frame's view, is
the worst one, and 90 degrees away is 6-17% better. The mechanism that fits is the
latitude blur rather than the Jacobian's `1/cos^2` growth. A body point at window +y is
carried through the near point by the pitch rotation a walking fly applies, so the ball's
best-resolved surface repeatedly lands in cells 12 degrees tall; a point at window +/-x
sits on that rotation axis and stays parked at the limb, where the camera resolves little
anyway and the coarse cells cost nothing. Clamping the Jacobian would not touch that,
which is why that arm was dropped rather than measured.

`projection`, over all 21 scenes at `map_scale 1.5` (90x180 = 16200 cells against six
52x52 faces = 16224), and the cube collects most of what the pole sweep says is on the
table without having to know where to point anything:

    median change  -7.2%      18 of 21 scenes improved, none dropped a frame
    best           -24.0%     ball_drop; then holder_shadow -21.8%, occluded -18.3%
    worst          +4.2%      static, whose error is 0.0024 deg either way
    cost         +25-35%      ms/frame, from a heavier projection and per-face blurs

Drift falls on most scenes too. It is not the default: this is new code, it costs a
quarter to a third of the tracking time, and it has not been put against the lab
agreement set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.geometry import matrix_to_rotvec, rotation_between
from spintrack.io.sources import VideoSource
from spintrack.tracker import Tracker
from spintrack_bench.metrics import accumulated_error_deg, frame_errors_deg

DATA = Path("benchmarks/data")
SCALES = (0.75, 1.0, 1.5, 2.25, 3.0)
DEFAULT_SCENES = ("clean_fly", "random_walk", "saccades", "sparse", "low_contrast")
N_POLES = 12


def fibonacci_sphere(n: int) -> np.ndarray:
    """`n` roughly evenly spaced unit vectors, the first one at +y."""
    i = np.arange(n)
    y = 1.0 - 2.0 * i / max(n - 1, 1)
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([r * np.sin(phi), y, r * np.cos(phi)], axis=1)


def track(scene: str, n_frames: int, params: TrackParams):
    """Track `scene`; returns (tracker, per-frame camera-frame increments)."""
    cfg = Config.load(DATA / scene / "config.txt")
    source = VideoSource(DATA / scene / cfg.src_fn)
    tracker = Tracker(cfg, source.width, source.height, params)
    estimates = []
    elapsed = 0.0
    for _ in range(n_frames):
        frame = source.read()
        if frame is None:
            break
        t0 = time.perf_counter()
        result = tracker.process_frame(frame.image, frame.ts_ms)
        elapsed += time.perf_counter() - t0
        estimates.append(result.w_cam if result is not None else [np.nan] * 3)
    source.close()
    tracker.tracking_ms = 1e3 * elapsed / max(len(estimates), 1)
    return tracker, np.asarray(estimates)


def report(scene: str, label: str, tracker, estimates: np.ndarray) -> None:
    truth = np.load(DATA / scene / "truth.npz")["w_cam"][: len(estimates)]
    dropped = int(np.isnan(estimates[:, 0]).sum())
    print(
        f"{label:>12} {np.nanmedian(frame_errors_deg(estimates, truth)):11.4f}"
        f" {accumulated_error_deg(estimates, truth)[-1]:12.3f}"
        f" {100.0 * tracker.engine.map_coverage():9.1f}% {dropped:8d}"
        f" {tracker.tracking_ms:10.2f}"
    )


HEADER = (
    f"{'':>12} {'median err':>11} {'drift (deg)':>12} {'coverage':>10} {'dropped':>8}"
    f" {'ms/frame':>10}"
)


def sweep_resolution(scene: str, n_frames: int) -> None:
    print(f"\n=== {scene}: map_scale ({n_frames} frames) ===")
    print(HEADER)
    for scale in SCALES:
        params = TrackParams(map_scale=scale)
        tracker, estimates = track(scene, n_frames, params)
        h = tracker.engine.map_shape
        report(scene, f"{scale:g} ({h[0]}x{h[1]})", tracker, estimates)


def sweep_poles(scene: str, n_frames: int) -> None:
    print(f"\n=== {scene}: grid pole placement ({n_frames} frames) ===")
    print(HEADER)
    for pole in fibonacci_sphere(N_POLES):
        frame = matrix_to_rotvec(rotation_between(np.array([0.0, 1.0, 0.0]), pole))
        tracker, estimates = track(scene, n_frames, TrackParams(map_frame=tuple(frame)))
        offset = np.degrees(np.arccos(np.clip(pole[1], -1.0, 1.0)))
        report(scene, f"pole {offset:.0f} deg", tracker, estimates)


def sweep_projection(scene: str, n_frames: int) -> None:
    print(f"\n=== {scene}: projection ({n_frames} frames) ===")
    print(HEADER)
    for projection in ("equal_area", "cube"):
        for scale in (1.5,):
            params = TrackParams(map_projection=projection, map_scale=scale)
            tracker, estimates = track(scene, n_frames, params)
            cells = int(np.prod(tracker.engine.map_shape))
            report(scene, f"{projection[:4]} {cells}", tracker, estimates)


ARMS = {
    "resolution": sweep_resolution,
    "poles": sweep_poles,
    "projection": sweep_projection,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=tuple(ARMS))
    parser.add_argument("scenes", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args(argv)
    for scene in args.scenes or DEFAULT_SCENES:
        ARMS[args.arm](scene, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
