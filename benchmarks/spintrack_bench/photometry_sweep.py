"""Which illumination correction actually helps, and do they combine?

The ball is lit by fixed lamps, so the shading at a window pixel is fixed in the camera
frame while the texture rotates past it. `spintrack.photometry` offers four ways of using
that, and there is no reason to believe a priori which of them - or which combination -
is worth its cost, so this runs the matrix and prints one table per scene.

Reported per arm: tracking error against ground truth, `static_bias_rms` (how much
camera-fixed structure is left that the ball-fixed map cannot explain - lower is better,
and it needs no ground truth so the same number can be read off a real recording), and
`map_corr` (correlation between the recovered surface map and the true albedo - this is
the "did the darkness end up in the map" number).

    uv run --group bench python -m spintrack_bench.photometry_sweep
    uv run --group bench python -m spintrack_bench.photometry_sweep holder_shadow_cut
    uv run --group bench python -m spintrack_bench.photometry_sweep --data <root>

What it found (12 seeds of `holder_shadow_lab`, medians; `lost` counts runs that dropped
more than 5% of frames, which is the tracker losing the ball rather than degrading):

    arm        median   static  mapcorr  lost
    baseline   0.0562   0.1459   0.9256   1/12
    A          0.0557   0.0650   0.9293   0/12
    A+B        0.0572   0.0722   0.9276   0/12
    A+C        0.0494   0.0586   0.9354   0/12
    A+B+C      0.0473   0.0608   0.9385   0/12

Only `A` moves `static`, and it more than halves it: separating the illumination is what
the bias field does, and no other arm substitutes for it. `A` also costs nothing in
accuracy and removes the one run in twelve where the baseline lost the ball, so it is on
by default. `C` is the accuracy arm, and would be tempting on this evidence - but on the
lab recordings its field never converges (over 4000 frames p95 climbs 1.13 -> 1.35 and the
maximum reaches 2.4, with 3.6% of pixels pinned at the clamp), because a per-pixel gain and
the map's amplitude are separable only when the ball turns enough to mix the two, and there
it does not. `B` is worse than nothing on its own. `D` flattens the shading further than
`A` but raises the photometric residual by up to 60% on the lab recordings: at roughly
three effectively independent samples per pixel its estimate contains texture, and
subtracting texture is exactly what this is supposed to avoid, and `illumination_real`
is where those recordings are measured.

`illum_tau` was chosen the same way and is not a free parameter: 500 frames is the floor.
At 250 a run starts failing again, at 120 five of twelve fail and at 60 all twelve do, and
the `static` reading keeps *improving* as they fail - a field fast enough to absorb what
the map cannot is also fast enough to absorb the texture, which is why `static` alone is
never sufficient evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spintrack_bench.metrics import summarize
from spintrack_bench.runners.spintrack_runner import run_spintrack
from spintrack_bench.synth.dataset import load_truth

DATA = Path("benchmarks/data")

# Each arm switches one field on; the combinations are what the sweep is for.
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

DEFAULT_SCENES = (
    "holder_shadow_cut",
    "holder_shadow_lab",
    "holder_shadow",
    "lighting",
    "lab_small_ball",
    "occluded",
    "sparse",
)


def run_arm(dataset: Path, overrides: dict) -> dict:
    truth = load_truth(dataset)
    # `illum_measure` keeps the diagnostic comparable across arms, including the baseline.
    est, timing = run_spintrack(dataset, {"illum_measure": True, **OFF, **overrides})
    s = summarize(est, truth["w_cam"], truth["cam_to_lab"], float(truth["fps"]))
    return {
        "median_deg": s.median_deg,
        "p95_deg": s.p95_deg,
        "drift_deg_per_min": s.drift_deg_per_min,
        "endpoint_err_pct": s.endpoint_err_pct,
        "fail_pct": 100.0 * s.fail_frac,
        "static_bias_rms": timing.get("static_bias_rms", float("nan")),
        "map_corr": timing.get("map_corr", float("nan")),
        "ms_per_frame": timing["tracking_ms_per_frame"],
    }


COLUMNS = [
    ("median_deg", "median", "{:8.4f}"),
    ("p95_deg", "p95", "{:8.4f}"),
    ("drift_deg_per_min", "drift/min", "{:10.3f}"),
    ("endpoint_err_pct", "endpoint%", "{:10.2f}"),
    ("fail_pct", "fail%", "{:7.1f}"),
    ("static_bias_rms", "static", "{:8.4f}"),
    ("map_corr", "mapcorr", "{:8.4f}"),
    ("ms_per_frame", "ms/frm", "{:7.2f}"),
]


def sweep(scene: str, arms: dict[str, dict], data: Path = DATA) -> None:
    dataset = data / scene
    if not (dataset / "video.mp4").exists():
        print(f"[skip] {scene}: not generated")
        return
    print(f"\n=== {scene}")
    print("  " + "arm".ljust(10) + "".join(h.rjust(10) for _, h, _ in COLUMNS))
    rows = {}
    for name, overrides in arms.items():
        row = run_arm(dataset, overrides)
        rows[name] = row
        cells = "".join(fmt.format(row[key]).rjust(10) for key, _, fmt in COLUMNS)
        print(f"  {name.ljust(10)}{cells}", flush=True)
    base = rows.get("baseline")
    if base:
        best = min(rows, key=lambda k: rows[k]["median_deg"])
        print(
            f"  best median: {best} "
            f"({100 * (rows[best]['median_deg'] / base['median_deg'] - 1):+.1f}% vs baseline)"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    parser.add_argument("--data", type=Path, default=DATA, help="dataset root")
    args = parser.parse_args(argv)
    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}
    for scene in args.scenes or DEFAULT_SCENES:
        sweep(scene, arms, args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
