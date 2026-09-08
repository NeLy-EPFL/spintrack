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

What it found (12 seeds of `holder_shadow_lab`, medians, re-measured 2026-09-08 on the
window pre-filter and the plain-EMA bias field; `lost` counts runs that dropped more than
5% of frames, which is the tracker losing the ball rather than degrading):

    arm        median      p95   mapcorr  lost
    baseline   0.0535   0.1143    0.9273  0/12
    A          0.0526   0.1097    0.9327  0/12
    B          0.0541   0.1251    0.9358  0/12
    C          0.0486   0.1020    0.9384  0/12
    D          0.0555   0.1155    0.9356  0/12
    A+B        0.0561   0.1286    0.9285  0/12
    A+C        0.0474   0.0983    0.9406  0/12
    A+D        0.0547   0.1155    0.9384  0/12
    A+B+C      0.0485   0.1033    0.9397  0/12

`static` is deliberately not in that table any more: with the bias field live it is zero
by construction (the field *is* the running mean the residual is measured against), so
it separates nothing. Read it with the field frozen instead - `illumination_real
--holdout` - where `A` takes it from 0.330 to 0.167 on ANXXX049 003 and from 0.050 to
0.048 on AN07B017 003, and every other arm raises it (`B` +18/+106%, `C` +10/+19%, `D`
-9/+16%).

`A` is on by default: it is the only arm that moves the static field on a real
recording, it costs nothing in accuracy, it lowers the photometric residual by 36-40%
and the map's own contrast with it, and it is what keeps a run from losing the ball
(with the pre-filter of `e19a8c3` no arm loses it on any of these twelve seeds; before
that the baseline lost one). `C` is the accuracy arm on synthetic truth - 9% of median
error on its own, 11% with `A`, and 14% of p95 - and would be tempting on this evidence,
but on the lab recordings its field never converges (trial 003, 4000 frames: p95 climbs
1.088 -> 1.220, the maximum reaches 1.52 and 4-5% of pixels sit at the clamp from frame
1500 on), it *raises* the held-out static field, and it raises the photometric residual
by 30% on 003; a per-pixel gain and the map's amplitude are separable only when the ball
turns enough to mix the two, and there it does not. `B` is worse than nothing on truth
and on the held-out field, and its cost win on real video is circular (the reported cost
is a weighted mean using those weights). `D` flattens the shading further than `A` (-10
to -24% of `shading_rms`) but raises the photometric residual by up to 54% on the lab
recordings: at roughly three effectively independent samples per pixel its estimate
contains texture, and subtracting texture is exactly what this is supposed to avoid.

`illum_tau` was chosen the same way and is not a free parameter: 500 frames is the
floor. At 250 a run starts failing again, at 120 five of twelve fail and at 60 all
twelve do, and the `static` reading kept *improving* as they failed - a field fast
enough to absorb what the map cannot is also fast enough to absorb the texture, which is
the same reason the unfrozen `static` is no evidence at all.
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
