"""Benchmark command line: `bench.py synth|run|report`."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from spintrack_bench.families import FAMILIES, get_family
from spintrack_bench.metrics import frame_errors_deg, summarize
from spintrack_bench.runners.fictrac_cpp import (
    BINARIES,
    run_fictrac,
    window_to_camera_frame,
)
from spintrack_bench.synth.dataset import generate, load_truth

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "benchmarks" / "data"
DEFAULT_RESULTS = ROOT / "benchmarks" / "results"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def _synth_one(name: str, out_root: str, frames: int | None, seed: int | None) -> str:
    spec = get_family(name, frames, seed)
    t0 = time.perf_counter()
    generate(spec, Path(out_root) / name)
    return f"[synth] {name}: {spec.n_frames} frames in {time.perf_counter() - t0:.1f}s"


def cmd_synth(args) -> int:
    from concurrent.futures import ProcessPoolExecutor

    names = list(FAMILIES) if args.families == ["all"] else args.families
    todo = []
    for name in names:
        if (Path(args.out) / name / "truth.npz").exists() and not args.force:
            print(f"[synth] {name}: exists, skipping")
        else:
            todo.append(name)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(_synth_one, n, args.out, args.frames, args.seed) for n in todo
        ]
        for fut in futures:
            print(fut.result(), flush=True)
    return 0


def _dataset_dirs(root: Path, names: list[str]) -> list[Path]:
    if names == ["all"]:
        return sorted(p for p in root.iterdir() if (p / "truth.npz").exists())
    return [root / n for n in names]


def run_system(system: str, dataset: Path, workdir: Path, overrides: dict) -> tuple:
    """Returns (est_cam (N,3) with NaN for missing frames, timing dict)."""
    truth = load_truth(dataset)
    n = len(truth["w_cam"])
    if system in BINARIES:
        res = run_fictrac(BINARIES[system], dataset, workdir, overrides, system=system)
        dat = window_to_camera_frame(res.dat, truth["centre"])
        frames = dat[:, 0].astype(int)
        est = np.full((n, 3), np.nan)
        ok = (frames >= 0) & (frames < n)
        est[frames[ok]] = dat[ok, 1:4]
        timing = {
            "wall_s": res.wall_s,
            "fps_reported": res.fps_reported,
            "evals_per_frame": res.evals_per_frame,
            "tracking_ms_per_frame": res.tracking_ms_per_frame,
        }
        return est, timing
    if system == "spintrack":
        from spintrack_bench.runners.spintrack_runner import run_spintrack

        return run_spintrack(dataset, overrides)
    raise ValueError(f"unknown system {system!r}")


def cmd_run(args) -> int:
    if args.pin_cpu is not None:
        import os

        os.sched_setaffinity(0, {args.pin_cpu})
        os.environ["SPINTRACK_PIN_CPU"] = str(args.pin_cpu)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.overrides) if args.overrides else {}
    overrides.setdefault(
        "src_fps", -1.0
    )  # do not throttle FicTrac to the video frame rate
    cfg_hash = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[
        :8
    ]
    sha = _git_sha()
    rows = []
    for dataset in _dataset_dirs(Path(args.datasets), args.datasets_names):
        truth = load_truth(dataset)
        for system in args.systems:
            label = args.label or system
            workdir = results_dir / "work" / dataset.name / f"{label}-{cfg_hash}"
            try:
                est, timing = run_system(system, dataset, workdir, overrides)
            except (RuntimeError, OSError, ValueError) as exc:  # report and keep going
                print(f"[run] {dataset.name} / {label}: FAILED: {exc}")
                continue
            summary = summarize(
                est, truth["w_cam"], truth["cam_to_lab"], float(truth["fps"])
            )
            row = {
                "dataset": dataset.name,
                "system": label,
                "config_hash": cfg_hash,
                "git_sha": sha,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **timing,
                **summary.as_dict(),
            }
            rows.append(row)
            np.savez_compressed(
                results_dir / f"{dataset.name}__{label}__{cfg_hash}.npz",
                est_cam=est,
                errors_deg=frame_errors_deg(est, truth["w_cam"]),
            )
            print(
                f"[run] {dataset.name:18s} {label:14s} median {summary.median_deg:.3f} deg  "
                f"p95 {summary.p95_deg:.3f}  fail {100 * summary.fail_frac:.1f}%  "
                f"drift {summary.drift_deg_per_min:.2f} deg/min  "
                f"endpoint {summary.endpoint_err_pct:.2f}%  "
                f"trk {timing.get('tracking_ms_per_frame')} ms"
            )
    if rows:
        import pandas as pd

        path = results_dir / "results.parquet"
        df = pd.DataFrame(rows)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
        print(f"[run] appended {len(rows)} rows to {path}")
    return 0


def cmd_agree(args) -> int:
    """Run spintrack on real recordings and compare with the FicTrac `.dat` next to them."""
    import pandas as pd

    from spintrack.io.dat import read_dat
    from spintrack_bench.agreement import compare
    from spintrack_bench.runners.real_video import track_video

    if args.pin_cpu is not None:
        import os

        os.sched_setaffinity(0, {args.pin_cpu})
    overrides = json.loads(args.overrides) if args.overrides else {}
    rows = []
    for trial in sorted(Path(p) for p in args.trials):
        configs = sorted(trial.rglob("config.txt")) if trial.is_dir() else [trial]
        for config in configs:
            dats = sorted(config.parent.glob("*.dat"))
            if not dats:
                print(f"[agree] {config.parent}: no FicTrac .dat found, skipping")
                continue
            ours, timing = track_video(
                config, overrides=overrides, max_frames=args.max_frames
            )
            theirs = read_dat(dats[-1])
            if args.max_frames:
                theirs = theirs[theirs[:, 0] < args.max_frames]
            agr = compare(ours, theirs)
            rows.append({"trial": str(config.parent), **timing, **agr.as_dict()})
            print(
                f"[agree] {config.parent.name:12s} frames {timing['frames']:5d} lost {timing['lost']:3d} "
                f"| median {agr.median_deg:.3f} deg p95 {agr.p95_deg:.3f} | fwd corr {agr.forward_corr:.3f} "
                f"turn corr {agr.turn_corr:.3f} | heading diff {agr.heading_diff_deg:+.1f} deg "
                f"endpoint {agr.endpoint_diff_pct:.1f}% | {timing['tracking_ms_per_frame']:.2f} ms/frame, "
                f"{timing['fps_total']:.0f} fps incl. decode"
            )
    if rows and args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[agree] wrote {args.out}")
    return 0


def cmd_report(args) -> int:
    import pandas as pd

    df = pd.read_parquet(Path(args.results) / "results.parquet")
    # Keep the latest row per (dataset, system).
    df = (
        df.sort_values("timestamp")
        .groupby(["dataset", "system"], as_index=False)
        .last()
    )
    if args.systems != ["all"]:
        df = df[df["system"].isin(args.systems)]
    cols = {
        "median_deg": "median err (deg)",
        "p95_deg": "p95 err (deg)",
        "fail_frac": "fail frac",
        "drift_deg_per_min": "drift (deg/min)",
        "endpoint_err_pct": "endpoint err (%)",
        "tracking_ms_per_frame": "tracking ms/frame",
    }
    lines = [
        "# Benchmark results",
        "",
        "Synthetic scenes rendered by `benchmarks/bench.py synth` (1000 frames at 100 fps, exact",
        "ground-truth rotations; see `benchmarks/spintrack_bench/families.py` for each family).",
        "`fictrac` and `fictrac-fork` are the C++ FicTrac 2.1.2 upstream and NeLy-EPFL builds run as",
        "black boxes on the same videos and configs (FicTrac's window-frame rotation vectors are",
        "mapped to camera coordinates before scoring). `spintrack` is the default configuration",
        "(solve on at most ~4000 window pixels, with the rig's static illumination separated",
        "from the ball's texture); `spintrack-full` solves on every pixel. Errors are",
        "per-frame rotation errors in degrees; drift is the slope of the accumulated",
        "orientation error; endpoint error is the fictive-path endpoint discrepancy as a",
        "percentage of the true path length; tracking time excludes video decoding and was",
        "measured pinned to one core.",
        "",
    ]
    for metric, title in cols.items():
        table = df.pivot(index="dataset", columns="system", values=metric)
        lines += [f"## {title}", "", table.to_markdown(floatfmt=".3f"), ""]
    agreement = Path(args.results) / "agreement_lab_trials.csv"
    if agreement.exists():
        ag = pd.read_csv(agreement)
        ag["trial"] = ag["trial"].map(lambda t: Path(t).name)
        keep = {
            "trial": "trial",
            "frames": "frames",
            "lost": "dropped",
            "median_deg": "median diff (deg)",
            "p95_deg": "p95 diff (deg)",
            "turn_corr": "turn corr",
            "heading_diff_deg": "heading diff (deg)",
            "endpoint_diff_pct": "endpoint diff (%)",
            "tracking_ms_per_frame": "tracking ms/frame",
            "fps_total": "fps incl. decode",
        }
        lines += [
            "## Agreement with FicTrac on real recordings",
            "",
            "Six 60 s trials (1600x1008 HEVC, 100 fps, `q_factor 12`, `accumulate_map: n`) tracked",
            "with spintrack and compared with the lab fork's FicTrac output for the same video.",
            "There is no ground truth here; differences are per-frame angles between the two",
            "trackers' lab-frame rotation increments.",
            "",
            ag[list(keep)]
            .rename(columns=keep)
            .to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[report] wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate synthetic scene families")
    s.add_argument("--out", default=str(DEFAULT_DATA))
    s.add_argument("--families", nargs="+", default=["all"])
    s.add_argument("--frames", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--jobs", type=int, default=8)
    s.set_defaults(func=cmd_synth)
    r = sub.add_parser("run", help="run systems on datasets and score them")
    r.add_argument("--datasets", default=str(DEFAULT_DATA))
    r.add_argument("--datasets-names", nargs="+", default=["all"])
    r.add_argument("--systems", nargs="+", default=["fictrac"])
    r.add_argument("--overrides", default=None, help="JSON dict of config overrides")
    r.add_argument("--label", default=None, help="system label for the results table")
    r.add_argument("--results", default=str(DEFAULT_RESULTS))
    r.add_argument(
        "--pin-cpu",
        type=int,
        default=None,
        help="pin this process (and FicTrac) to one CPU",
    )
    r.set_defaults(func=cmd_run)
    a = sub.add_parser(
        "agree", help="compare spintrack with FicTrac outputs on real videos"
    )
    a.add_argument(
        "--trials", nargs="+", required=True, help="dirs (searched for config.txt)"
    )
    a.add_argument("--overrides", default=None)
    a.add_argument("--max-frames", type=int, default=None)
    a.add_argument("--pin-cpu", type=int, default=None)
    a.add_argument("--out", default=None, help="CSV path for the table")
    a.set_defaults(func=cmd_agree)
    p = sub.add_parser("report", help="render results tables")
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--out", default=None)
    # Ad-hoc `--label` runs stay in the parquet but out of the published tables.
    p.add_argument(
        "--systems",
        nargs="+",
        default=["fictrac", "fictrac-fork", "spintrack", "spintrack-full"],
        help="systems to show as columns ('all' for every label in the results)",
    )
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
