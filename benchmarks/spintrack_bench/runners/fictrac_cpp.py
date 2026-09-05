"""Run a C++ FicTrac binary on a dataset directory and collect its `.dat` output."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spintrack.config import Config
from spintrack.geometry import rotation_between
from spintrack.io.dat import read_dat

FICTRAC_UPSTREAM = Path("~/fictrac-upstream/bin/fictrac").expanduser()
FICTRAC_FORK = Path("~/.local/opt/fictrac-fork/src-copy/bin/fictrac").expanduser()

BINARIES = {"fictrac": FICTRAC_UPSTREAM, "fictrac-fork": FICTRAC_FORK}


@dataclass
class RunResult:
    system: str
    dat: np.ndarray  # (M, 25) rows actually logged by the tracker
    wall_s: float
    fps_reported: float | None = None
    evals_per_frame: float | None = None
    tracking_ms_per_frame: float | None = None  # opt + map time from the log
    log_path: Path | None = None
    extra: dict = field(default_factory=dict)


def run_fictrac(
    binary: Path,
    dataset_dir: Path,
    workdir: Path,
    overrides: dict | None = None,
    system: str = "fictrac",
    timeout: float = 3600.0,
) -> RunResult:
    """Run `binary` on `dataset_dir/config.txt` inside `workdir`; parse its outputs."""
    dataset_dir = Path(dataset_dir)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(dataset_dir / "config.txt")
    cfg.src_fn = str((dataset_dir / cfg.src_fn).resolve())
    cfg.do_display = False
    cfg.save_debug = False
    cfg.save_raw = False
    cfg.output_fn = str((workdir / "run").resolve())
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value
    cfg.save(workdir / "config.txt")

    before = set(workdir.glob("*.dat"))
    t0 = time.perf_counter()
    import os

    cmd = [str(binary), "config.txt", "-v", "INF"]
    if os.environ.get("SPINTRACK_PIN_CPU"):
        cmd = ["taskset", "-c", os.environ["SPINTRACK_PIN_CPU"], *cmd]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    wall = time.perf_counter() - t0
    new_dats = sorted(
        set(workdir.glob("*.dat")) - before, key=lambda p: p.stat().st_mtime
    )
    if proc.returncode != 0 or not new_dats:
        raise RuntimeError(
            f"{binary.name} failed (exit {proc.returncode}); stderr tail:\n"
            + proc.stderr[-2000:]
            + proc.stdout[-2000:]
        )
    dat = read_dat(new_dats[-1])
    logs = sorted(workdir.glob("fictrac-*.log"), key=lambda p: p.stat().st_mtime)
    result = RunResult(
        system=system, dat=dat, wall_s=wall, log_path=logs[-1] if logs else None
    )
    if logs:
        _parse_log(logs[-1].read_text(errors="replace"), result)
    return result


def _parse_log(text: str, result: RunResult) -> None:
    m = re.search(r"Average fps:\s*([0-9.]+)", text)
    if m:
        result.fps_reported = float(m.group(1))
    m = re.search(r"Average number evals / frame:\s*([0-9.]+)", text)
    if m:
        result.evals_per_frame = float(m.group(1))
    m = re.search(
        r"Average grab/opt/map/plot/log/disp time:\s*([0-9.]+) / ([0-9.]+) / ([0-9.]+)",
        text,
    )
    if m:
        result.tracking_ms_per_frame = float(m.group(2)) + float(m.group(3))


def window_to_camera_frame(dat: np.ndarray, centre) -> np.ndarray:
    """Re-express FicTrac's "camera" rotation vectors in the true camera frame.

    FicTrac reports its delta and absolute rotation vectors (columns 1-3 and 8-10) in its
    tracking-window frame, whose z axis points at the ball centre, but labels them as camera
    coordinates. For an on-axis ball the two frames coincide; for an off-axis ball they differ
    by the rotation taking +z onto the ball centre direction, which this undoes.
    """
    out = np.array(dat, dtype=np.float64, copy=True)
    R = rotation_between(
        np.array([0.0, 0.0, 1.0]), np.asarray(centre, dtype=np.float64)
    )
    for sl in (slice(1, 4), slice(8, 11)):
        out[:, sl] = dat[:, sl] @ R.T
    return out
