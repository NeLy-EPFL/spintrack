"""Run a C++ FicTrac binary on a dataset directory and collect its `.dat` output."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spintrack.geometry import rotation_between
from spintrack.io.records import read_dat
from spintrack_bench.fictrac_config import read_fictrac_config, write_fictrac_config

# The C++ builds are local; point the environment at others.
BINARIES = {
    "fictrac": Path(
        os.environ.get("FICTRAC_BIN", "~/fictrac-upstream/bin/fictrac")
    ).expanduser(),
    "fictrac-fork": Path(
        os.environ.get(
            "FICTRAC_FORK_BIN", "~/.local/opt/fictrac-fork/src-copy/bin/fictrac"
        )
    ).expanduser(),
}


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
    cfg = read_fictrac_config(dataset_dir / "config.txt")
    cfg.update(
        src_fn=str((dataset_dir / cfg["src_fn"]).resolve()),
        do_display=False,
        save_debug=False,
        save_raw=False,
        output_fn=str((workdir / "run").resolve()),
    )
    cfg.update(overrides or {})
    write_fictrac_config(workdir / "config.txt", cfg)

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


def window_to_camera_frame(dat: np.ndarray, center) -> np.ndarray:
    """Re-express FicTrac's "camera" rotation vectors in the true camera frame.

    FicTrac reports its delta and absolute rotation vectors (columns 1-3 and 8-10) in
    its tracking-window frame, whose z axis points at the ball center, but labels them
    as camera coordinates. For an on-axis ball the two frames coincide; for an off-axis
    ball they differ by the rotation taking +z onto the ball center direction, which
    this undoes.
    """
    out = np.array(dat, dtype=np.float64, copy=True)
    R = rotation_between(
        np.array([0.0, 0.0, 1.0]), np.asarray(center, dtype=np.float64)
    )
    for sl in (slice(1, 4), slice(8, 11)):
        out[:, sl] = dat[:, sl] @ R.T
    return out


def fictrac_lab_frame(dat: np.ndarray, center, cam_to_lab) -> np.ndarray:
    """Put FicTrac's lab-frame increments (columns 5-7) in the true lab frame.

    FicTrac applies `c2a_r` to its window-frame vectors (see `window_to_camera_frame`),
    so its lab columns are rotated by the ball's off-axis angle: `C R C^T` undoes it.
    On a rig whose sideslip and turning are correlated this is worth percents of
    turning.
    """
    out = np.array(dat, dtype=np.float64, copy=True)
    R = rotation_between(
        np.array([0.0, 0.0, 1.0]), np.asarray(center, dtype=np.float64)
    )
    C = np.asarray(cam_to_lab, dtype=np.float64)
    out[:, 5:8] = dat[:, 5:8] @ (C @ R @ C.T).T
    return out
