"""FicTrac's `config.txt`, which only the benchmark still needs.

C++ FicTrac reads its own format, so every scene gets one, and the lab's recordings come
with one. `to_spintrack` translates one into a spintrack `Config`, by the table in
`docs/fictrac.md`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from spintrack.calibrate.sliders import rotation_from_fictrac
from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.sphere import ball_outline


def read_fictrac_config(path: str | Path) -> dict:
    """`key : value` lines as a dict: `{ ... }` vectors as lists, `y`/`n` as bools."""
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "#%" or ":" not in line:
            continue
        key, _, text = line.partition(":")
        text = re.split(r"\s#", text, maxsplit=1)[0].strip()
        if text in ("y", "n"):
            values[key.strip()] = text == "y"
            continue
        try:
            values[key.strip()] = ast.literal_eval(
                text.replace("{", "[").replace("}", "]")
            )
        except ValueError, SyntaxError:  # a name or a path
            values[key.strip()] = text
    return values


def write_fictrac_config(path: str | Path, values: dict) -> Path:
    """Write `values` as FicTrac's `key : value` lines."""
    path = Path(path)
    lines = [f"{key:<16} : {_fictrac_value(value)}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def _fictrac_value(value) -> str:
    if isinstance(value, bool):
        return "y" if value else "n"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "{ " + ", ".join(_fictrac_value(v) for v in value) + " }"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def to_spintrack(path: str | Path, size: tuple[int, int]) -> Config:
    """The spintrack config of the FicTrac config at `path`, for frames of `size`.

    FicTrac tracks the ball its calibration wrote as `roi_c`/`roi_r` when there is one,
    so that becomes rim points on its outline; otherwise the clicked `roi_circ` are the
    rim. The keys spintrack has no use for are dropped.
    """
    path = Path(path)
    v = read_fictrac_config(path)
    vfov, fisheye = float(v["vfov"]), bool(v.get("fisheye", False))
    if "roi_c" in v and "roi_r" in v:
        camera = source_camera(*size, vfov, fisheye)
        rim = ball_outline(camera, v["roi_c"], float(v["roi_r"]), 16).tolist()
    else:
        rim = _pairs(v["roi_circ"])
    ignore = v.get("roi_ignr", [])
    if ignore and not isinstance(ignore[0], list):
        ignore = [ignore]  # a single polygon written without the outer braces
    q_factor = v.get("q_factor", 6)
    tracking = {
        "window_px": 10 * (q_factor if q_factor > 0 else 6),
        "global_search": bool(v.get("opt_do_global", False)),
        "forget_outside_view": not v.get("accumulate_map", True),
    }
    if v.get("max_bad_frames", -1) >= 0:
        tracking["max_bad_frames"] = v["max_bad_frames"]
    if v.get("opt_bound", -1) > 0:
        tracking["max_step_rad"] = v["opt_bound"]
    if v.get("thr_win_pc", -1) > 0:
        tracking["norm_window"] = v["thr_win_pc"]
    if v.get("sphere_map_fn"):
        tracking["initial_map"] = str(path.parent / v["sphere_map_fn"])
    rotation = rotation_from_fictrac(v["c2a_r"]) if v.get("c2a_r") else None
    return Config(
        video=str(path.parent / v["src_fn"]) if v.get("src_fn") else None,
        camera={"vfov_deg": vfov, "fisheye": fisheye, "rotation": rotation},
        ball={"rim": rim},
        mask={"ignore": [_pairs(polygon) for polygon in ignore]},
        tracking=tracking,
    )


def _pairs(flat) -> list[tuple]:
    return list(zip(flat[::2], flat[1::2], strict=True))
