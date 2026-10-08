"""FicTrac-compatible 25-column `.dat` records.

Columns (0-based index in parentheses): frame (0); delta rotation vector in camera
coordinates (1-3); error score (4); delta rotation vector in lab coordinates (5-7);
absolute rotation vector, camera (8-10) and lab (11-13); integrated position x, y
(14-15); heading (16); movement direction (17); movement speed (18); integrated forward
and side motion (19-20); timestamp in ms (21); sequence counter (22); delta timestamp
(23); milliseconds since midnight (24). `COLUMNS` names them for Python and Parquet; the
file itself has no header. Fields are separated by `", "`; floats use 14 significant
digits; `frame` and `seq` are integers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import IO, Self

import numpy as np

COLUMNS = (
    "frame",
    "dr_cam_x",
    "dr_cam_y",
    "dr_cam_z",
    "err",
    "dr_lab_x",
    "dr_lab_y",
    "dr_lab_z",
    "r_cam_x",
    "r_cam_y",
    "r_cam_z",
    "r_lab_x",
    "r_lab_y",
    "r_lab_z",
    "pos_x",
    "pos_y",
    "heading",
    "direction",
    "speed",
    "forward_total",
    "side_total",
    "timestamp",
    "seq",
    "delta_ts",
    "wall_ms",
)
N_COLUMNS = len(COLUMNS)
INT_COLUMNS = frozenset({0, 22})
DELIMITER = ", "


def format_row(values: Sequence[float]) -> str:
    """One `.dat` line (no newline) from 25 numbers."""
    if len(values) != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} values, got {len(values)}")
    parts = []
    for i, v in enumerate(values):
        parts.append(str(int(v)) if i in INT_COLUMNS else format(float(v), ".14g"))
    return DELIMITER.join(parts)


def parse_row(line: str) -> np.ndarray:
    """Parse one `.dat` line into a float64 array of 25 values."""
    parts = [p for p in line.strip().split(",") if p.strip()]
    if len(parts) != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} fields, got {len(parts)}: {line!r}")
    return np.array([float(p) for p in parts], dtype=np.float64)


def read_dat(path: str | Path) -> np.ndarray:
    """Load a `.dat` file as an (N, 25) float64 array (empty -> shape (0, 25))."""
    data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.size == 0:
        return np.zeros((0, N_COLUMNS))
    if data.shape[1] != N_COLUMNS:
        raise ValueError(f"{path}: expected {N_COLUMNS} columns, found {data.shape[1]}")
    return data


class DatWriter:
    """Append `.dat` rows to a file, one per call, flushing so tails stay readable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: IO[str] | None = self.path.open("w", buffering=1)

    def write(self, values: Sequence[float]) -> None:
        if self._fh is None:
            raise ValueError("writer is closed")
        self._fh.write(format_row(values) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
