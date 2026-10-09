"""The tracking records as a Parquet table.

The 25 columns of `spintrack.io.records.COLUMNS`, then the ball's position and the
window's offset from it (`BALL_COLUMNS`, `docs/output.md`); `frame` and `seq` are
int64, `ball_seen` boolean, the rest float64 at full precision. The file's key-value
metadata holds the spintrack version (`spintrack`), each column's unit as JSON
(`units`) and the run's provenance as JSON (`provenance`);
`polars.read_parquet_metadata` reads them.

The rows are held in memory and written on close, to `<path>.partial` first, which then
moves to `path`, so `path` only ever holds a finished file. The ball columns come once
the run ends (`write_ball`), since the ball's path is smoothed over the whole run; a
file closed without them has them NaN. A process killed before closing leaves no
records; the CLI handles SIGTERM and SIGHUP as Ctrl-C to close it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from spintrack.io.records import INT_COLUMNS, N_COLUMNS, TABLE_COLUMNS, with_ball

# Columns not listed are rotation vectors, in rad. Positions and the integrated
# forward and side motion are in ball radii, which is radians of ball rotation (multiply
# by the ball's radius for a distance); `err` is a mean squared residual in normalized
# intensity units. The ball's position is in source image pixels.
UNITS = {
    "frame": "",
    "err": "",
    "pos_x": "ball radii",
    "pos_y": "ball radii",
    "heading": "rad",
    "direction": "rad",
    "speed": "rad/frame",
    "forward_total": "ball radii",
    "side_total": "ball radii",
    "timestamp": "ms",
    "seq": "",
    "delta_ts": "ms",
    "wall_ms": "ms since midnight",
    "ball_x_px": "px",
    "ball_y_px": "px",
    "ball_seen": "",
    "window_offset": "ball radii",
}
BOOL_COLUMNS = frozenset({TABLE_COLUMNS.index("ball_seen")})


def to_frame(table: np.ndarray) -> pl.DataFrame:
    """An (n, 29) array of `TABLE_COLUMNS` as a DataFrame with named, typed columns."""
    table = np.asarray(table, dtype=np.float64).reshape(-1, len(TABLE_COLUMNS))
    columns = {}
    for i, name in enumerate(TABLE_COLUMNS):
        if i in INT_COLUMNS:
            columns[name] = table[:, i].astype(np.int64)
        elif i in BOOL_COLUMNS:
            columns[name] = table[:, i] > 0.5
        else:
            columns[name] = table[:, i]
    return pl.DataFrame(columns)


def metadata(provenance: dict | None = None) -> dict[str, str]:
    """The file's key-value metadata: the version, the units and `provenance`."""
    from spintrack import __version__

    units = {name: UNITS.get(name, "rad") for name in TABLE_COLUMNS}
    meta = {"spintrack": __version__, "units": json.dumps(units)}
    if provenance is not None:
        meta["provenance"] = json.dumps(provenance, default=float)
    return meta


class ParquetWriter:
    """Collect records into a Parquet file, one per call (see the module doc)."""

    def __init__(self, path: str | Path, provenance: dict | None = None):
        self.path = Path(path)
        self._metadata = metadata(provenance)
        self._rows: list[np.ndarray] | None = []
        self._ball: np.ndarray | None = None

    def write_ball(self, columns: np.ndarray) -> None:
        """Take `BALL_COLUMNS` for every frame, an (n, 4) array indexed by frame."""
        self._ball = np.asarray(columns, dtype=np.float64)

    def write(self, values: Sequence[float]) -> None:
        if self._rows is None:
            raise ValueError("writer is closed")
        if len(values) != N_COLUMNS:
            raise ValueError(f"expected {N_COLUMNS} values, got {len(values)}")
        self._rows.append(np.array(values, dtype=np.float64))

    def close(self) -> None:
        if self._rows is None:
            return
        partial = self.path.with_name(self.path.name + ".partial")
        table = with_ball(np.array(self._rows), self._ball)
        to_frame(table).write_parquet(partial, metadata=self._metadata)
        self._rows = None
        os.replace(partial, self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
