"""The tracking records as a Parquet table.

The 25 columns of `spintrack.io.records.COLUMNS`; `frame` and `seq` are int64, the rest
float64 at full precision. The file's key-value metadata holds the spintrack version
(`spintrack`), each column's unit as JSON (`units`) and the run's provenance as JSON
(`provenance`); `polars.read_parquet_metadata` reads them.

The rows are held in memory and written on close, to `<path>.partial` first, which then
moves to `path`, so `path` only ever holds a finished file. A process killed before
closing leaves no records; the CLI handles SIGTERM and SIGHUP as Ctrl-C to close it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from spintrack.io.records import COLUMNS, INT_COLUMNS, N_COLUMNS

# Columns not listed are rotation vectors, in rad. Positions and the integrated
# forward and side motion are in ball radii, which is radians of ball rotation (multiply
# by the ball's radius for a distance); `err` is a mean squared residual in normalized
# intensity units.
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
}


def to_frame(records: np.ndarray) -> pl.DataFrame:
    """An (n, 25) array of records as a DataFrame with named, typed columns."""
    records = np.asarray(records, dtype=np.float64).reshape(-1, N_COLUMNS)
    return pl.DataFrame(
        {
            name: records[:, i].astype(np.int64) if i in INT_COLUMNS else records[:, i]
            for i, name in enumerate(COLUMNS)
        }
    )


def metadata(provenance: dict | None = None) -> dict[str, str]:
    """The file's key-value metadata: the version, the units and `provenance`."""
    from spintrack import __version__

    units = {name: UNITS.get(name, "rad") for name in COLUMNS}
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
        to_frame(np.array(self._rows)).write_parquet(partial, metadata=self._metadata)
        self._rows = None
        os.replace(partial, self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
