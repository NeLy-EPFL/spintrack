"""The `.dat` records as a Parquet table.

Same 25 columns as `spintrack.io.dat`, under the names in `COLUMNS`; `frame` and `seq`
are int64, the rest float64 at full precision. Each field carries its unit in its
metadata (`unit`); the file metadata carries the spintrack version (`spintrack`) and
the run's provenance as JSON (`provenance`).

Rows are written in row groups to `<path>.partial`, which moves to `path` on close, so
`path` only ever holds a finished file. A process killed before closing leaves the
`.partial` file behind, without the footer a reader needs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spintrack.io.dat import COLUMNS, INT_COLUMNS, N_COLUMNS

# Columns not listed are rotation vectors, in rad. Positions are in radians of ball
# rotation (multiply by the ball radius for distance); `err` is a mean squared residual
# in normalized intensity units.
UNITS = {
    "frame": "",
    "err": "",
    "pos_x": "rad",
    "pos_y": "rad",
    "heading": "rad",
    "step_dir": "rad",
    "step_mag": "rad/frame",
    "int_x": "rad",
    "int_y": "rad",
    "ts": "ms",
    "seq": "",
    "delta_ts": "ms",
    "ms": "ms since midnight",
}


def schema(provenance: dict | None = None) -> pa.Schema:
    """The table schema, with the version and `provenance` in its metadata."""
    from spintrack import __version__

    fields = [
        pa.field(
            name,
            pa.int64() if i in INT_COLUMNS else pa.float64(),
            metadata={"unit": UNITS.get(name, "rad")},
        )
        for i, name in enumerate(COLUMNS)
    ]
    metadata = {"spintrack": __version__}
    if provenance is not None:
        metadata["provenance"] = json.dumps(provenance, default=float)
    return pa.schema(fields, metadata=metadata)


class ParquetWriter:
    """Collect `.dat` rows into a Parquet file, one per call (see the module doc)."""

    def __init__(
        self,
        path: str | Path,
        provenance: dict | None = None,
        row_group_size: int = 65536,
    ):
        self.path = Path(path)
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._schema = schema(provenance)
        self._writer: pq.ParquetWriter | None = pq.ParquetWriter(
            self._partial, self._schema
        )
        # Column-major, so each column of a row group is contiguous.
        self._cols = np.empty((N_COLUMNS, row_group_size))
        self._n = 0

    def write(self, values: Sequence[float]) -> None:
        if self._writer is None:
            raise ValueError("writer is closed")
        if len(values) != N_COLUMNS:
            raise ValueError(f"expected {N_COLUMNS} values, got {len(values)}")
        self._cols[:, self._n] = values
        self._n += 1
        if self._n == self._cols.shape[1]:
            self._flush()

    def _flush(self) -> None:
        if self._n == 0 or self._writer is None:
            return
        arrays = [
            pa.array(col.astype(np.int64) if i in INT_COLUMNS else col)
            for i, col in enumerate(self._cols[:, : self._n])
        ]
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self._schema))
        self._n = 0

    def close(self) -> None:
        if self._writer is None:
            return
        self._flush()
        self._writer.close()
        self._writer = None
        os.replace(self._partial, self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
