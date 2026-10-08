import json

import numpy as np
import polars as pl

from spintrack.io.parquet import ParquetWriter
from spintrack.io.records import COLUMNS, N_COLUMNS, format_row, parse_row


def test_line_round_trip():
    row = np.arange(N_COLUMNS, dtype=np.float64) * 0.123456789012345
    row[0] = 7
    row[22] = 3
    line = format_row(row)
    fields = line.split(", ")
    assert len(fields) == N_COLUMNS and fields[0] == "7" and fields[22] == "3"
    assert np.allclose(parse_row(line), row, rtol=1e-13)


def test_parquet_writer_types_and_metadata(tmp_path):
    rows = np.random.default_rng(0).normal(size=(7, N_COLUMNS))
    rows[:, 0] = np.arange(7)
    rows[:, 22] = np.arange(7) + 1
    path = tmp_path / "out.parquet"
    with ParquetWriter(path, {"source": "ball.mp4"}) as w:
        for row in rows:
            w.write(row)
        assert not path.exists()
    assert not (tmp_path / "out.parquet.partial").exists()
    table = pl.read_parquet(path)
    assert table.columns == list(COLUMNS)
    assert table.schema["seq"] == pl.Int64 and table.schema["err"] == pl.Float64
    meta = pl.read_parquet_metadata(path)
    units = json.loads(meta["units"])
    assert units["timestamp"] == "ms" and units["forward_total"] == "ball radii"
    assert json.loads(meta["provenance"]) == {"source": "ball.mp4"}
    assert np.array_equal(table.to_numpy(), rows)
