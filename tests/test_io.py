import json

import numpy as np
import polars as pl

from spintrack.io.parquet import ParquetWriter
from spintrack.io.records import (
    BALL_COLUMNS,
    COLUMNS,
    N_COLUMNS,
    TABLE_COLUMNS,
    format_row,
    parse_row,
)


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
    rows[:, 0] = [0, 1, 2, 4, 5, 6, 7]  # frame 3 was dropped
    rows[:, 22] = np.arange(7) + 1
    ball = np.random.default_rng(1).normal(size=(7, len(BALL_COLUMNS)))
    ball[:, 2] = np.arange(7) % 2
    path = tmp_path / "out.parquet"
    with ParquetWriter(path, {"source": "ball.mp4"}) as w:
        for row in rows:
            w.write(row)
        w.write_ball(ball)  # by frame, and short of the last row's frame
        assert not path.exists()
    assert not (tmp_path / "out.parquet.partial").exists()
    table = pl.read_parquet(path)
    assert table.columns == list(TABLE_COLUMNS)
    assert table.schema["seq"] == pl.Int64 and table.schema["err"] == pl.Float64
    assert table.schema["ball_seen"] == pl.Boolean
    meta = pl.read_parquet_metadata(path)
    units = json.loads(meta["units"])
    assert units["timestamp"] == "ms" and units["forward_total"] == "ball radii"
    assert units["ball_x_px"] == "px" and units["window_offset"] == "ball radii"
    assert json.loads(meta["provenance"]) == {"source": "ball.mp4"}
    assert np.array_equal(table.select(COLUMNS).to_numpy(), rows)
    want = ball[[0, 1, 2, 4, 5, 6, 6]]
    want[-1] = [np.nan, np.nan, 0.0, np.nan]
    got = table.select(BALL_COLUMNS).cast(pl.Float64).to_numpy()
    assert np.array_equal(got, want, equal_nan=True)
