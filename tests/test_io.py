import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spintrack.io.dat import (
    COLUMNS,
    N_COLUMNS,
    DatWriter,
    format_row,
    parse_row,
    read_dat,
)
from spintrack.io.parquet import ParquetWriter


def test_dat_round_trip(tmp_path):
    row = np.arange(N_COLUMNS, dtype=np.float64) * 0.123456789012345
    row[0] = 7
    row[22] = 3
    line = format_row(row)
    fields = line.split(", ")
    assert len(fields) == N_COLUMNS and fields[0] == "7" and fields[22] == "3"
    assert np.allclose(parse_row(line), row, rtol=1e-13)
    path = tmp_path / "out.dat"
    with DatWriter(path) as w:
        w.write(row)
        w.write(row * 2)
    data = read_dat(path)
    assert data.shape == (2, N_COLUMNS) and np.allclose(data[1], row * 2, rtol=1e-13)


def test_parquet_writer_row_groups_and_metadata(tmp_path):
    rows = np.random.default_rng(0).normal(size=(7, N_COLUMNS))
    rows[:, 0] = np.arange(7)
    rows[:, 22] = np.arange(7) + 1
    path = tmp_path / "out.parquet"
    # Row groups of 3: two full ones, and the last row flushed on close.
    with ParquetWriter(path, {"source": "ball.mp4"}, row_group_size=3) as w:
        for row in rows:
            w.write(row)
        assert not path.exists()
    assert not (tmp_path / "out.parquet.partial").exists()
    assert pq.ParquetFile(path).num_row_groups == 3
    table = pq.read_table(path)
    assert table.column_names == list(COLUMNS)
    assert table.schema.field("seq").type == pa.int64()
    assert table.schema.field("timestamp").metadata[b"unit"] == b"ms"
    assert table.schema.field("forward_total").metadata[b"unit"] == b"ball radii"
    assert json.loads(table.schema.metadata[b"provenance"]) == {"source": "ball.mp4"}
    assert np.array_equal(np.column_stack([c.to_numpy() for c in table.columns]), rows)
