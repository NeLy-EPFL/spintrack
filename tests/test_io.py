import numpy as np

from spintrack.io.dat import N_COLUMNS, DatWriter, format_row, parse_row, read_dat


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
