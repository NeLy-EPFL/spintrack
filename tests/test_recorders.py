import io
import socket

import numpy as np

from spintrack.io.dat import N_COLUMNS, parse_row, read_dat
from spintrack.io.recorders import (
    STREAM_PREFIX,
    Broadcast,
    FileRecorder,
    TerminalRecorder,
    UdpRecorder,
)

ROW = np.arange(N_COLUMNS, dtype=np.float64)


def test_broadcast_to_file_and_terminal(tmp_path):
    buf = io.StringIO()
    rec = Broadcast([FileRecorder(tmp_path / "a.dat"), TerminalRecorder(buf)])
    rec.write(ROW)
    rec.close()
    assert read_dat(tmp_path / "a.dat").shape == (1, N_COLUMNS)
    assert np.allclose(parse_row(buf.getvalue()), ROW)


def test_udp_recorder_sends_prefixed_line():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2.0)
    rec = UdpRecorder("127.0.0.1", server.getsockname()[1])
    rec.write(ROW)
    rec.close()
    msg = server.recv(4096).decode()
    server.close()
    assert msg.startswith(STREAM_PREFIX) and msg.endswith("\n")
    assert np.allclose(parse_row(msg[len(STREAM_PREFIX) :]), ROW)
