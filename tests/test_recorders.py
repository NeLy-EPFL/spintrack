import socket

import numpy as np

from spintrack.io.dat import N_COLUMNS, parse_row
from spintrack.io.recorders import STREAM_PREFIX, UdpRecorder

ROW = np.arange(N_COLUMNS, dtype=np.float64)


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
