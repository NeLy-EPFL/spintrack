"""Record sinks: the `Recorder` protocol, and the live ones.

The live ones are UDP/TCP sockets, a serial port and the terminal;
`spintrack.io.parquet.ParquetWriter` is the file one.

The live sinks send each record as FicTrac's 25-field line in FicTrac's frame
(`spintrack.io.records.to_fictrac`, `format_row`), so that existing FicTrac clients read
the signs they expect. Socket and serial sinks prefix it with `FT, ` and end it with a
newline, which is what those clients parse. They ignore the ball's path that the run
hands every recorder when it ends (`write_ball`).
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from spintrack.io.records import format_row, to_fictrac

STREAM_PREFIX = "FT, "


def _line(values: Sequence[float]) -> str:
    return format_row(to_fictrac(values))


class Recorder(Protocol):
    def write(self, values: Sequence[float]) -> None: ...

    def write_ball(self, columns: np.ndarray) -> None: ...

    def close(self) -> None: ...


class _Stream:
    def write_ball(self, columns: np.ndarray) -> None:
        """Nothing: the stream has sent its records already."""


class TerminalRecorder(_Stream):
    def __init__(self, stream=None):
        self._stream = stream or sys.stdout

    def write(self, values: Sequence[float]) -> None:
        self._stream.write(_line(values) + "\n")

    def close(self) -> None:
        self._stream.flush()


class UdpRecorder(_Stream):
    """Send each record as one UDP datagram to `host:port`."""

    def __init__(self, host: str, port: int):
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def write(self, values: Sequence[float]) -> None:
        self._sock.sendto((STREAM_PREFIX + _line(values) + "\n").encode(), self._addr)

    def close(self) -> None:
        self._sock.close()


class TcpRecorder(_Stream):
    """Stream records over a TCP connection to `host:port` (connects on creation)."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self._sock = socket.create_connection((host, int(port)), timeout=timeout)
        self._sock.settimeout(None)

    def write(self, values: Sequence[float]) -> None:
        self._sock.sendall((STREAM_PREFIX + _line(values) + "\n").encode())

    def close(self) -> None:
        self._sock.close()


class SerialRecorder(_Stream):
    """Write records to a serial port (needs the `serial` extra: pyserial)."""

    def __init__(self, port: str, baud: int = 115200):
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "pip install 'spintrack[serial]' for serial output"
            ) from exc
        self._port = serial.Serial(port, int(baud))

    def write(self, values: Sequence[float]) -> None:
        self._port.write((STREAM_PREFIX + _line(values) + "\n").encode())

    def close(self) -> None:
        self._port.close()
