"""Result sinks: `.dat` file, UDP/TCP sockets, serial port, terminal.

Socket and serial sinks send the same 25-field line as the `.dat` file, prefixed with
`FT, ` and terminated by a newline, which is what existing FicTrac clients parse.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from spintrack.io.dat import DatWriter, format_row

STREAM_PREFIX = "FT, "


class Recorder(Protocol):
    def write(self, values: Sequence[float]) -> None: ...

    def close(self) -> None: ...


class FileRecorder:
    def __init__(self, path: str | Path):
        self._writer = DatWriter(path)

    def write(self, values: Sequence[float]) -> None:
        self._writer.write(values)

    def close(self) -> None:
        self._writer.close()


class TerminalRecorder:
    def __init__(self, stream=None):
        self._stream = stream or sys.stdout

    def write(self, values: Sequence[float]) -> None:
        self._stream.write(format_row(values) + "\n")

    def close(self) -> None:
        self._stream.flush()


class UdpRecorder:
    """Send each record as one UDP datagram to `host:port`."""

    def __init__(self, host: str, port: int):
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def write(self, values: Sequence[float]) -> None:
        self._sock.sendto(
            (STREAM_PREFIX + format_row(values) + "\n").encode(), self._addr
        )

    def close(self) -> None:
        self._sock.close()


class TcpRecorder:
    """Stream records over a TCP connection to `host:port` (connects on creation)."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self._sock = socket.create_connection((host, int(port)), timeout=timeout)
        self._sock.settimeout(None)

    def write(self, values: Sequence[float]) -> None:
        self._sock.sendall((STREAM_PREFIX + format_row(values) + "\n").encode())

    def close(self) -> None:
        self._sock.close()


class SerialRecorder:
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
        self._port.write((STREAM_PREFIX + format_row(values) + "\n").encode())

    def close(self) -> None:
        self._port.close()


class Broadcast:
    """Fan one record out to several recorders."""

    def __init__(self, recorders: Sequence[Recorder]):
        self.recorders = list(recorders)

    def write(self, values: Sequence[float]) -> None:
        for r in self.recorders:
            r.write(values)

    def close(self) -> None:
        for r in self.recorders:
            r.close()
