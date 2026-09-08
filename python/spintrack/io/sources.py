"""Frame sources: anything that yields grayscale frames with timestamps.

A source is any object with `width`, `height`, `fps`, `read()` and `close()`. `read()`
returns a `Frame` or `None` at the end of the stream. Frames are 2-D uint8 arrays; a
color input is converted to gray on the way in, so the tracker never sees color.
"""

from __future__ import annotations

import datetime as _dt
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


def ms_since_midnight() -> float:
    """Local wall-clock time as milliseconds since midnight (FicTrac column 25)."""
    now = _dt.datetime.now(tz=_dt.UTC).astimezone()
    return (
        now - now.replace(hour=0, minute=0, second=0, microsecond=0)
    ).total_seconds() * 1e3


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 view/copy of `image` (BGR or BGRA inputs are converted)."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


@dataclass
class Frame:
    image: np.ndarray  # 2-D uint8
    ts_ms: float  # position in the video (ms) or capture time (ms)
    wall_ms: float  # milliseconds since midnight when the frame was obtained
    index: int


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class VideoSource:
    """Frames from a video file (or any URL OpenCV can open)."""

    def __init__(self, path: str | Path):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise OSError(f"could not open video {path!s}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or -1.0
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        ts = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        frame = Frame(to_gray(image), ts, ms_since_midnight(), self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()

    def __iter__(self):
        while (frame := self.read()) is not None:
            yield frame


class CameraSource:
    """Frames from a camera OpenCV can open (webcams and other UVC devices)."""

    def __init__(self, device: int = 0, fps: float | None = None, size=None):
        self._cap = cv2.VideoCapture(int(device))
        if not self._cap.isOpened():
            raise OSError(f"could not open camera {device}")
        if size is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if fps is not None and fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or -1.0
        self._index = 0

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(
            to_gray(image), time.monotonic() * 1e3, ms_since_midnight(), self._index
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()


class QueueSource:
    """Frames pushed from another thread via `put(image, ts_ms)`; `put(None)` ends it."""

    def __init__(self, width: int, height: int, fps: float = -1.0, maxsize: int = 64):
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._index = 0

    def put(self, image: np.ndarray | None, ts_ms: float | None = None) -> None:
        if image is None:
            self._q.put(None)
            return
        ts = time.monotonic() * 1e3 if ts_ms is None else float(ts_ms)
        self._q.put((image, ts))

    def read(self) -> Frame | None:
        item = self._q.get()
        if item is None:
            return None
        image, ts = item
        frame = Frame(to_gray(np.asarray(image)), ts, ms_since_midnight(), self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        pass


class CallableSource:
    """Frames pulled from `grab()`, which returns `(image, ts_ms)` or `None` when done."""

    def __init__(
        self, grab: Callable[[], tuple | None], width: int, height: int, fps=-1.0
    ):
        self._grab = grab
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self._index = 0

    def read(self) -> Frame | None:
        item = self._grab()
        if item is None:
            return None
        image, ts = item
        frame = Frame(
            to_gray(np.asarray(image)), float(ts), ms_since_midnight(), self._index
        )
        self._index += 1
        return frame

    def close(self) -> None:
        pass


def open_source(spec: str | int | Path) -> FrameSource:
    """A camera index (`0`, `"1"`) opens a camera; anything else is a video path."""
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return CameraSource(int(spec))
    return VideoSource(spec)
