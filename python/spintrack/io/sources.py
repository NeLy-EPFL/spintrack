"""Frame sources: anything that yields grayscale frames with timestamps.

A source is any object with `width`, `height`, `fps`, `read()` and `close()`. `read()`
returns a `Frame` or `None` at the end of the stream. Frames are 2-D uint8 arrays: video
files are decoded straight to luma by PyAV, and camera frames converted to gray on the
way in, so the tracker never sees color.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import av
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
    """Frames from a video file (or any URL FFmpeg can open)."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        try:
            self._container = av.open(self._path)
        except av.error.FFmpegError as exc:
            raise OSError(f"could not open video {path!s}") from exc
        stream = self._container.streams.video[0]
        self.width = stream.codec_context.width
        self.height = stream.codec_context.height
        rate = stream.average_rate
        self.fps = float(rate) if rate else -1.0
        self.n_frames = stream.frames or None  # from the container; None if it says 0
        self._start = stream.start_time or 0
        self._tick = stream.time_base.numerator / stream.time_base.denominator
        # AVI stores no presentation times: FFmpeg makes them up from the decode
        # order, which B-frames scramble, and its seeks by them can land far off. Its
        # frames are evenly spaced by design, so they are timed and found by count.
        self._by_count = self._container.format.name == "avi" and self.fps > 0
        self._keyframes: list[int] | None = None  # packet numbers, for `_seek_by_count`
        self._use(stream, self._container.decode(stream))
        self._index = 0
        self._pending = None  # a frame `seek` decoded and `read` has not returned

    def _use(self, stream, frames) -> None:
        # Frame-threaded decoding: the decoder, not the tracker, bounds a batch run.
        stream.thread_type = "AUTO"
        self._stream, self._frames = stream, frames

    def seek(self, index: int) -> None:
        """Make frame `index` the next `read`: from the keyframe before, decoding on.

        Frames are numbered from their timestamps, which assumes a constant rate.
        """
        if self.fps <= 0:
            raise OSError("cannot seek in a video with no frame rate")
        self._pending = None
        self._index = index
        if self._by_count:
            self._seek_by_count(index)
            return
        target = self._start + round(index / self.fps / self._tick)
        self._container.seek(target, stream=self._stream, backward=True)
        self._frames = self._container.decode(self._stream)
        for frame in self._frames:
            at = round((frame.pts - self._start) * self._tick * self.fps)
            if at > index and self._pending is None:  # landed past it: count instead
                self._seek_by_count(index)
                return
            self._pending = frame
            if at >= index:
                return
        self._pending = None

    def _seek_by_count(self, index: int) -> None:
        """Decode from the last keyframe at or before frame `index`.

        The keyframe is found by counting packets, which is frames: the decode order
        shows each keyframe at its place.
        """
        if self._keyframes is None:
            with av.open(self._path) as container:
                stream = container.streams.video[0]
                packets = (p for p in container.demux(stream) if p.size)
                self._keyframes = [n for n, p in enumerate(packets) if p.is_keyframe]
        start = max((n for n in self._keyframes if n <= index), default=0)
        self._container.close()
        self._container = av.open(self._path)
        stream = self._container.streams.video[0]
        self._use(stream, self._decode_from(stream, start))
        for _ in range(index - start):
            next(self._frames, None)

    def _decode_from(self, stream, first: int):
        """Frames decoded from packet `first` on (the packets before are skipped)."""
        n = 0
        for packet in self._container.demux(stream):
            if packet.size and n < first:
                n += 1
                continue
            yield from stream.codec_context.decode(packet)

    def read(self) -> Frame | None:
        frame, self._pending = self._pending, None
        if frame is None:
            frame = next(self._frames, None)
        if frame is None:
            return None
        if self._by_count:
            ts = self._index * 1e3 / self.fps
        elif frame.pts is None:
            ts = -1.0
        else:  # computed as OpenCV's `CAP_PROP_POS_MSEC` is, to the last bit
            ts = (frame.pts - self._start) * self._tick * 1e3
        image = frame.to_ndarray(format="gray")
        out = Frame(image, ts, ms_since_midnight(), self._index)
        self._index += 1
        return out

    def close(self) -> None:
        self._container.close()

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


def open_source(spec: str | int | Path) -> FrameSource:
    """A camera index (`0`, `"1"`) opens a camera; anything else is a video path."""
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return CameraSource(int(spec))
    return VideoSource(spec)
