"""Offline/online driver: pull frames from a source, track, fan results out to recorders.

Decoding runs in a background thread so that video decode and tracking overlap.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.recorders import Recorder
from spintrack.io.sources import Frame, FrameSource
from spintrack.tracker import FrameResult, Tracker


@dataclass
class RunStats:
    frames: int = 0
    tracked: int = 0
    dropped: int = 0
    wall_s: float = 0.0
    tracking_s: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def tracking_ms_per_frame(self) -> float:
        return 1e3 * self.tracking_s / self.frames if self.frames else 0.0


def _prefetch(source: FrameSource, q: queue.Queue, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            frame = source.read()
            q.put(frame)
            if frame is None:
                return
    except Exception as exc:  # noqa: BLE001 - forward any decoder failure to the consumer
        q.put(exc)


def run(
    cfg: Config,
    source: FrameSource,
    recorders: Sequence[Recorder] = (),
    params: TrackParams | None = None,
    on_result: Callable[[FrameResult], None] | None = None,
    max_frames: int | None = None,
    prefetch: bool = True,
    progress: Callable[[RunStats], None] | None = None,
) -> RunStats:
    """Track every frame of `source`; returns run statistics."""
    tracker = Tracker(cfg, source.width, source.height, params)
    stats = RunStats()
    t0 = time.perf_counter()
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=32)
    if prefetch:
        threading.Thread(target=_prefetch, args=(source, q, stop), daemon=True).start()

    def next_frame() -> Frame | None:
        if not prefetch:
            return source.read()
        item = q.get()
        if isinstance(item, Exception):
            raise item
        return item

    try:
        while True:
            frame = next_frame()
            if frame is None:
                break
            t1 = time.perf_counter()
            result = tracker.process_frame(frame.image, frame.ts_ms, frame.wall_ms)
            stats.tracking_s += time.perf_counter() - t1
            stats.frames += 1
            if result is None:
                stats.dropped += 1
            else:
                stats.tracked += 1
                for rec in recorders:
                    rec.write(result.values)
                if on_result is not None:
                    on_result(result)
            if progress is not None and stats.frames % 500 == 0:
                stats.wall_s = time.perf_counter() - t0
                progress(stats)
            if max_frames is not None and stats.frames >= max_frames:
                break
    finally:
        stop.set()
        stats.wall_s = time.perf_counter() - t0
    return stats
