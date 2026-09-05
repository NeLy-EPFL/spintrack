"""Offline/online driver: pull frames from a source, track, fan results out to recorders.

Decoding runs in a background thread so that video decode and tracking overlap.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.io.recorders import Recorder
from spintrack.io.sources import Frame, FrameSource
from spintrack.tracker import FrameResult, Tracker

log = logging.getLogger("spintrack")


@dataclass
class RunStats:
    frames: int = 0
    tracked: int = 0
    dropped: int = 0
    wall_s: float = 0.0
    tracking_s: float = 0.0
    refine: dict = field(default_factory=dict)

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
    save_map: str | None = None,
    debug_video: str | None = None,
    refine_sweeps: int = 0,
    refined_out: str | None = None,
) -> RunStats:
    """Track every frame of `source`; returns run statistics.

    `debug_video` writes an annotated video; `refine_sweeps > 0` keeps every normalized
    window in memory, re-estimates all orientations against the complete map afterwards and
    writes the refined records to `refined_out`.
    """
    tracker = Tracker(cfg, source.width, source.height, params)
    stats = RunStats()
    canvas = writer = None
    if debug_video:
        from spintrack.debug_video import DebugCanvas, DebugVideoWriter

        canvas = DebugCanvas(tracker)
        writer = DebugVideoWriter(debug_video, canvas.size, source.fps, cfg.vid_codec)
    keep: list[tuple] = []  # (frame index, ts, wall, normalized window, orientation)
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
            if writer is not None:
                fps = stats.frames / max(time.perf_counter() - t0, 1e-9)
                writer.write(canvas.render(frame.image, result, fps))
            if refine_sweeps > 0 and tracker.engine.last_obs is not None:
                R_win = result.step.R_win if result is not None else None
                obs16 = tracker.engine.last_obs.astype(np.float16)
                keep.append((frame.index, frame.ts_ms, frame.wall_ms, obs16, R_win))
            if progress is not None and stats.frames % 500 == 0:
                stats.wall_s = time.perf_counter() - t0
                progress(stats)
            if max_frames is not None and stats.frames >= max_frames:
                break
    finally:
        stop.set()
        stats.wall_s = time.perf_counter() - t0
        if writer is not None:
            writer.close()
    if save_map:
        tracker.save_map(save_map)
    if refine_sweeps > 0 and keep:
        from spintrack.io.dat import DatWriter
        from spintrack.refine import refine_orientations

        n = tracker.geometry.size
        mb = len(keep) * n * n * 2 / 1e6
        log.info(
            "refining %d frames (%.0f MB of windows), %d sweeps",
            len(keep),
            mb,
            refine_sweeps,
        )
        frames = [k[0] for k in keep]
        ts_list = [k[1] for k in keep]
        wall_list = [k[2] for k in keep]
        windows = [k[3] for k in keep]
        online = [k[4] for k in keep]
        refined, rstats = refine_orientations(
            tracker.engine, windows, online, refine_sweeps
        )
        rows = tracker.records_from_orientations(refined, ts_list, wall_list, frames)
        stats.refine = {**rstats, "frames": len(rows)}
        if refined_out:
            with DatWriter(refined_out) as w:
                for row in rows:
                    w.write(row)
    return stats
