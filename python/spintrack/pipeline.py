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
from spintrack.quality import RunQuality, summarize_run, write_sidecar
from spintrack.sphere import pixel_circle
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
    quality: RunQuality | None = None

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


def _join_prefetch(reader: threading.Thread, q: queue.Queue) -> None:
    """Wait for the prefetch thread, draining the queue so it can notice `stop`.

    It only checks `stop` between frames, so a producer blocked on a full queue would
    never see it. Leaving it inside `source.read()` is worse than untidy: the caller then
    closes the source underneath it, and `cv2.VideoCapture` deadlocks on a concurrent
    read and release.
    """
    while reader.is_alive():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        reader.join(timeout=0.05)


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
    summary_out: str | None = None,
    provenance: dict | None = None,
    checks: dict | None = None,
    tracker: Tracker | None = None,
    two_pass_source: Callable[[], FrameSource] | None = None,
) -> RunStats:
    """Track every frame of `source`; returns run statistics.

    `debug_video` writes an annotated video; `refine_sweeps > 0` keeps every normalized
    window in memory, re-estimates all orientations against the complete map afterwards and
    writes the refined records to `refined_out`. The run quality summary always lands in
    `RunStats.quality`; `summary_out` also writes it as a JSON sidecar.

    `two_pass_source` opens the same recording a second time: the ball is mapped in a
    throwaway first pass and this run starts from that map, so the opening frames are
    matched against a finished ball instead of an empty one. It fixes the cold start, not
    the drift that accumulates afterwards - that is what `refine_sweeps` is for, and the
    two compose. `tracker` supplies a pre-built tracker instead of building one, which is
    how the first pass hands its map over.
    """
    first = None
    if tracker is None:
        if two_pass_source is not None:
            log.info("pass 1 of 2: mapping the ball")
            first = Tracker(cfg, source.width, source.height, params)
            source_1 = two_pass_source()
            try:
                run(
                    cfg,
                    source_1,
                    tracker=first,
                    params=params,
                    max_frames=max_frames,
                    prefetch=prefetch,
                    progress=progress,
                )
            finally:
                source_1.close()
            log.info(
                "pass 2 of 2: tracking from a map covering %.0f%% of the ball",
                100.0 * first.engine.map_coverage(),
            )
        tracker = Tracker(cfg, source.width, source.height, params)
        if first is not None:
            tracker.prime_from(first)
    stats = RunStats()
    canvas = writer = None
    if debug_video:
        from spintrack.debug_video import DebugCanvas, DebugVideoWriter

        canvas = DebugCanvas(tracker)
        writer = DebugVideoWriter(debug_video, canvas.size, source.fps, cfg.vid_codec)
    keep: list[tuple] = []  # (frame index, ts, wall, normalized window, orientation)
    # (frame, ts, tracked, cost, iterations, solve source, camera-frame increment)
    per_frame: list[tuple] = []
    t0 = time.perf_counter()
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=32)
    reader = None
    if prefetch:
        reader = threading.Thread(target=_prefetch, args=(source, q, stop), daemon=True)
        reader.start()

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
            step = result.step if result is not None else None
            per_frame.append((
                frame.index,
                frame.ts_ms,
                result is not None,
                step.cost if step is not None else np.nan,
                step.iters if step is not None else 0,
                step.source if step is not None else "lost",
                result.w_cam if result is not None else np.zeros(3),
            ))  # fmt: skip
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
                # Orientations recorded before a window move are in the old window frame;
                # `tracked_version` says which frame each one belongs to, and they are
                # brought forward together once the run is over. It is not
                # `geometry_version`: a frame whose own window moved was tracked in the
                # frame before the move.
                keep.append((frame.index, frame.ts_ms, frame.wall_ms, obs16, R_win,
                             tracker.tracked_version))  # fmt: skip
            if progress is not None and stats.frames % 500 == 0:
                stats.wall_s = time.perf_counter() - t0
                progress(stats)
            if max_frames is not None and stats.frames >= max_frames:
                break
    finally:
        stop.set()
        if reader is not None:
            _join_prefetch(reader, q)
        stats.wall_s = time.perf_counter() - t0
        if writer is not None:
            writer.close()
    scale_report = None
    if per_frame:
        frames, ts, ok, cost, iters, sources, w_cam = zip(*per_frame, strict=True)
        fps = source.fps if source.fps and source.fps > 0 else cfg.src_fps
        stats.quality = summarize_run(
            frames,
            ts,
            ok,
            cost,
            iters,
            sources,
            np.asarray(w_cam),
            fps if fps > 0 else None,
            map_coverage=tracker.engine.map_coverage(),
            illumination=tracker.engine.photometry.report(),
        )
        stats.quality.checks.update(checks or {})
        if tracker.refits:
            stats.quality.checks["ball centre"] = (
                f"followed a moving ball over {len(tracker.refits)} stretch(es): "
                + ", ".join(
                    f"frames {e.start}-{e.end}, up to {e.max_shift_px:.0f} px"
                    for e in tracker.refits[:3]
                )
            )
        elif tracker.watch is not None:
            stats.quality.checks["ball centre"] = "stable"
        if first is not None:
            stats.quality.checks["two-pass"] = (
                f"first pass mapped {100.0 * first.engine.map_coverage():.0f}% of the "
                f"ball; this run started from it"
            )
        if tracker.scale_check is not None:
            verdict = tracker.scale_check.result()
            stats.quality.checks["rotation scale"] = verdict.line()
            scale_report = verdict.report()
        if summary_out:
            cx, cy, r = pixel_circle(tracker.camera, tracker.centre, tracker.half_angle)
            geometry = {
                "centre_px": [cx, cy],
                "radius_px": r,
                "half_angle_deg": float(np.degrees(tracker.half_angle)),
                "window_size": tracker.geometry.size,
            }
            if scale_report is not None:
                geometry["scale_check"] = scale_report
            geometry["centre_initial"] = [float(v) for v in tracker.centre_initial]
            geometry["refits"] = [e.as_dict() for e in tracker.refits]
            write_sidecar(
                summary_out, stats.quality, {**(provenance or {}), "geometry": geometry}
            )
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
        online = tracker.orientations_in_current_window([(k[4], k[5]) for k in keep])
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
