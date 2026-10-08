"""Offline/online driver: pull frames from a source, track, fan the results out.

`run` is the loop under `spintrack run`, and `track` the one-call Python API on top of
it. Decoding runs in a background thread so that video decode and tracking overlap.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from spintrack.config import Config, apply_overrides
from spintrack.io.recorders import Recorder
from spintrack.io.records import COLUMNS, N_COLUMNS
from spintrack.io.sources import FrameSource, open_source
from spintrack.quality import RunQuality, summarize_run, walking_check
from spintrack.sphere import pixel_circle
from spintrack.tracker import Tracker

if TYPE_CHECKING:
    import polars as pl

log = logging.getLogger("spintrack")

PROGRESS_S = 10.0  # seconds between progress reports


@dataclass
class RunStats:
    frames: int = 0
    tracked: int = 0
    dropped: int = 0
    wall_s: float = 0.0
    tracking_s: float = 0.0
    total: int | None = None  # frames the run will see, when the source knows
    quality: RunQuality | None = None
    geometry: dict | None = None  # the final ball geometry, for the sidecar

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def tracking_ms_per_frame(self) -> float:
        return 1e3 * self.tracking_s / self.frames if self.frames else 0.0


def open_config(
    config: Config | str | Path | None,
    src: str | int | None = None,
    overrides: Sequence[str] = (),
    two_pass: bool = False,
) -> tuple[Config, str]:
    """Load `config`, apply `overrides`, and resolve its source.

    `config` is a config file, a `Config` (copied), or None for the defaults, and
    `overrides` are `KEY=VALUE` strings. `src` (a video path or a camera index)
    overrides the config's `video`. The returned config names the source as `video`;
    what it leaves open (the ball, the field of view, the camera position),
    `autofit.complete_config` fills in. Raises `ValueError` when there is no source,
    or `two_pass` is asked of a live camera, which cannot be read twice.
    """
    if config is None:
        cfg = Config()
    elif isinstance(config, Config):
        cfg = config.model_copy(deep=True)
    else:
        cfg = Config.load(config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if src is not None:
        cfg.video = int(src) if str(src).isdigit() else str(src)
    if cfg.video is None:
        raise ValueError(
            "no source: name the video (spintrack run VIDEO), or set `video` in the "
            "config"
        )
    src = str(cfg.video)
    if two_pass and src.isdigit():
        raise ValueError(
            "--two-pass needs a recording it can read twice, not a live camera"
        )
    return cfg, src


def _prefetch(source: FrameSource, q: queue.Queue, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            frame = source.read()
            q.put(frame)
            if frame is None:
                return
    except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
        q.put(exc)


def _join_prefetch(reader: threading.Thread, q: queue.Queue) -> None:
    """Wait for the prefetch thread, draining the queue so it can notice `stop`.

    It only checks `stop` between frames, so a producer blocked on a full queue would
    never see it. Leaving it inside `source.read()` is worse than untidy: the caller
    then closes the source underneath it, and `cv2.VideoCapture` deadlocks on a
    concurrent read and release.
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
    *,
    two_pass_source: Callable[[], FrameSource] | None = None,
    max_frames: int | None = None,
    progress: Callable[[RunStats], None] | None = None,
    debug_video: str | Path | None = None,
    save_map: str | Path | None = None,
    view=None,
) -> RunStats:
    """Track `source` and write every record to each of `recorders`.

    Returns the run statistics, with the quality summary and the final ball geometry.
    `debug_video` writes an annotated video (with the ball's axes if the config's
    `output.debug_axes`), `save_map` the final surface map. `progress` is called every
    few seconds. `view` (a `spintrack.web.live.LiveView`) sees every frame, and its
    `stop_requested` ends the run early, as Ctrl-C does.

    `two_pass_source` opens the same recording a second time: the ball is mapped in a
    throwaway first pass and this run starts from that map, so the opening frames are
    matched against a finished ball instead of an empty one. It fixes the cold start,
    not the drift that accumulates afterwards.
    """
    first = None
    if two_pass_source is not None:
        log.info("pass 1 of 2: mapping the ball")
        first = Tracker(cfg, source.width, source.height)
        again = two_pass_source()
        try:
            _track(first, again, (), max_frames, progress, view=view)
        finally:
            again.close()
        log.info(
            "pass 2 of 2: tracking from a map covering %.0f%% of the ball",
            100.0 * first.engine.map_coverage(),
        )
    tracker = Tracker(cfg, source.width, source.height)
    if first is not None:
        tracker.prime_from(first)
    debug = None
    if debug_video:
        from spintrack.debug_video import DebugCanvas, DebugVideoWriter

        canvas = DebugCanvas(tracker, axes=cfg.output.debug_axes)
        writer = DebugVideoWriter(
            debug_video, canvas.size, source.fps, cfg.output.debug_codec
        )
        debug = (canvas, writer)
    try:
        stats, per_frame = _track(
            tracker, source, recorders, max_frames, progress, debug, view
        )
    finally:
        if debug is not None:
            debug[1].close()
    if per_frame:
        frames, ts, ok, cost, iters, sources, w_cam = zip(*per_frame, strict=True)
        fps = source.fps if source.fps and source.fps > 0 else cfg.camera.fps
        stats.quality = summarize_run(
            frames,
            ts,
            ok,
            cost,
            iters,
            sources,
            np.asarray(w_cam),
            fps,
            map_coverage=tracker.engine.map_coverage(),
            illumination=tracker.engine.photometry.report(),
        )
        checks, stats.geometry = _checks(tracker, first)
        stats.quality.checks.update(checks)
        tracked = np.asarray(w_cam)[np.asarray(ok, dtype=bool)]
        walking = walking_check(tracked @ tracker.cam_to_lab.T, cfg.camera.position_deg)
        if walking is not None:
            stats.quality.checks["walking"] = walking
    if save_map:
        tracker.save_map(save_map)
    return stats


def _track(tracker, source, recorders, max_frames, progress, debug=None, view=None):
    """The frame loop of `run`; returns its stats and per-frame solver rows."""
    total = getattr(source, "n_frames", None)
    if max_frames is not None:
        total = max_frames if total is None else min(total, max_frames)
    stats = RunStats(total=total)
    if view is not None:
        view.attach(tracker, stats)
    # (frame, ts, tracked, cost, iterations, solve source, camera-frame increment)
    per_frame: list[tuple] = []
    t0 = reported = time.perf_counter()
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=32)
    reader = threading.Thread(target=_prefetch, args=(source, q, stop), daemon=True)
    reader.start()
    try:
        while (frame := q.get()) is not None:
            if isinstance(frame, Exception):
                raise frame
            t1 = time.perf_counter()
            result = tracker.process_frame(frame.image, frame.ts_ms, frame.wall_ms)
            now = time.perf_counter()
            stats.tracking_s += now - t1
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
            if debug is not None:
                canvas, writer = debug
                fps = stats.frames / max(now - t0, 1e-9)
                writer.write(canvas.render(frame.image, result, fps))
            if view is not None:
                view.frame(frame, result)
                if view.stop_requested:
                    break
            if progress is not None and now - reported >= PROGRESS_S:
                reported = now
                stats.wall_s = now - t0
                progress(stats)
            if max_frames is not None and stats.frames >= max_frames:
                break
    finally:
        stop.set()
        _join_prefetch(reader, q)
        stats.wall_s = time.perf_counter() - t0
    return stats, per_frame


def _checks(tracker: Tracker, first: Tracker | None) -> tuple[dict, dict]:
    """The tracker's lines for the quality summary, and its geometry for the sidecar."""
    checks = {}
    if tracker.watch is not None:
        from spintrack.refit import watch_checks

        checks.update(watch_checks(tracker.watch))
    if first is not None:
        note = (
            f"first pass mapped {100.0 * first.engine.map_coverage():.0f}% of the "
            f"ball; this run started from it"
        )
        moves = getattr(tracker.watch, "episodes", None)
        if moves:
            note += (
                f" and placed the window on the ball's trajectory measured there "
                f"({len(moves)} move(s))"
            )
        checks["two-pass"] = note
    cx, cy, r = pixel_circle(tracker.camera, tracker.center, tracker.half_angle)
    geometry = {
        "center_px": [cx, cy],
        "radius_px": r,
        "half_angle_deg": float(np.degrees(tracker.half_angle)),
        "window_size": tracker.geometry.size,
    }
    geometry["center_initial"] = [float(v) for v in tracker.center_initial]
    if tracker.watch is not None:
        from spintrack.refit import watch_record

        geometry["follower"] = watch_record(tracker.watch)
    return checks, geometry


@dataclass
class Track:
    """What `track` returns: the records, as in `tracks.parquet`, and quality."""

    records: np.ndarray  # (n, 25), one row per tracked frame
    quality: RunQuality | None
    columns: tuple[str, ...] = COLUMNS

    def to_polars(self) -> pl.DataFrame:
        """The records as a DataFrame with named columns."""
        from spintrack.io.parquet import to_frame

        return to_frame(self.records)


class _Rows(list):
    """A recorder that keeps the records in memory."""

    def write(self, values) -> None:
        self.append(values)

    def close(self) -> None:
        pass


def track(
    config: Config | str | Path | None,
    *,
    src: str | int | None = None,
    two_pass: bool = False,
    max_frames: int | None = None,
) -> Track:
    """Track the recording a config describes and return its records.

    `config` is a config file, a `Config` or None (the defaults); `src` overrides its
    `video`. The run is that of `spintrack run`, with the same refusals and the same
    geometry found in the recording, minus the files.
    """
    from spintrack.autofit import complete_config

    cfg, spec = open_config(config, src, two_pass=two_pass)
    complete_config(cfg, spec)
    rows = _Rows()
    source = open_source(spec)
    try:
        stats = run(
            cfg,
            source,
            [rows],
            two_pass_source=(lambda: open_source(spec)) if two_pass else None,
            max_frames=max_frames,
        )
    finally:
        source.close()
    return Track(np.array(rows).reshape(-1, N_COLUMNS), stats.quality)
