"""`spintrack gui`: a config being edited, tracked live on a clip of its video.

The session holds the config and a clip, a run of the video's frames. The main thread
tracks the clip with the config, over and over, and shows every frame on the page
(`LiveView`); a change from the page restarts it, so its effect shows within seconds.
Slow work (finding the ball, fitting the field of view, tracking the whole video) runs
on the same thread between clips, so that there is only ever one tracker.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spintrack.autofit import complete_config, prepare_config
from spintrack.config import Config, leading_comments, with_changes
from spintrack.detect import DetectionError, fit_circle
from spintrack.io.sources import VideoSource
from spintrack.pipeline import RunStats, open_config
from spintrack.tracker import Tracker

if TYPE_CHECKING:
    from spintrack.web.live import LiveView

log = logging.getLogger("spintrack")

CLIP_S = 10.0  # the clip's default length, in seconds
HOLD_S = 1.0  # the pause on a clip's last frame before it starts again


class _Changes(BaseModel):
    changes: dict


class _Square(BaseModel):
    corners: list[tuple[float, float]]
    plane: str = "xy"


class _Play(BaseModel):
    paused: bool | None = None
    speed: float | None = None
    clip: tuple[int, int] | None = None
    step: bool = False
    resume: bool = False


class _Save(BaseModel):
    path: str | None = None


class _Track(BaseModel):
    overwrite: bool = False


class GuiSession:
    """The state of `spintrack gui` and the loop that tracks with it.

    The page's requests arrive on the server's thread and only change the state under
    the lock; `serve` does the tracking and the slow work on the calling thread.
    """

    def __init__(self, config, src, overrides, save_to: Path):
        cfg, src = open_config(config, src, overrides)
        if src.isdigit():
            raise ValueError(
                "the gui works on a recording, not a live camera: record a clip and "
                "open that"
            )
        source = VideoSource(src)
        try:
            self.width, self.height = source.width, source.height
            self.fps = source.fps if source.fps > 0 else cfg.camera.fps or 30.0
            n = source.n_frames
            if n is None:
                duration = source._container.duration  # microseconds, or None
                n = round(duration / 1e6 * self.fps) if duration else 0
            self.n_frames = int(n)
        finally:
            source.close()
        self.cfg = cfg
        self.src = src
        self.save_to = Path(save_to)
        self.clip = (0, max(1, min(self.n_frames, round(CLIP_S * self.fps))))
        self.speed = 1.0  # times real time; 0: as fast as it tracks
        self.paused = False
        self.version = 0  # bumped by every change of the config or the clip
        # The version last saved; the first, when it is the file Save writes to.
        same = config is not None and Path(config).resolve() == self.save_to.resolve()
        self.saved = 0 if same and not overrides else None
        self.busy: str | None = None  # the slow work under way
        self.error: str | None = None
        self.notes: dict[str, str] = {}  # what was found in the recording
        self.full_run: dict | None = None  # the last whole-video run
        self.hold = False  # after a whole-video run, until the page resumes
        self.view: LiveView | None = None
        self._defaults = Config().model_dump(mode="json")
        self._task: tuple[str, dict] | None = None
        self._steps = 0
        self._completed = -1  # the version `complete_config` last ran on
        # The camera at the animal's back, with the elevation and twist its silhouette
        # gave, for tracking until the page says where the camera sits.
        self._provisional: tuple[float, float, float] | None = None
        self._lock = threading.Lock()
        self._changed = threading.Event()  # restart the clip
        self._quit = threading.Event()

    @property
    def _page(self) -> LiveView:
        """The page's live view, which `serve` attaches."""
        assert self.view is not None
        return self.view

    # ----- the loop -----
    def serve(self, view: LiveView) -> None:
        """Track until the page quits (or Ctrl-C); `view` is the page's `LiveView`."""
        self.view = view
        view.video(self.src)
        while not self._quit.is_set():
            with self._lock:
                task, self._task = self._task, None
            if task is not None:
                self._do(*task)
            elif self.hold:
                self._changed.wait(0.2)
                self._changed.clear()
            elif not self._ready():
                self._complete()
            else:
                self._play()

    def close(self) -> None:
        self._quit.set()
        self._changed.set()
        if self.view is not None:
            self.view.request_stop()

    def _ready(self) -> bool:
        """Whether the clip can be tracked.

        With the camera's position, or a provisional one until the page says where the
        camera sits.
        """
        cfg = self.cfg
        placed = cfg.camera.to_animal() is not None or self._provisional is not None
        return bool(cfg.ball.rim and cfg.camera.vfov_deg is not None and placed)

    def _set_busy(self, message: str | None) -> None:
        self.busy = message
        self._page.status(message or "")

    def _complete(self) -> None:
        """Fill in what the config leaves open, once per version of it."""
        if self._completed == self.version:
            self._changed.wait(0.2)
            self._changed.clear()
            return
        self._completed = self.version
        with self._lock:
            cfg, version = self.cfg.model_copy(deep=True), self.version
        if not cfg.ball.rim:
            self._still(cfg)
        self._set_busy("finding what the config leaves open: the ball, the camera")
        try:
            prepared = complete_config(cfg, self.src, require_position=False)
        except (DetectionError, ValueError) as exc:
            self.error = f"{exc}. Drag the circle onto the ball."
            return
        finally:
            self._set_busy(None)
        self._adopt(cfg, version, prepared)

    def _adopt(self, cfg: Config, version: int, prepared) -> None:
        """Take a config the slow work completed, unless the page changed it since."""
        with self._lock:
            if self.version != version:
                return
            self.cfg = cfg
            self.version += 1
            self._completed = self.version
            self.error = None
            if prepared is not None:
                self.notes["ball"] = prepared.line()
                if prepared.vfov is not None:
                    self.notes["field of view"] = prepared.vfov.line()
                camera = prepared.camera
                if camera is not None:
                    self.notes["camera"] = (
                        f"{camera.line()}. Check that the trail starts under the "
                        f"animal."
                    )
                if prepared.camera_position == "unknown":
                    self._provisional = (camera.elevation_deg, 180.0, camera.twist_deg)
        self._changed.set()

    def _still(self, cfg: Config) -> None:
        """Show the clip's first frame, untracked, to place a ball on."""
        source = VideoSource(self.src)
        try:
            source.seek(self.clip[0])
            frame = source.read()
        finally:
            source.close()
        if frame is not None:
            self._page.still(frame.image)

    def _play(self) -> None:
        """Track the clip once, paced to `speed`, unless a change cuts it short."""
        with self._lock:
            cfg, (start, end) = self.cfg.model_copy(deep=True), self.clip
            self._changed.clear()
        if cfg.camera.to_animal() is None:  # until the page says where it sits
            cfg.camera.position_deg = self._provisional
        try:
            tracker = Tracker(cfg, self.width, self.height)
        except ValueError as exc:
            self.error = str(exc)
            self._changed.wait()
            return
        stats = RunStats(total=end - start)
        self._page.attach(tracker, stats)
        source = VideoSource(self.src)
        try:
            source.seek(start)
            due = time.perf_counter()
            while stats.frames < end - start:
                if self._changed.is_set() or self._quit.is_set():
                    return
                if self.paused and stats.frames > 0:
                    if self._steps <= 0:
                        self._changed.wait(0.05)
                        due = time.perf_counter()
                        continue
                    self._steps -= 1
                frame = source.read()
                if frame is None:
                    break
                result = tracker.process_frame(frame.image, frame.ts_ms)
                stats.frames += 1
                stats.dropped += result is None
                stats.tracked += result is not None
                self._page.frame(frame, result)
                if self.speed > 0 and not self.paused:
                    due += 1.0 / (self.fps * self.speed)
                    wait = due - time.perf_counter()
                    if wait > 0:
                        self._changed.wait(wait)
                    elif wait < -0.25:  # far behind: do not race to catch up
                        due = time.perf_counter()
        finally:
            source.close()
        self._changed.wait(HOLD_S)

    def _do(self, kind: str, payload: dict) -> None:
        with self._lock:
            cfg, version = self.cfg.model_copy(deep=True), self.version
        try:
            if kind == "detect":
                self._set_busy("finding the ball")
                cfg.ball.rim = []
                self._adopt(cfg, version, prepare_config(cfg, self.src))
            elif kind == "vfov":
                self._set_busy("fitting the field of view")
                cfg.camera.vfov_deg = None
                self._adopt(cfg, version, prepare_config(cfg, self.src))
            elif kind == "track":
                self._track_all(payload["overwrite"])
        except (DetectionError, OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            self._set_busy(None)

    def _track_all(self, overwrite: bool) -> None:
        """Save the config and track the whole video with it, as `spintrack run -c`."""
        from spintrack.cli import _run_job

        path = self._save(None)
        args = argparse.Namespace(
            out=None,
            force=overwrite,
            two_pass=False,
            max_frames=None,
            debug_video=False,
            save_map=False,
            udp=None,
            tcp=None,
            serial=None,
            print=False,
            verbose=False,
        )
        view = self._page
        view.stop_requested = False
        self.full_run = {"state": "tracking", "folder": str(self._folder())}
        self._set_busy("tracking the whole video")
        view.video(self.src)
        ok = _run_job(args, str(path), self.src, [], view)
        stopped = view.stop_requested
        view.stop_requested = False
        state = "stopped" if stopped else "done" if ok else "failed"
        self.full_run = {"state": state, "folder": str(self._folder())}
        self.hold = True

    def _folder(self) -> Path:
        """Where a run of the video writes: NAME_spintrack next to it."""
        name = Path(self.cfg.output.name or Path(self.src).stem).name
        return Path(self.src).parent / f"{name}_spintrack"

    def _save(self, path: str | None) -> Path:
        with self._lock:
            target = Path(path) if path else self.save_to
            cfg, version = self.cfg, self.version
        if target.suffix.lower() != ".toml":
            raise ValueError(f"{target}: a config is a .toml file")
        if target.exists():
            comments = leading_comments(target)
        else:
            video = Path(self.src).name
            comments = [
                f"# Written by `spintrack gui` on {video}. Track videos with it:",
                f"# spintrack run VIDEO... -c {target.name}",
            ]
        target.parent.mkdir(parents=True, exist_ok=True)
        cfg.save(target, comments)
        with self._lock:
            self.save_to = target
            self.saved = version
        log.info("saved %s", target)
        return target

    # ----- the page's side -----
    def state(self) -> dict:
        """The gui's part of the page's state."""
        with self._lock:
            cfg = self.cfg
            ball = None
            if len(cfg.ball.rim) >= 3:
                cx, cy, r = fit_circle(cfg.ball.rim)
                ball = [float(cx), float(cy), float(r)]
            command = f"spintrack run VIDEO... -c {self.save_to}"
            return {
                "gui": {
                    "config": cfg.model_dump(mode="json"),
                    "defaults": self._defaults,
                    "ball": ball,
                    "version": self.version,
                    "saved": self.saved == self.version,
                    "save_to": str(self.save_to),
                    "command": command,
                    "clip": list(self.clip),
                    "frames": self.n_frames,
                    "fps": self.fps,
                    "size": [self.width, self.height],
                    "paused": self.paused,
                    "speed": self.speed,
                    "busy": self.busy,
                    "error": self.error,
                    "notes": self.notes,
                    "camera_unset": cfg.camera.to_animal() is None,
                    "provisional": self._provisional,
                    "full_run": self.full_run,
                    "hold": self.hold,
                },
            }

    def change(self, changes: dict) -> None:
        """Apply the page's changes to the config (dotted keys) and restart the clip."""
        with self._lock:
            if self.busy:
                raise HTTPException(409, f"busy: {self.busy}")
            try:
                self.cfg = with_changes(self.cfg, changes, "gui")
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            self.version += 1
            self.error = None
            if changes.keys() & {"camera.position_deg", "camera.rotation"}:
                self.notes["camera"] = "set on this page"
        self._changed.set()

    def add_routes(self, api: APIRouter) -> None:
        @api.post("/config")
        def config(body: _Changes) -> dict:
            self.change(body.changes)
            return {}

        @api.post("/detect")
        def detect() -> dict:
            self._schedule("detect")
            return {}

        @api.post("/fit-vfov")
        def fit_vfov() -> dict:
            self._schedule("vfov")
            return {}

        @api.post("/square")
        def square(body: _Square) -> dict:
            from spintrack.calibrate.square import camera_to_lab_from_square
            from spintrack.camera import source_camera
            from spintrack.geometry import matrix_to_rotvec

            camera_cfg = self.cfg.camera
            if camera_cfg.vfov_deg is None:
                raise HTTPException(409, "the field of view is not known yet")
            camera = source_camera(
                self.width, self.height, camera_cfg.vfov_deg, camera_cfg.fisheye
            )
            try:
                R = camera_to_lab_from_square(body.corners, camera, body.plane)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(400, str(exc)) from None
            rotation = [round(float(v), 6) for v in matrix_to_rotvec(R)]
            self.change({"camera.rotation": rotation})
            return {"rotation": rotation}

        @api.post("/play")
        def play(body: _Play) -> dict:
            restart = False
            with self._lock:
                if body.paused is not None:
                    self.paused = body.paused
                if body.speed is not None:
                    self.speed = max(0.0, body.speed)
                if body.step:
                    self._steps += 1
                if body.clip is not None:
                    start = int(np.clip(body.clip[0], 0, max(self.n_frames - 1, 0)))
                    end = int(np.clip(body.clip[1], start + 1, max(self.n_frames, 1)))
                    self.clip = (start, end)
                    restart = True
                if body.resume:
                    self.hold = False
                    restart = True
            if restart:
                self._changed.set()
            return {}

        @api.post("/save")
        def save(body: _Save) -> dict:
            try:
                path = self._save(body.path)
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from None
            return {"path": str(path)}

        @api.post("/track")
        def track(body: _Track) -> dict:
            if self.cfg.camera.to_animal() is None:
                raise HTTPException(409, "say where the camera sits first")
            folder = self._folder()
            if (folder / "tracks.parquet").exists() and not body.overwrite:
                raise HTTPException(409, f"{folder} holds an earlier run")
            self._schedule("track", overwrite=body.overwrite)
            return {"folder": str(folder)}

        @api.post("/quit")
        def quit_() -> dict:
            self._quit.set()
            self._changed.set()
            return {}

    def _schedule(self, kind: str, **payload) -> None:
        with self._lock:
            if self.busy:
                raise HTTPException(409, f"busy: {self.busy}")
            if self._task is not None:
                raise HTTPException(409, f"busy: {self._task[0]}")
            self._task = (kind, payload)
        self._changed.set()
