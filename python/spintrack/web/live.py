"""What the page shows of a run, kept by the tracking loop for the page's server.

The loop calls `attach` when a tracker starts and `frame` on every frame; the server
reads `snapshot` and `image`. The per-frame numbers and the path are always kept, so a
page opened mid-run shows the run so far, but images are captured only while a page has
asked for something in the last `IDLE_S` seconds, and at most `CAPTURE_HZ` times a
second, so a run nobody watches pays almost nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from spintrack import __version__
from spintrack.maps import NET_LABELS, NET_SHAPE
from spintrack.scene import Scene

if TYPE_CHECKING:
    from spintrack.pipeline import RunStats

log = logging.getLogger("spintrack")

IDLE_S = 15.0  # a page that has asked nothing for this long is gone
CAPTURE_HZ = 12.0  # frame, window and overlay captures, at most
MAP_HZ = 3.0  # map captures, at most
SERIES = 8192  # per-frame numbers kept for the traces
PATH_POINTS = 2000  # the path is sent thinned to about this many points
MAX_WIDTH = 960  # the widest frame sent, in pixels
LOG_LINES = 300
LINGER_S = 2.0  # how long a watched page has to read a run's end before it closes
# The per-frame numbers, by name: the frame index, its timestamp (ms), the motion over
# the frame (rad) and the solver's cost; NaN but for the first two when dropped.
FIELDS = ("frame", "ts", "forward", "side", "turn", "cost")


def _round(points, digits: int = 1) -> list:
    return np.round(np.asarray(points, np.float64), digits).tolist()


def _shrink(image: np.ndarray) -> np.ndarray:
    """`image` halved until it is at most `MAX_WIDTH` wide.

    The page draws it over the source's size, so this moves no point.
    """
    factor = 1
    while image.shape[1] / factor > MAX_WIDTH:
        factor *= 2
    if factor == 1:
        return image
    size = (image.shape[1] // factor, image.shape[0] // factor)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class _LogLines(logging.Handler):
    def __init__(self, view: LiveView):
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter("%(message)s"))
        self.view = view

    def emit(self, record: logging.LogRecord) -> None:
        self.view.add_log(self.format(record))


class LiveView:
    """A run's state for a page, written by one tracking thread, read by the server.

    `mode` is "run" or "gui". `stop_requested` is the page's Stop, which the run loop
    checks after every frame.
    """

    def __init__(self, mode: str = "run", clock=time.monotonic):
        self.mode = mode
        self.url: str | None = None
        self.stop_server = None  # set by whoever serves the page
        self.extra = None  # a callable adding the gui's fields to the snapshot
        self.stop_requested = False
        self._clock = clock
        self._lock = threading.Lock()
        self._asked = -np.inf
        self._log: deque[tuple[int, str]] = deque(maxlen=LOG_LINES)
        self._log_seq = 0
        self._handler = _LogLines(self)
        log.addHandler(self._handler)
        self._video: dict[str, object] = {"name": None, "index": 0, "count": 1}
        self._state, self._message = "starting", ""
        self._summary: str | None = None
        self._run = 0
        self._detach()

    def _detach(self) -> None:
        """Forget the tracker and everything kept of it."""
        self._scene: Scene | None = None
        self._stats: RunStats | None = None
        self._source = {}
        self._series = np.full((SERIES, len(FIELDS)), np.nan)
        self._n = 0  # numbers taken in; row `i % SERIES` holds number i
        self._image = None  # the last frame, as decoded
        self._last: dict | None = None  # the last frame's numbers
        self._capture: dict | None = None
        self._captured = self._mapped = -np.inf
        self._seq = self._map_seq = 0
        self._images: dict[str, tuple[int, np.ndarray]] = {}
        self._encoded: dict[str, tuple[int, bytes]] = {}
        self._t0 = self._clock()
        self._t1 = None  # when the tracker stopped

    # ----- the tracking side -----
    def video(self, name, index: int = 0, count: int = 1) -> None:
        """A new video starts (`index` of `count`)."""
        with self._lock:
            self._detach()
            self._run += 1
            self._video = {"name": str(name), "index": index, "count": count}
            self._state, self._message, self._summary = "preparing", "", None

    def rename(self, name) -> None:
        """The video's name, once a config has said which it is."""
        with self._lock:
            self._video["name"] = str(name)

    def status(self, message: str) -> None:
        with self._lock:
            self._message = message

    def attach(self, tracker, stats) -> None:
        """A tracker starts on a source; `stats` is the run's live `RunStats`."""
        with self._lock:
            self._detach()
            self._run += 1
            self._scene = Scene(tracker)
            self._stats = stats
            self._source = {"width": tracker.width, "height": tracker.height}
            self._state, self._message = "tracking", ""

    def still(self, image: np.ndarray) -> None:
        """Show a frame with no tracker, for the gui to place a ball on."""
        with self._lock:
            self._detach()
            self._source = {"width": image.shape[1], "height": image.shape[0]}
            self._seq += 1
            self._images["frame"] = (self._seq, _shrink(image))
            self._state = "preparing"

    def frame(self, frame, result) -> None:
        """Take in a frame and its result (None: dropped)."""
        now = self._clock()
        with self._lock:
            scene = self._scene
            if scene is None:
                return
            scene.update(result)
            row = self._series[self._n % SERIES]
            row[:2] = frame.index, frame.ts_ms
            if result is None:
                row[2:] = np.nan
            else:
                row[2:] = (result.forward, result.side, result.turn, result.step.cost)
            self._n += 1
            self._image = frame.image
            self._last = {
                "frame": frame.index,
                "ok": result is not None,
                "heading": None if result is None else result.heading,
                "source": "lost" if result is None else result.step.source,
                "iterations": 0 if result is None else result.step.iters,
            }
            watched = now - self._asked < IDLE_S
            if watched and now - self._captured >= 1.0 / CAPTURE_HZ:
                self._captured = now
                self._capture_frame(scene)
            if watched and now - self._mapped >= 1.0 / MAP_HZ:
                self._mapped = now
                self._capture_map(scene)

    def finish(self, state: str, summary: str | None = None) -> None:
        """The video ended: "done", "stopped" or "failed" (`summary` says why)."""
        with self._lock:
            if self._scene is not None:
                self._capture_frame(self._scene)
                self._capture_map(self._scene)
                self._t1 = self._clock()
            self._state, self._summary, self._message = state, summary, ""

    def add_log(self, line: str) -> None:
        with self._lock:
            self._log_seq += 1
            self._log.append((self._log_seq, line))

    def close(self) -> None:
        """Give a watching page time to read the end, then stop serving it."""
        log.removeHandler(self._handler)
        if self.stop_server is not None:
            if self.watched:
                time.sleep(LINGER_S)
            self.stop_server()

    # ----- captures, under the lock -----
    def _capture_frame(self, scene: Scene) -> None:
        tr = scene.tracker
        if self._image is not None:
            self._seq += 1
            self._images["frame"] = (self._seq, _shrink(self._image))
        obs = tr.engine.last_obs
        if obs is not None:
            window = np.clip(128 + 40 * obs, 0, 255).astype(np.uint8)
            self._images["window"] = (self._seq, window)
        pts, seen = scene.trail_points()
        trail = np.where(seen[:, None], np.round(pts, 1), np.nan)
        self._capture = {
            "outline": _round(scene.outline),
            "center": _round(scene.center_px),
            "radius": round(float(tr.ball_radius_px), 2),
            "ignore": [_round(poly) for poly in tr.cfg.mask.ignore],
            "trail": [None if np.isnan(x) else [x, y] for x, y in trail.tolist()],
            "axes": {
                k: None if v is None else _round(v) for k, v in scene.axes().items()
            },
            "coverage": round(float(tr.engine.map_coverage()), 4),
        }

    def _capture_map(self, scene: Scene) -> None:
        engine = scene.tracker.engine
        self._map_seq += 1
        self._images["map"] = (self._map_seq, engine.map_image("cube"))
        lighting = engine.illumination_image()
        if lighting is not None:
            self._images["lighting"] = (self._map_seq, lighting)

    # ----- the server side -----
    def touch(self) -> None:
        """A page asked for something: keep capturing for it."""
        self._asked = self._clock()

    @property
    def watched(self) -> bool:
        return self._clock() - self._asked < IDLE_S

    def request_stop(self) -> None:
        self.stop_requested = True

    def image(self, kind: str) -> tuple[bytes, str] | None:
        """The last capture of `kind` (frame, window, map, lighting), encoded."""
        with self._lock:
            held = self._images.get(kind)
            cached = self._encoded.get(kind)
        if held is None:
            return None
        seq, image = held
        if cached is None or cached[0] != seq:
            if kind == "frame":
                ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                ok, data = cv2.imencode(".png", image)
            if not ok:
                return None
            cached = (seq, data.tobytes())
            with self._lock:
                self._encoded[kind] = cached
        return cached[1], "image/jpeg" if kind == "frame" else "image/png"

    def snapshot(self, since: int = -1, log_since: int = -1) -> dict:
        """The state for the page.

        Per-frame numbers after `since`, log lines after `log_since`, and the rest
        whole.
        """
        with self._lock:
            out = {
                "mode": self.mode,
                "version": __version__,
                "run": self._run,
                "video": {**self._video, **self._source},
                "state": self._state,
                "message": self._message,
                "summary": self._summary,
                "log": [[i, line] for i, line in self._log if i > log_since],
                "net": {
                    "shape": list(NET_SHAPE),
                    "labels": [[r, c, n] for (r, c), n in NET_LABELS.items()],
                },
                "capture": {
                    "images": {k: seq for k, (seq, _) in self._images.items()},
                    "overlay": self._capture,
                },
            }
            if self._scene is not None:
                out.update(self._tracking(since))
        if self.extra is not None:
            out.update(self.extra())
        return out

    def _tracking(self, since: int) -> dict:
        stats, n, scene = self._stats, self._n, self._scene
        assert stats is not None and scene is not None  # `attach` sets both
        elapsed = max((self._t1 or self._clock()) - self._t0, 1e-9)
        first = max(since + 1, n - SERIES, 0)
        rows = self._series[np.arange(first, n) % SERIES]
        series = {
            name: [None if np.isnan(v) else v for v in rows[:, i].tolist()]
            for i, name in enumerate(FIELDS)
        }
        path = scene.path_pts
        step = max(1, len(path) // PATH_POINTS)
        thinned = path[::step]
        if len(path) and (len(thinned) == 0 or step > 1):
            thinned = np.vstack([thinned, path[-1:]])
        return {
            "progress": {
                "frames": stats.frames,
                "total": stats.total,
                "tracked": stats.tracked,
                "dropped": stats.dropped,
                "fps": stats.frames / elapsed,
            },
            "series": {"start": first, "end": n, **series},
            "path": {"n": len(path), "points": _round(thinned, 4)},
            "last": self._last,
        }
