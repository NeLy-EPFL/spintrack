"""Annotated debug video.

The left panel is the source frame with the ball's outline and the trail the animal has
walked over its surface (and, on request, the ball's orientation axes). On the right are
the normalized tracking window and the fictive path, and below them the surface map as
an unfolded dice centered on the face the camera looks at and oriented like the image
(`maps.NET_LABELS`), so at `R = I` its middle tile is the window. Two of the net's empty
tiles hold the static lighting field and the frame's numbers.

Enabled with `debug_video = true` in the config's `[output]` or `spintrack run
--debug-video`. The config's `mask.ignore` regions are outlined in red.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from spintrack.maps import NET_LABELS, NET_SHAPE
from spintrack.scene import TRAIL_FRAMES, Scene
from spintrack.tracker import FrameResult, Tracker

log = logging.getLogger(__name__)

AXIS_BGR = ((80, 80, 255), (80, 220, 80), (255, 140, 80))  # x red, y green, z blue
LABEL_BGR = (255, 255, 0)
TEXT_BGR = (230, 230, 230)
OUTLINE_BGR = (0, 200, 0)
IGNORE_BGR = (60, 60, 200)
PATH_BGR = (0, 255, 255)
HEAD_BGR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

__all__ = ["TRAIL_FRAMES", "DebugCanvas", "DebugVideoWriter"]

# Trail color by age, oldest first: dark blue to cyan.
TRAIL_BGR = [
    tuple(bgr)
    for bgr in np.linspace((160, 60, 0), (255, 255, 0), 6).round().astype(int).tolist()
]


def _label(panel: np.ndarray, text: str, x: int = 6, y: int = 16) -> None:
    cv2.putText(panel, text, (x, y), FONT, 0.45, LABEL_BGR, 1, cv2.LINE_AA)


FOURCC = {
    "h264": "avc1",
    "avc1": "avc1",
    "mp4v": "mp4v",
    "xvid": "XVID",
    "mjpg": "MJPG",
    "raw": "I420",
}

# `debug_codec` -> (PyAV encoder, encoder options) for the codecs OpenCV cannot write.
AV_CODEC = {
    "h264": ("libx264", {"preset": "veryfast", "crf": "20"}),
    "avc1": ("libx264", {"preset": "veryfast", "crf": "20"}),
    "hevc": ("libx265", {"preset": "veryfast", "crf": "24"}),
    "h265": ("libx265", {"preset": "veryfast", "crf": "24"}),
    "vp9": ("libvpx-vp9", {"deadline": "realtime", "cpu-used": "5", "crf": "30"}),
}


def _import_av():
    """Import PyAV lazily: it costs ~50 ms and only the debug video needs it."""
    import av

    return av


class DebugCanvas:
    """Compose one debug frame from the tracker state.

    The right-hand column is five tiles tall and four wide: the window and the path take
    two by two each, the map net the 3x4 below them.
    """

    def __init__(self, tracker: Tracker, height: int = 480, axes: bool = False):
        self.tracker = tracker
        self.axes = axes
        # `height` is a hint: snap it to an exact integer decimation of the source when
        # one is within reach, because INTER_AREA is ~50x faster there.
        k = max(1, round(tracker.height / height))
        if tracker.height % k == 0 and tracker.width % k == 0:
            height = tracker.height // k
        self.height = height
        self.scale = height / tracker.height
        self.main_w = round(tracker.width * self.scale)
        self.tile = height // 5
        self.width = self.main_w + 4 * self.tile
        self.ignore = [
            np.round(np.reshape(poly, (-1, 2)) * self.scale).astype(np.int32)
            for poly in tracker.cfg.mask.ignore
        ]
        self.scene = Scene(tracker)
        self._outline_version = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def trail_points(self) -> tuple[np.ndarray, np.ndarray]:
        """The scene's trail in main-panel coordinates, and which points are seen."""
        pts, seen = self.scene.trail_points()
        return pts * self.scale, seen

    def render(
        self, gray: np.ndarray, result: FrameResult | None, fps: float | None = None
    ) -> np.ndarray:
        t, scene = self.tile, self.scene
        scene.update(result)
        if self._outline_version != scene.geometry_version:
            self._outline_version = scene.geometry_version
            self.outline = np.round(scene.outline * self.scale).astype(np.int32)
        main = cv2.resize(
            gray, (self.main_w, self.height), interpolation=cv2.INTER_AREA
        )
        main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
        cv2.polylines(main, [self.outline], True, OUTLINE_BGR, 1, cv2.LINE_AA)
        cv2.polylines(main, self.ignore, True, IGNORE_BGR, 1, cv2.LINE_AA)
        self._draw_trail(main)
        if self.axes:
            self._draw_axes(main, result)
        side = np.zeros((self.height, 4 * t, 3), np.uint8)
        self._draw_window(side[: 2 * t, : 2 * t])
        self._draw_path(side[: 2 * t, 2 * t :], result)
        net = side[2 * t : 5 * t]
        self._draw_net(net)
        self._draw_lighting(net[:t, 3 * t :])
        self._draw_status(net[2 * t :, 2 * t :], result, fps)
        return np.hstack([main, side])

    def _draw_axes(self, main: np.ndarray, result: FrameResult | None) -> None:
        """The ball's axes (arrows, turning with it) and the lab axes (thin, fixed)."""
        axes, scale = self.scene.axes(), self.scale
        c = tuple(round(v * scale) for v in self.scene.center_px)

        def px(tip):
            return round(tip[0] * scale), round(tip[1] * scale)

        if result is not None:
            for tip, color in zip(axes["ball"], AXIS_BGR, strict=True):
                cv2.arrowedLine(main, c, px(tip), color, 2, cv2.LINE_AA, tipLength=0.2)
        for i, (tip, color) in enumerate(zip(axes["lab"], AXIS_BGR, strict=True)):
            end = px(tip)
            cv2.line(main, c, end, color, 1, cv2.LINE_AA)
            cv2.putText(main, "xyz"[i], (end[0] + 3, end[1] + 3), FONT, 0.45, color, 1)

    def _draw_trail(self, main: np.ndarray) -> None:
        """Draw the trail the animal has walked over the ball, as FicTrac does: only
        the near side, brighter with recency (`Scene.trail_points`)."""
        n = self.scene.n_trail
        if self.scene.R_cam is None or n < 2:
            return
        pts, seen = self.trail_points()
        pts = pts.astype(np.int32)
        bands = len(TRAIL_BGR)
        for band in range(bands):
            lo = band * n // bands
            hi = (band + 1) * n // bands + 1  # overlap, to join the bands
            edges = np.flatnonzero(np.diff(np.r_[False, seen[lo:hi], False]))
            runs = [
                pts[lo + i : lo + j]
                for i, j in zip(edges[::2], edges[1::2])
                if j - i > 1
            ]
            cv2.polylines(main, runs, False, TRAIL_BGR[band], 1, cv2.LINE_AA)
        if seen[-1]:
            cv2.circle(main, tuple(pts[-1]), 3, TRAIL_BGR[-1], -1, cv2.LINE_AA)

    def _draw_window(self, panel: np.ndarray) -> None:
        obs = self.tracker.engine.last_obs
        if obs is not None:
            win = np.clip(128 + 40 * obs, 0, 255).astype(np.uint8)
            win = cv2.resize(win, panel.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            panel[:] = win[..., None]
        _label(panel, "window")

    def _draw_path(
        self, panel: np.ndarray, result: FrameResult | None, margin: int = 12
    ) -> None:
        """The fictive path (x up, y left), scaled to fit the panel."""
        h, w = panel.shape[:2]
        path = self.scene.path_pts
        if len(path) >= 2 and min(h, w) >= 60:
            b = self.scene.path_bbox
            span = max(b[1] - b[0], b[3] - b[2], 1e-6)
            size = min(h, w) - 2 * margin
            path = path[-20000:]
            pts = np.empty((len(path), 2), np.int32)
            # Centered: the bounding box's shorter side gets the slack.
            x0 = (w - (b[3] - b[2]) / span * size) / 2
            y0 = (h + (b[1] - b[0]) / span * size) / 2
            pts[:, 0] = x0 + (b[3] - path[:, 1]) / span * size
            pts[:, 1] = y0 - (path[:, 0] - b[0]) / span * size
            cv2.polylines(panel, [pts], False, PATH_BGR, 1, cv2.LINE_AA)
            head = tuple(int(v) for v in pts[-1])
            cv2.circle(panel, head, 3, HEAD_BGR, -1, cv2.LINE_AA)
            if result is not None:
                hd = result.heading
                tip = (int(head[0] - 14 * np.sin(hd)), int(head[1] - 14 * np.cos(hd)))
                cv2.arrowedLine(panel, head, tip, HEAD_BGR, 1, cv2.LINE_AA, 0, 0.4)
            text = f"{span:.2g} r"
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.4, 1)
            cv2.putText(panel, text, (w - tw - 6, h - 6), FONT, 0.4, TEXT_BGR, 1)
        _label(panel, "path")

    def _draw_net(self, panel: np.ndarray) -> None:
        """The map as an unfolded dice filling `panel` (3 x 4 tiles); the six unused
        tiles stay black."""
        rows, cols = NET_SHAPE
        t = panel.shape[0] // rows
        net = self.tracker.engine.map_image("cube")
        size = (cols * t, rows * t)
        interp = cv2.INTER_AREA if net.shape[1] > size[0] else cv2.INTER_LINEAR
        net = cv2.resize(net, size, interpolation=interp)
        for (r, c), name in NET_LABELS.items():
            tile = panel[r * t : (r + 1) * t, c * t : (c + 1) * t]
            tile[:] = net[r * t : (r + 1) * t, c * t : (c + 1) * t, None]
            cv2.rectangle(tile, (0, 0), (t - 1, t - 1), (70, 70, 70), 1)
            cv2.putText(tile, name, (4, 12), FONT, 0.35, LABEL_BGR, 1, cv2.LINE_AA)
        _label(panel[:t, :t], "map")

    def _draw_lighting(self, panel: np.ndarray) -> None:
        """The lighting gain the tracker divides out of the window (dark: shadow)."""
        illum = self.tracker.engine.illumination_image()
        if illum is None:
            return
        s = min(panel.shape[:2])
        panel[:s, :s] = cv2.resize(illum, (s, s), interpolation=cv2.INTER_AREA)[
            ..., None
        ]
        _label(panel, "lighting")

    def _draw_status(
        self, panel: np.ndarray, result: FrameResult | None, fps: float | None
    ) -> None:
        lines = [f"frame {self.tracker.frame - 1}"]
        if result is None:
            lines.append("dropped")
        else:
            st = result.step
            lines += [
                f"{st.source}  cost {st.cost:.3f}",
                f"{st.iters} iterations",
                f"heading {np.degrees(result.heading):.0f} deg",
            ]
        if fps:
            lines.append(f"{fps:.0f} fps")
        step = max(12, min(18, panel.shape[0] // (len(lines) + 1)))
        for i, line in enumerate(lines):
            y = 6 + step * (i + 1)
            cv2.putText(panel, line, (8, y), FONT, 0.42, TEXT_BGR, 1, cv2.LINE_AA)


class DebugVideoWriter:
    """Write debug canvases to a video file (codec from the config's `debug_codec`).

    H.264, HEVC and VP9 are encoded with PyAV. That is not a preference: the
    `opencv-python` wheels bundle an FFmpeg built without libx264, so `cv2.VideoWriter`
    cannot open an H.264 stream and silently falls back to MPEG-4 Part 2, which browsers
    and most editors refuse to play. The PyAV wheels bundle their own FFmpeg with those
    encoders, so nothing has to be installed system-wide. Frame threading also lets the
    encoder work on other cores while tracking continues.

    The fourcc codecs OpenCV does handle (`mp4v`, `xvid`, `mjpg`, `raw`) still go
    through `cv2.VideoWriter`. Without PyAV, H.264 falls back to it too, and a warning
    names the consequence.
    """

    def __init__(
        self, path: str | Path, size: tuple[int, int], fps: float, codec: str = "h264"
    ):
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.codec = codec.lower()
        self._container = self._stream = self._writer = None
        self._pad = (0, 0)
        self._pts = 0
        if self.codec in AV_CODEC:
            self._open_av()
        if self._container is None:
            self._open_opencv()

    # ----- PyAV -----
    def _open_av(self) -> None:
        try:
            av = _import_av()
        except ImportError:
            log.warning(
                "PyAV is not installed, so %s cannot be encoded; writing %s as MPEG-4 "
                "Part 2, which browsers and most editors cannot play. "
                "Install it with `pip install av`.",
                self.codec,
                self.path,
            )
            return
        encoder, options = AV_CODEC[self.codec]
        w, h = self.size
        # yuv420p is what every player can decode, and it needs even dimensions
        self._pad = (w % 2, h % 2)
        container = None
        try:
            mp4 = self.path.suffix.lower() in (".mp4", ".m4v", ".mov")
            container = av.open(
                str(self.path), "w", options={"movflags": "+faststart"} if mp4 else {}
            )
            stream = container.add_stream(
                encoder, rate=Fraction(self.fps).limit_denominator(65535)
            )
            stream.width = w + self._pad[0]
            stream.height = h + self._pad[1]
            stream.pix_fmt = "yuv420p"
            stream.options = options
            # encode on other cores while tracking continues: ~1.4x over slice
            # threading, and over piping raw frames to a system ffmpeg
            stream.thread_type = "FRAME"
        except (av.FFmpegError, ValueError) as exc:  # pragma: no cover
            if container is not None:
                container.close()
            log.warning(
                "PyAV cannot encode %s (%s); using OpenCV's encoder", self.codec, exc
            )
            return
        self._container, self._stream = container, stream
        self._frame_from = av.VideoFrame.from_ndarray

    # ----- OpenCV -----
    def _open_opencv(self) -> None:
        fourcc = FOURCC.get(self.codec, "mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size
        )
        if not self._writer.isOpened() and fourcc != "mp4v":
            log.warning(
                "OpenCV cannot encode %s (its bundled FFmpeg has no such encoder); "
                "writing %s as MPEG-4 Part 2, which browsers and most editors cannot "
                "play.",
                self.codec,
                self.path,
            )
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
            )
        if not self._writer.isOpened():
            raise OSError(f"could not open debug video writer for {self.path}")

    # ----- writing -----
    def write(self, canvas: np.ndarray) -> None:
        if self._container is None:
            self._writer.write(canvas)
            return
        if canvas.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError(
                f"canvas is {canvas.shape[1]}x{canvas.shape[0]}, "
                f"writer opened for {self.size[0]}x{self.size[1]}"
            )
        if any(self._pad):
            canvas = np.pad(canvas, ((0, self._pad[1]), (0, self._pad[0]), (0, 0)))
        elif not canvas.flags.c_contiguous:
            canvas = np.ascontiguousarray(canvas)
        frame = self._frame_from(canvas, format="bgr24")
        frame.pts = self._pts
        self._pts += 1
        self._container.mux(self._stream.encode(frame))

    def close(self) -> None:
        if self._container is not None:
            container, stream = self._container, self._stream
            self._container = self._stream = None
            try:
                container.mux(stream.encode())  # flush the encoder's buffered frames
            finally:
                container.close()
        elif self._writer is not None:
            self._writer.release()
            self._writer = None
