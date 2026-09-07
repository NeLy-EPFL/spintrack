"""Annotated debug video: source frame with ball and axes, tracking window, map, path.

Enabled with `save_debug: y` in the config or `spintrack run --debug-video`. Rendering costs
a few milliseconds per frame, so it is off by default.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from spintrack.geometry import normalize
from spintrack.sphere import ball_outline
from spintrack.tracker import FrameResult, Tracker

log = logging.getLogger(__name__)

AXIS_BGR = (
    (80, 80, 255),
    (80, 220, 80),
    (255, 140, 80),
)  # x red, y green, z blue (BGR)
FOURCC = {
    "h264": "avc1",
    "avc1": "avc1",
    "mp4v": "mp4v",
    "xvid": "XVID",
    "mjpg": "MJPG",
    "raw": "I420",
}

# `vid_codec` -> (PyAV encoder, encoder options) for the codecs OpenCV cannot write.
# H.264 defaults to the `veryfast` preset: measured on a 1280x504 canvas it runs at
# ~1300 fps, faster than OpenCV's MPEG-4 and a quarter of the size.
AV_CODEC = {
    "h264": ("libx264", {"preset": "veryfast", "crf": "20"}),
    "avc1": ("libx264", {"preset": "veryfast", "crf": "20"}),
    "hevc": ("libx265", {"preset": "veryfast", "crf": "24"}),
    "h265": ("libx265", {"preset": "veryfast", "crf": "24"}),
    "vp9": ("libvpx-vp9", {"deadline": "realtime", "cpu-used": "5", "crf": "30"}),
}


def _import_av():
    """Import PyAV, lazily: it costs ~50 ms and only the debug video needs it."""
    import av

    return av


class DebugCanvas:
    """Compose one debug frame from the tracker state."""

    def __init__(self, tracker: Tracker, height: int = 480, panel: int = 240):
        self.tracker = tracker
        # `height` is a hint. OpenCV's INTER_AREA has a fast path for exact integer
        # decimation and a general resampler otherwise, and the two differ by ~50x on a
        # frame this size, so snap to an exact factor when one is within reach.
        k = max(1, round(tracker.height / height))
        if k > 1 and tracker.height % k == 0 and tracker.width % k == 0:
            height = tracker.height // k
        self.height = height
        self.panel = panel
        scale = height / tracker.height
        self.scale = scale
        self.main_w = round(tracker.width * scale)
        self.width = self.main_w + panel * 2
        self.axis_len = 0.8 * np.sin(tracker.half_angle)
        self._geometry_version = -1
        self._update_outline()
        self._path = np.empty((1024, 2), np.float64)  # grown by doubling
        self._n_path = 0
        self.path_bbox = [-0.1, 0.1, -0.1, 0.1]

    @property
    def path_pts(self) -> np.ndarray:
        """The integrated path so far, as an (n, 2) array of (x, y)."""
        return self._path[: self._n_path]

    def _append_path(self, x: float, y: float) -> None:
        if self._n_path == len(self._path):
            self._path = np.resize(self._path, (2 * len(self._path), 2))
        self._path[self._n_path] = (x, y)
        self._n_path += 1

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def _project_axis(self, axis_cam: np.ndarray) -> tuple[int, int]:
        tip = normalize(self.tracker.centre + self.axis_len * axis_cam)
        x, y, _ = self.tracker.camera.project(tip)
        return int(x * self.scale), int(y * self.scale)

    def _update_outline(self) -> None:
        """Re-project the ball outline; the window may have been moved onto a moved ball."""
        tracker, scale = self.tracker, self.scale
        self._geometry_version = tracker.geometry_version
        self.outline = np.round(
            ball_outline(tracker.camera, tracker.centre, tracker.half_angle, 90) * scale
        )
        cx, cy, _ = tracker.camera.project(tracker.centre)
        self.centre_px = (float(cx) * scale, float(cy) * scale)

    def render(
        self, gray: np.ndarray, result: FrameResult | None, fps: float | None = None
    ):
        tr = self.tracker
        if tr.geometry_version != self._geometry_version:
            self._update_outline()
        main = cv2.resize(
            gray, (self.main_w, self.height), interpolation=cv2.INTER_AREA
        )
        main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
        cv2.polylines(
            main, [self.outline.astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA
        )
        c = (int(self.centre_px[0]), int(self.centre_px[1]))
        if result is not None:
            # Ball orientation: a gnomon that rotates with the ball (camera frame).
            for i, colour in enumerate(AXIS_BGR):
                tip = self._project_axis(result.R_cam[:, i])
                cv2.arrowedLine(main, c, tip, colour, 2, cv2.LINE_AA, tipLength=0.2)
        # Lab axes at the ball centre (fixed): thin lines with labels.
        for i, colour in enumerate(AXIS_BGR):
            tip = self._project_axis(tr.cam_to_lab.T[:, i])
            cv2.line(main, c, tip, colour, 1, cv2.LINE_AA)
            cv2.putText(
                main,
                "xyz"[i],
                (tip[0] + 3, tip[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )
        text = f"frame {tr.frame - 1}"
        if result is not None:
            st = result.step
            text += f"  {st.source}  cost {st.cost:.3f}  iters {st.iters}  heading {np.degrees(result.heading):.1f} deg"
        else:
            text += "  DROPPED"
        if fps:
            text += f"  {fps:.0f} fps"
        cv2.putText(
            main,
            text,
            (8, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # Tracking window (normalized) and map.
        obs = tr.engine.last_obs
        win = (
            np.clip(128 + 40 * obs, 0, 255).astype(np.uint8)
            if obs is not None
            else np.zeros((8, 8), np.uint8)
        )
        win = cv2.resize(win, (self.panel, self.panel), interpolation=cv2.INTER_NEAREST)
        map_img = tr.engine.map_image()
        map_img = cv2.resize(
            map_img, (self.panel, self.panel // 2), interpolation=cv2.INTER_AREA
        )
        col1 = np.zeros((self.height, self.panel), np.uint8)
        col1[: self.panel] = win
        col1[self.panel : self.panel + self.panel // 2] = map_img
        illum = tr.engine.illumination_image()
        half = self.panel // 2
        if illum is not None and self.height >= self.panel + 2 * half:
            col1[self.panel + half : self.panel + 2 * half, :half] = cv2.resize(
                illum, (half, half), interpolation=cv2.INTER_NEAREST
            )
        col1 = cv2.cvtColor(col1, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            col1, "window", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
        )
        cv2.putText(
            col1,
            "map",
            (6, self.panel + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
        if illum is not None:
            cv2.putText(
                col1,
                "illumination",
                (6, self.panel + half + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )

        # Fictive path (world frame: x north/up, y east/right).
        col2 = np.zeros((self.height, self.panel, 3), np.uint8)
        if result is not None:
            x, y = float(result.values[14]), float(result.values[15])
            self._append_path(x, y)
            b = self.path_bbox
            b[0], b[1], b[2], b[3] = (
                min(b[0], x),
                max(b[1], x),
                min(b[2], y),
                max(b[3], y),
            )
        if self._n_path > 1:
            b = self.path_bbox
            span = max(b[1] - b[0], b[3] - b[2], 1e-6)
            margin = 12
            size = self.panel - 2 * margin
            path = self._path[max(0, self._n_path - 20000) : self._n_path]
            pts = np.empty((len(path), 2), np.int32)
            pts[:, 0] = margin + (path[:, 1] - b[2]) / span * size
            pts[:, 1] = self.panel - margin - (path[:, 0] - b[0]) / span * size
            cv2.polylines(col2, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(col2, tuple(pts[-1]), 3, (0, 0, 255), -1)
            if result is not None:
                h = result.heading
                tip = (
                    int(pts[-1][0] + 14 * np.sin(h)),
                    int(pts[-1][1] - 14 * np.cos(h)),
                )
                cv2.arrowedLine(
                    col2,
                    tuple(pts[-1]),
                    tip,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.4,
                )
            cv2.putText(
                col2,
                f"path {span:.1f} rad span",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
        return np.hstack([main, col1, col2])


class DebugVideoWriter:
    """Write debug canvases to a video file (codec from the config's `vid_codec`).

    H.264, HEVC and VP9 are encoded with PyAV. That is not a preference: the
    `opencv-python` wheels bundle an FFmpeg built without libx264, so `cv2.VideoWriter`
    cannot open an H.264 stream and silently falls back to MPEG-4 Part 2, which browsers
    and most editors refuse to play. The PyAV wheels bundle their own FFmpeg with those
    encoders, so nothing has to be installed system-wide. Frame threading also lets the
    encoder work on other cores while tracking continues.

    The fourcc codecs OpenCV does handle (`mp4v`, `xvid`, `mjpg`, `raw`) still go through
    `cv2.VideoWriter`. Without PyAV, H.264 falls back to it too, and a warning names the
    consequence.
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
