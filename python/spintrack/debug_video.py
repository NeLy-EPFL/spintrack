"""Annotated debug video: source frame with the ball, its axes and the animal's trail
over it, plus the tracking window, map and fictive path.

The main panel is the source frame with the ball's outline, its orientation axes and the
trail the animal has walked over the surface. The side panels are the tracking window and
the fictive path, then the map and the static illumination field. The map is drawn as an
unfolded dice centered on the face the camera looks at and oriented like the image, with
the ball's top and bottom above and below it (`maps.NET_LABELS`), so at `R = I` its
middle tile is the window.

Enabled with `save_debug: y` in the config or `spintrack run --debug-video`. Rendering
costs a few milliseconds per frame, so it is off by default.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from spintrack.geometry import normalize
from spintrack.maps import NET_LABELS, NET_SHAPE
from spintrack.sphere import ball_outline
from spintrack.tracker import FrameResult, Tracker

log = logging.getLogger(__name__)

AXIS_BGR = (
    (80, 80, 255),
    (80, 220, 80),
    (255, 140, 80),
)  # x red, y green, z blue (BGR)
LABEL_BGR = (255, 255, 0)  # panel titles and face names

# The animal's trail over the ball: how many frames of it to keep, as FicTrac's
# DRAW_SPHERE_HIST_LENGTH.
TRAIL_FRAMES = 1024
# How much of the ball to draw it on: the cosine of the grazing angle a surface point
# must beat. Within a few degrees of the limb the surface is so foreshortened that any
# point of it lands on the outline, so a trail there hugs the outline and says nothing
# about where the animal walked. The cut costs the outermost percent of the disk.
TRAIL_LIMB_COS = 0.1
# Trail color by age, oldest first. FicTrac blends each segment into the image under it;
# a ramp from dark blue to cyan is the same idea without reading the pixels back.
TRAIL_BGR = [
    tuple(bgr)
    for bgr in np.linspace((90, 40, 0), (255, 255, 0), 6).round().astype(int).tolist()
]


def _label(panel: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_BGR, 1)


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
        self.sin_half = np.sin(tracker.half_angle)
        self.axis_len = 0.8 * self.sin_half
        # The animal rides on top of the ball, so the surface point it touches is the
        # lab frame's up in camera coordinates. Without a `c2a_r` in the config this is
        # the identity's up, which points at the camera rather than at the animal.
        self.up_cam = -tracker.cam_to_lab[2]
        self._geometry_version = -1
        self._update_outline()
        self._path = np.empty((1024, 2), np.float64)  # grown by doubling
        self._n_path = 0
        self.path_bbox = [-0.1, 0.1, -0.1, 0.1]
        # Where the animal has touched the ball, in its body frame, oldest first.
        self._trail = np.empty((TRAIL_FRAMES, 3), np.float64)
        self._n_trail = 0
        self._R_cam = None  # last tracked orientation, so a dropped frame still draws

    @property
    def path_pts(self) -> np.ndarray:
        """The integrated path so far, as an (n, 2) array of (x, y)."""
        return self._path[: self._n_path]

    def _append_path(self, x: float, y: float) -> None:
        if self._n_path == len(self._path):
            self._path = np.resize(self._path, (2 * len(self._path), 2))
        self._path[self._n_path] = (x, y)
        self._n_path += 1

    def _append_trail(self, contact_body: np.ndarray) -> None:
        """Keep the last `TRAIL_FRAMES` contact points, oldest first."""
        if self._n_trail == TRAIL_FRAMES:
            self._trail[:-1] = self._trail[1:]
            self._n_trail -= 1
        self._trail[self._n_trail] = contact_body
        self._n_trail += 1

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def _project_axis(self, axis_cam: np.ndarray) -> tuple[int, int]:
        tip = normalize(self.tracker.center + self.axis_len * axis_cam)
        x, y, _ = self.tracker.camera.project(tip)
        return int(x * self.scale), int(y * self.scale)

    def _update_outline(self) -> None:
        """Re-project the ball outline; the window may have moved onto a moved ball."""
        tracker, scale = self.tracker, self.scale
        self._geometry_version = tracker.geometry_version
        self.outline = np.round(
            ball_outline(tracker.camera, tracker.center, tracker.half_angle, 90) * scale
        )
        cx, cy, _ = tracker.camera.project(tracker.center)
        self.center_px = (float(cx) * scale, float(cy) * scale)

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
        self._draw_trail(main, result)
        c = (int(self.center_px[0]), int(self.center_px[1]))
        if result is not None:
            # Ball orientation: a gnomon that rotates with the ball (camera frame).
            for i, color in enumerate(AXIS_BGR):
                tip = self._project_axis(result.R_cam[:, i])
                cv2.arrowedLine(main, c, tip, color, 2, cv2.LINE_AA, tipLength=0.2)
        # Lab axes at the ball center (fixed): thin lines with labels.
        for i, color in enumerate(AXIS_BGR):
            tip = self._project_axis(tr.cam_to_lab.T[:, i])
            cv2.line(main, c, tip, color, 1, cv2.LINE_AA)
            cv2.putText(
                main,
                "xyz"[i],
                (tip[0] + 3, tip[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        text = f"frame {tr.frame - 1}"
        if result is not None:
            st = result.step
            text += (
                f"  {st.source}  cost {st.cost:.3f}  iters {st.iters}"
                f"  heading {np.degrees(result.heading):.1f} deg"
            )
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

        # Side panels, two rows: the tracking window (normalized) and the fictive path
        # on top, the map as an unfolded dice and the static illumination field below.
        panel = self.panel
        side = np.zeros((self.height, 2 * panel, 3), np.uint8)
        obs = tr.engine.last_obs
        win = (
            np.clip(128 + 40 * obs, 0, 255).astype(np.uint8)
            if obs is not None
            else np.zeros((8, 8), np.uint8)
        )
        box = min(panel, self.height)
        win = cv2.resize(win, (box, box), interpolation=cv2.INTER_NEAREST)
        side[:box, :box] = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        _label(side, "window", 6, 16)
        self._draw_path(side, panel, 0, box, result)
        # The net takes what height is left, up to what leaves the illumination field
        # half a panel of width.
        top = box
        tile = min((self.height - top) // 3, (2 * panel - panel // 2) // 4)
        if tile >= 16:
            self._draw_net(side, top, tile)
            illum = tr.engine.illumination_image()
            if illum is not None:
                x0 = 4 * tile
                s = min(2 * panel - x0, self.height - top)
                illum = cv2.resize(illum, (s, s), interpolation=cv2.INTER_NEAREST)
                side[top : top + s, x0 : x0 + s] = cv2.cvtColor(
                    illum, cv2.COLOR_GRAY2BGR
                )
                _label(side, "illumination", x0 + 6, top + 16)
        return np.hstack([main, side])

    def trail_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the animal's past contact points sit now: panel coordinates (n, 2),
        oldest first, and which of them the camera can see."""
        tr = self.tracker
        # Each stored point is body-fixed, so the ball's current orientation says where
        # the surface carried it, and the ball's radius puts it back on the surface.
        contact = self._trail[: self._n_trail] @ self._R_cam.T
        # A surface point faces the camera when `-(c . u)` beats the ball's radius over
        # its distance, which is a little short of a hemisphere; `TRAIL_LIMB_COS` more
        # asks it to face the camera squarely enough to be drawn where it really is.
        seen = contact @ tr.center < -self.sin_half - TRAIL_LIMB_COS
        x, y, inside = tr.camera.project(tr.center + self.sin_half * contact)
        seen &= inside
        pts = np.stack([np.where(seen, x, 0.0), np.where(seen, y, 0.0)], axis=1)
        return pts * self.scale, seen

    def _draw_trail(self, main: np.ndarray, result: FrameResult | None) -> None:
        """Draw the trail the animal has walked over the ball, as FicTrac does.

        The animal stays put while the ball turns under it, so the surface point it
        touched at frame `i` is now `R_cam @ R_cam(i).T @ up`, and the trail is that
        point for every frame still in the history - the animal's path, inverted,
        painted on the ball it walked. Only the near side of the ball is drawn, and the
        trail brightens with recency.
        """
        if result is not None:
            self._R_cam = result.R_cam
            self._append_trail(result.R_cam.T @ self.up_cam)
        if self._R_cam is None or self._n_trail < 2:
            return
        pts, seen = self.trail_points()
        pts = pts.astype(np.int32)
        bands = len(TRAIL_BGR)
        for band in range(bands):
            lo = band * self._n_trail // bands
            hi = (band + 1) * self._n_trail // bands + 1  # overlap, to join the bands
            edges = np.flatnonzero(np.diff(np.r_[False, seen[lo:hi], False]))
            runs = [
                pts[lo + i : lo + j]
                for i, j in zip(edges[::2], edges[1::2])
                if j - i > 1
            ]
            cv2.polylines(main, runs, False, TRAIL_BGR[band], 1, cv2.LINE_AA)
        if seen[-1]:
            cv2.circle(main, tuple(pts[-1]), 3, TRAIL_BGR[-1], -1, cv2.LINE_AA)
        _label(main, "path on ball", 8, 16)

    def _draw_net(self, side: np.ndarray, top: int, tile: int) -> None:
        """Draw the map as an unfolded dice of `tile`-pixel faces at the left of `side`,
        from row `top` down.

        The net is centered on the face the camera looks at and oriented like the image
        (see `maps.NET_LABELS`), so at `R = I` its middle tile is the tracking window.
        The six unused tiles of the 4x3 net stay black.
        """
        rows, cols = NET_SHAPE
        w, h = cols * tile, rows * tile
        net = self.tracker.engine.map_image("cube")
        interp = cv2.INTER_AREA if net.shape[1] > w else cv2.INTER_LINEAR
        net = cv2.resize(net, (w, h), interpolation=interp)
        net = cv2.cvtColor(net, cv2.COLOR_GRAY2BGR)
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in NET_LABELS:
                    net[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = 0
        for (r, c), name in NET_LABELS.items():
            x0, y0 = c * tile, r * tile
            cv2.rectangle(
                net, (x0, y0), (x0 + tile - 1, y0 + tile - 1), (70, 70, 70), 1
            )
            cv2.putText(
                net,
                name,
                (x0 + 3, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                LABEL_BGR,
                1,
            )
        side[top : top + h, :w] = net
        _label(side, "map", 6, top + 16)

    def _draw_path(
        self, side: np.ndarray, x0: int, y0: int, box: int, result: FrameResult | None
    ) -> None:
        """Draw the fictive path (world frame: x north/up, y east/right) in the `box`
        pixel square at `(x0, y0)` of `side`."""
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
        if self._n_path < 2 or box < 60:
            return
        b = self.path_bbox
        span = max(b[1] - b[0], b[3] - b[2], 1e-6)
        margin = 12
        size = box - 2 * margin
        path = self._path[max(0, self._n_path - 20000) : self._n_path]
        pts = np.empty((len(path), 2), np.int32)
        pts[:, 0] = x0 + margin + (path[:, 1] - b[2]) / span * size
        pts[:, 1] = y0 + box - margin - (path[:, 0] - b[0]) / span * size
        cv2.polylines(side, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(side, tuple(pts[-1]), 3, (0, 0, 255), -1)
        if result is not None:
            h = result.heading
            tip = (
                int(pts[-1][0] + 14 * np.sin(h)),
                int(pts[-1][1] - 14 * np.cos(h)),
            )
            cv2.arrowedLine(
                side, tuple(pts[-1]), tip, (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.4
            )
        _label(side, f"path {span:.1f} rad span", x0 + 6, y0 + 16)


class DebugVideoWriter:
    """Write debug canvases to a video file (codec from the config's `vid_codec`).

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
