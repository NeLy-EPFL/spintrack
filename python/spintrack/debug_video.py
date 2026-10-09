"""Annotated debug video.

The left panel is the source frame with the ball's outline, the trail the animal has
walked over its surface, and (unless `output.debug_axes = false`) the ball's axes from
its center, turning with it, and the animal's from where it stands on the ball: x
forward, y left, z the ball's normal. On the right are the normalized tracking window
and the fictive path, and below them the surface map as an unfolded dice centered on
the face the camera looks at and oriented like the image (`maps.NET_LABELS`), so at
`R = I` its middle tile is the window. Two of the net's empty tiles hold the static
lighting field and the frame's numbers, and a third, when SAM 3 found the ball or
placed the camera, its masks over the first frame: the ball's (yellow, as its
outline), found on the opening frames' 90th percentile, and the animal's (magenta), on
one opening frame. They are found once, so the tile does not change.

Enabled with `debug_video = true` in the config's `[output]` or `spintrack run
--debug-video`. The config's `mask.ignore` regions are outlined in red.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spintrack.maps import NET_LABELS, NET_SHAPE
from spintrack.scene import TRAIL_FRAMES, Scene
from spintrack.tracker import FrameResult, Tracker

if TYPE_CHECKING:
    import av

log = logging.getLogger(__name__)

AXIS_BGR = ((80, 80, 255), (80, 220, 80), (255, 140, 80))  # x red, y green, z blue
TEXT_BGR = (255, 255, 255)
BALL_BGR = (0, 220, 255)  # yellow: its outline, and its mask
ANIMAL_BGR = (255, 80, 255)  # magenta: its mask
IGNORE_BGR = (60, 60, 200)
PATH_BGR = (0, 255, 255)
HEAD_BGR = (0, 0, 255)
EDGE_BGR = (25, 25, 25)  # under arrows and dots, so they read on any background
SHIFT = 4  # sub-pixel bits for OpenCV's drawing
ANIMAL_ARROW = 40  # the animal's axes' length, as for a 480-row canvas
# An axis imaged shorter than this, of `ANIMAL_ARROW`, is along the line of sight
# (within 24 deg), and drawn as a ring.
ALONG_SIGHT = 0.4
# SAM 3's masks, in `Prepared.masks`' order.
MASK_BGR = (BALL_BGR, ANIMAL_BGR)
MASK_ALPHA = 0.35
MASK_MARGIN = 0.08  # around the masks in their tile, of the larger side

__all__ = ["TRAIL_FRAMES", "DebugCanvas", "DebugVideoWriter"]

# Trail color by age, oldest first: dark blue to cyan.
TRAIL_BGR = [
    tuple(bgr)
    for bgr in np.linspace((160, 60, 0), (255, 255, 0), 6).round().astype(int).tolist()
]


FOURCC = {
    "h264": "avc1",
    "avc1": "avc1",
    "mp4v": "mp4v",
    "xvid": "XVID",
    "mjpg": "MJPG",
    "raw": "I420",
}

# `debug_codec` -> (PyAV encoder, encoder options) for the codecs OpenCV cannot write.
AV_CODEC: dict[str, tuple[str, dict[str, object]]] = {
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


@lru_cache(maxsize=32)
def _font(px: int) -> ImageFont.FreeTypeFont:
    """Pillow's default font at `px` pixels to the em, kept because loading costs."""
    font = ImageFont.load_default(size=max(px, 1))
    # Given a size, Pillow's default font is its bundled FreeType one.
    assert isinstance(font, ImageFont.FreeTypeFont)
    return font


@lru_cache(maxsize=1024)
def _char(ch: str, px: int) -> tuple[np.ndarray, int, int, float]:
    """A character's FreeType coverage in `[0, 1]`, its offset, and the pen's advance.

    The offset is its top left from the pen on the baseline.
    """
    font = _font(px)
    left, top, right, bottom = font.getbbox(ch, anchor="ls")
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    image = Image.new("L", (max(right - left, 0), max(bottom - top, 0)))
    ImageDraw.Draw(image).text((-left, -top), ch, 255, font, "ls")
    return np.asarray(image, np.float32) / 255, left, top, font.getlength(ch)


@lru_cache(maxsize=1024)
def _glyphs(text: str, px: int, border: int, anchor: str):
    """`text`'s coverage in `[0, 1]` and that of a halo, or None for blank text.

    The halo is `border` pixels about it; both are `(H, W, 1)`, returned with their top
    left from the anchor point.

    `anchor` is two of Pillow's anchor letters, `l`, `m` or `r` across and `s` or `m`
    down, except that `m` centers the ink. The text is laid out from cached
    characters, without kerning, since the frame's numbers change every frame: Pillow
    takes about 1 ms to draw a stroked string.
    """
    placed, pen = [], 0.0
    for ch in text:
        ink, left, top, advance = _char(ch, px)
        if ink.size:
            placed.append((ink, round(pen) + left, top))
        pen += advance
    if not placed:
        return None
    x0 = min(x for _, x, _ in placed) - border
    y0 = min(y for _, _, y in placed) - border
    x1 = max(x + ink.shape[1] for ink, x, _ in placed) + border
    y1 = max(y + ink.shape[0] for ink, _, y in placed) + border
    cover = np.zeros((y1 - y0, x1 - x0), np.float32)
    for ink, x, y in placed:
        region = cover[y - y0 : y - y0 + ink.shape[0], x - x0 : x - x0 + ink.shape[1]]
        np.maximum(region, ink, out=region)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1,) * 2)
    halo = cv2.dilate(cover, disk)
    dx = {"l": 0, "m": -(x0 + x1) / 2, "r": -pen}[anchor[0]]
    dy = {"s": 0, "m": -(y0 + y1) / 2}[anchor[1]]
    return halo[..., None], cover[..., None], x0 + round(dx), y0 + round(dy)


def _arrow(image: np.ndarray, start, end, color, width: int, head: float) -> None:
    """An arrow with a filled head `head` pixels long, at most 40% of the arrow.

    It is edged in dark so that it reads on any background.
    """
    p0, p1 = np.asarray(start, np.float64), np.asarray(end, np.float64)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1.0:
        return
    along = (p1 - p0) / length
    head = min(head, 0.4 * length)
    base = p1 - head * along
    side = 0.5 * head * np.array([-along[1], along[0]])
    k = 1 << SHIFT
    tip = np.round(np.array([p1, base + side, base - side]) * k).astype(np.int32)
    # The shaft runs into the head, so that no seam shows between them.
    a, b = (tuple(np.round(p * k).astype(int)) for p in (p0, base + 0.2 * head * along))
    for c, edge in ((EDGE_BGR, 2), (color, 0)):
        cv2.line(image, a, b, c, width + edge, cv2.LINE_AA, SHIFT)
        if edge:
            cv2.polylines(image, [tip], True, c, edge, cv2.LINE_AA, SHIFT)
        cv2.fillPoly(image, [tip], c, cv2.LINE_AA, SHIFT)


class DebugCanvas:
    """Compose one debug frame from the tracker state.

    The right-hand column is five tiles tall and four wide: the window and the path take
    two by two each, the map net the 3x4 below them.
    """

    def __init__(
        self,
        tracker: Tracker,
        height: int = 960,
        axes: bool = False,
        masks: Sequence[tuple[str, np.ndarray, float | None]] = (),
    ):
        """`masks` are SAM 3's, as `Prepared.masks` gives them."""
        self.tracker = tracker
        self.axes = axes
        # `height` is a hint: snap it to an integer multiple of a smaller source, and to
        # an exact integer decimation of a larger one when one is within reach, because
        # INTER_AREA is ~50x faster there.
        if tracker.height < height:
            height = tracker.height * round(height / tracker.height)
        else:
            k = round(tracker.height / height)
            if tracker.height % k == 0 and tracker.width % k == 0:
                height = tracker.height // k
        self.height = height
        # Text, offsets and line widths are sized for 480 rows and scaled from there.
        self.u = max(1.0, height / 480)
        self.lw = max(1, round(self.u))
        self.scale = height / tracker.height
        self.main_w = round(tracker.width * self.scale)
        self.tile = height // 5
        self.width = self.main_w + 4 * self.tile
        self.ignore = [
            np.round(np.reshape(poly, (-1, 2)) * self.scale).astype(np.int32)
            for poly in tracker.cfg.mask.ignore
        ]
        self.masks = [m for m in masks if m[1].any()]
        self._mask_tile = None
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
        interp = cv2.INTER_AREA if self.scale <= 1 else cv2.INTER_LINEAR
        main = cv2.resize(gray, (self.main_w, self.height), interpolation=interp)
        main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
        lw = self.lw
        for c, extra in ((EDGE_BGR, 2), (BALL_BGR, 0)):
            cv2.polylines(main, [self.outline], True, c, lw + extra, cv2.LINE_AA)
        cv2.polylines(main, self.ignore, True, IGNORE_BGR, lw, cv2.LINE_AA)
        self._draw_trail(main)
        if self.axes:
            self._draw_axes(main, result)
        side = np.zeros((self.height, 4 * t, 3), np.uint8)
        self._draw_window(side[: 2 * t, : 2 * t])
        self._draw_path(side[: 2 * t, 2 * t :], result)
        net = side[2 * t : 5 * t]
        self._draw_net(net)
        self._draw_lighting(net[:t, 3 * t :])
        if self.masks:
            if self._mask_tile is None:
                self._mask_tile = self._masks_tile(gray, t)
            net[:t, 2 * t : 3 * t] = self._mask_tile
        self._draw_status(net[2 * t :, 2 * t :], result, fps)
        return np.hstack([main, side])

    def _text(
        self, panel, text: str, org, size: float, color, anchor: str = "ls"
    ) -> None:
        """`text` at pixel `org`, over a dark halo to read on any background.

        `size` pixels to the em as for a 480-row canvas, placed as `anchor` says
        (`_glyphs`; by default, `org` is its baseline's left).
        """
        found = _glyphs(text, round(size * self.u), self.lw, anchor)
        if found is None:
            return
        border, ink, dx, dy = found
        gx, gy = round(org[0]) + dx, round(org[1]) + dy
        x0, y0 = max(gx, 0), max(gy, 0)
        x1 = min(gx + ink.shape[1], panel.shape[1])
        y1 = min(gy + ink.shape[0], panel.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        crop = np.s_[y0 - gy : y1 - gy, x0 - gx : x1 - gx]
        under = panel[y0:y1, x0:x1].astype(np.float32) * (1 - border[crop])
        under += ink[crop] * (np.float32(color) - under)
        panel[y0:y1, x0:x1] = np.rint(under).astype(np.uint8)

    def _label(self, panel: np.ndarray, text: str) -> None:
        """A panel's name, at its top left."""
        u = self.u
        self._text(panel, text, (round(5 * u), round(16 * u)), 14, TEXT_BGR)

    def _masks_tile(self, gray: np.ndarray, t: int) -> np.ndarray:
        """SAM 3's masks over `gray`, cropped to them, in a `t` x `t` tile."""
        h, w = gray.shape
        boxes = np.array(
            [cv2.boundingRect(m.astype(np.uint8)) for _, m, _ in self.masks]
        )
        (x0, y0), (x1, y1) = boxes[:, :2].min(0), (boxes[:, :2] + boxes[:, 2:]).max(0)
        pad = MASK_MARGIN * max(x1 - x0, y1 - y0)
        x0, y0 = max(int(x0 - pad), 0), max(int(y0 - pad), 0)
        x1, y1 = min(int(x1 + pad), w), min(int(y1 + pad), h)
        k = t / max(x1 - x0, y1 - y0)
        size = (max(round(k * (x1 - x0)), 1), max(round(k * (y1 - y0)), 1))
        crop = np.s_[y0:y1, x0:x1]
        image = cv2.resize(gray[crop], size, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (_, mask, _), color in zip(self.masks, MASK_BGR, strict=False):
            small = mask[crop].astype(np.float32)
            inside = cv2.resize(small, size, interpolation=cv2.INTER_AREA) >= 0.5
            fill = np.full_like(image, color)
            tinted = cv2.addWeighted(image, 1 - MASK_ALPHA, fill, MASK_ALPHA, 0)
            image[inside] = tinted[inside]
            contours, _ = cv2.findContours(
                inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c, extra in ((EDGE_BGR, 2), (color, 0)):
                cv2.drawContours(image, contours, -1, c, self.lw + extra, cv2.LINE_AA)
        tile = np.zeros((t, t, 3), np.uint8)
        ox, oy = (t - size[0]) // 2, (t - size[1]) // 2
        tile[oy : oy + size[1], ox : ox + size[0]] = image
        self._label(tile, "segmentation")
        # The legend: a swatch of the mask's color, then its name and score.
        u, n = self.u, len(self.masks)
        side, step = round(7 * u), round(14 * u)
        for i, ((name, _, score), color) in enumerate(
            zip(self.masks, MASK_BGR, strict=False)
        ):
            y = t - round(5 * u) - step * (n - 1 - i)
            x = round(4 * u)
            cv2.rectangle(tile, (x, y - side), (x + side, y), color, -1)
            cv2.rectangle(tile, (x, y - side), (x + side, y), (0, 0, 0), 1)
            org = (x + side + round(4 * u), y)
            label = name if score is None else f"{name} {score:.2f}"
            self._text(tile, label, org, 13, TEXT_BGR)
        return tile

    def _draw_axes(self, main: np.ndarray, result: FrameResult | None) -> None:
        """The ball's axes and the animal's, labeled.

        The ball's turn with it from its center; the animal's start where it stands.
        """
        axes, scale, u, lw = self.scene.axes(), self.scale, self.u, self.lw
        center = np.asarray(self.scene.center_px) * scale
        if result is not None and axes["ball"] is not None:
            for tip, color in zip(axes["ball"], AXIS_BGR, strict=True):
                _arrow(main, center, np.asarray(tip) * scale, color, lw, 7 * u)
        contact = np.asarray(axes["contact"]) * scale
        steps = np.asarray(axes["animal"]) * scale - contact
        # The animal's axes are `ANIMAL_ARROW` long across the line of sight, whatever
        # the ball's size: imaged, orthonormal axes' squared lengths sum to twice that.
        full = ANIMAL_ARROW * u
        steps *= full / max(np.sqrt((steps**2).sum() / 2), 1e-9)
        dot = tuple(np.round(contact).astype(int))
        for i, (step, color) in enumerate(zip(steps, AXIS_BGR, strict=True)):
            length = np.hypot(*step)
            if length >= ALONG_SIGHT * full:
                _arrow(main, contact, contact + step, color, lw + 1, 9 * u)
                beyond = contact + step * (1 + 9 * u / length)
                self._text(main, "xyz"[i], beyond, 15, color, "mm")
                continue
            # Along the line of sight (one axis at most), a ring: dotted when the axis
            # points toward the camera, crossed when away from it (into the image).
            ring, r = round(8 * u), round(3 * u)
            away = self.tracker.cam_to_lab[i] @ self.tracker.center > 0
            for c, extra in ((EDGE_BGR, 2), (color, 0)):
                cv2.circle(main, dot, ring, c, lw + extra, cv2.LINE_AA)
                if not away:
                    cv2.circle(main, dot, r + extra // 2, c, -1, cv2.LINE_AA)
                    continue
                for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    a = np.array(dot) + 0.71 * ring * np.array([sx, sy])
                    p, q = dot, tuple(np.round(a).astype(int))
                    cv2.line(main, p, q, c, lw + extra, cv2.LINE_AA)
            org = (dot[0] + ring + 4 * u, dot[1])
            self._text(main, "xyz"[i], org, 15, color, "lm")

    def _draw_trail(self, main: np.ndarray) -> None:
        """Draw the trail the animal has walked over the ball, as FicTrac does.

        Only the near side, brighter with recency (`Scene.trail_points`).
        """
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
                for i, j in zip(edges[::2], edges[1::2], strict=True)
                if j - i > 1
            ]
            cv2.polylines(main, runs, False, TRAIL_BGR[band], self.lw, cv2.LINE_AA)
        if seen[-1]:
            dot = round(3 * self.u)
            cv2.circle(main, tuple(pts[-1]), dot, TRAIL_BGR[-1], -1, cv2.LINE_AA)

    def _draw_window(self, panel: np.ndarray) -> None:
        obs = self.tracker.engine.last_obs
        if obs is not None:
            win = np.clip(128 + 40 * obs, 0, 255).astype(np.uint8)
            win = cv2.resize(win, panel.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            panel[:] = win[..., None]
        self._label(panel, "window")

    def _draw_path(self, panel: np.ndarray, result: FrameResult | None) -> None:
        """The fictive path (x up, y left), scaled to fit the panel."""
        h, w = panel.shape[:2]
        u, lw = self.u, self.lw
        margin = round(12 * u)
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
            cv2.polylines(panel, [pts], False, PATH_BGR, lw, cv2.LINE_AA)
            head = tuple(int(v) for v in pts[-1])
            cv2.circle(panel, head, round(3 * u), HEAD_BGR, -1, cv2.LINE_AA)
            if result is not None:
                hd, n = result.heading, 14 * u
                tip = (int(head[0] - n * np.sin(hd)), int(head[1] - n * np.cos(hd)))
                cv2.arrowedLine(panel, head, tip, HEAD_BGR, lw, cv2.LINE_AA, 0, 0.4)
            text = f"{span:.2g} r"
            org = (w - 6 * u, h - 6 * u)
            self._text(panel, text, org, 13, TEXT_BGR, "rs")
        self._label(panel, "path")

    def _draw_net(self, panel: np.ndarray) -> None:
        """The map as an unfolded dice filling `panel` (3 x 4 tiles).

        The six unused tiles stay black.
        """
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
            org = (round(4 * self.u), round(13 * self.u))
            self._text(tile, name, org, 12, TEXT_BGR)
        self._label(panel[:t, :t], "map")

    def _draw_lighting(self, panel: np.ndarray) -> None:
        """The lighting gain the tracker divides out of the window (dark: shadow)."""
        illum = self.tracker.engine.illumination_image()
        if illum is None:
            return
        s = min(panel.shape[:2])
        panel[:s, :s] = cv2.resize(illum, (s, s), interpolation=cv2.INTER_AREA)[
            ..., None
        ]
        self._label(panel, "lighting")

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
        u = self.u
        step = max(12 * u, min(18 * u, panel.shape[0] / (len(lines) + 1)))
        for i, line in enumerate(lines):
            org = (round(8 * u), round(6 * u + step * (i + 1)))
            self._text(panel, line, org, 14, TEXT_BGR)


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
        self._container: av.container.OutputContainer | None = None
        self._stream: av.VideoStream | None = None
        self._writer: cv2.VideoWriter | None = None
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
            assert isinstance(stream, av.VideoStream)  # a video encoder's stream
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
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*fourcc), self.fps, self.size
        )
        if not writer.isOpened() and fourcc != "mp4v":
            log.warning(
                "OpenCV cannot encode %s (its bundled FFmpeg has no such encoder); "
                "writing %s as MPEG-4 Part 2, which browsers and most editors cannot "
                "play.",
                self.codec,
                self.path,
            )
            writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, self.size
            )
        if not writer.isOpened():
            raise OSError(f"could not open debug video writer for {self.path}")
        self._writer = writer

    # ----- writing -----
    def write(self, canvas: np.ndarray) -> None:
        if self._container is None or self._stream is None:
            assert self._writer is not None  # opened by `_open_opencv`
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
        container, stream = self._container, self._stream
        if container is not None and stream is not None:
            self._container = self._stream = None
            try:
                container.mux(stream.encode())  # flush the encoder's buffered frames
            finally:
                container.close()
        elif self._writer is not None:
            self._writer.release()
            self._writer = None
