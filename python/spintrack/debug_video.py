"""Annotated debug video: source frame with ball and axes, tracking window, map, path.

Enabled with `save_debug: y` in the config or `spintrack run --debug-video`. Rendering costs
a few milliseconds per frame, so it is off by default.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from spintrack.geometry import normalize
from spintrack.sphere import ball_outline
from spintrack.tracker import FrameResult, Tracker

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


class DebugCanvas:
    """Compose one debug frame from the tracker state."""

    def __init__(self, tracker: Tracker, height: int = 480, panel: int = 240):
        self.tracker = tracker
        self.height = height
        self.panel = panel
        scale = height / tracker.height
        self.scale = scale
        self.main_w = round(tracker.width * scale)
        self.width = self.main_w + panel * 2
        cam = tracker.camera
        self.outline = np.round(
            ball_outline(cam, tracker.centre, tracker.half_angle, 90) * scale
        )
        cx, cy, _ = cam.project(tracker.centre)
        self.centre_px = (float(cx) * scale, float(cy) * scale)
        self.axis_len = 0.8 * np.sin(tracker.half_angle)
        self.path_pts: list[tuple[float, float]] = []
        self.path_bbox = [-0.1, 0.1, -0.1, 0.1]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def _project_axis(self, axis_cam: np.ndarray) -> tuple[int, int]:
        tip = normalize(self.tracker.centre + self.axis_len * axis_cam)
        x, y, _ = self.tracker.camera.project(tip)
        return int(x * self.scale), int(y * self.scale)

    def render(
        self, gray: np.ndarray, result: FrameResult | None, fps: float | None = None
    ):
        tr = self.tracker
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

        # Fictive path (world frame: x north/up, y east/right).
        col2 = np.zeros((self.height, self.panel, 3), np.uint8)
        if result is not None:
            x, y = float(result.values[14]), float(result.values[15])
            self.path_pts.append((x, y))
            b = self.path_bbox
            b[0], b[1], b[2], b[3] = (
                min(b[0], x),
                max(b[1], x),
                min(b[2], y),
                max(b[3], y),
            )
        if len(self.path_pts) > 1:
            b = self.path_bbox
            span = max(b[1] - b[0], b[3] - b[2], 1e-6)
            margin = 12
            size = self.panel - 2 * margin

            def to_px(p):
                px = margin + (p[1] - b[2]) / span * size
                py = self.panel - margin - (p[0] - b[0]) / span * size
                return int(px), int(py)

            pts = np.array([to_px(p) for p in self.path_pts[-20000:]], np.int32)
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
    """Write debug canvases to a video file (codec from the config's `vid_codec`)."""

    def __init__(
        self, path: str | Path, size: tuple[int, int], fps: float, codec: str = "h264"
    ):
        self.path = Path(path)
        fps = fps if fps and fps > 0 else 30.0
        fourcc = FOURCC.get(codec.lower(), "mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if not self._writer.isOpened() and fourcc != "mp4v":
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )
        if not self._writer.isOpened():
            raise OSError(f"could not open debug video writer for {self.path}")

    def write(self, canvas: np.ndarray) -> None:
        self._writer.write(canvas)

    def close(self) -> None:
        self._writer.release()
