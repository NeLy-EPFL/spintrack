"""Interactive calibration window (tkinter, no extra dependencies).

Keys: `c` mark ball rim points (Enter fits the circle), `i` draw an ignore polygon
(Enter closes it), `s` mark the four corners of a calibration square (TL, TR, BR, BL;
Enter solves; `p` cycles the plane xy/yz/xz), `a` set the camera position with sliders,
`u` undo the last point, `w` write the config, `q` quit. The status bar shows the cursor
angle about the ball center (a protractor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spintrack.calibrate.headless import open_config
from spintrack.calibrate.session import CalibrationSession
from spintrack.io.sources import open_source

HELP = (
    "c: rim points  i: ignore polygon  s: square corners  p: plane  a: angles  "
    "u: undo  Enter: confirm  w: write  q: quit"
)


def _to_photo(tk, rgb: np.ndarray):
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode()
    return tk.PhotoImage(
        data=header + np.ascontiguousarray(rgb).tobytes(), format="PPM"
    )


class CalibrationApp:
    def __init__(
        self, session: CalibrationSession, max_width: int = 1400, max_height: int = 900
    ):
        import tkinter as tk

        self.tk = tk
        self.session = session
        h, w = session.frame.shape[:2]
        self.scale = min(1.0, max_width / w, max_height / h)
        self.mode = "circle"
        self.root = tk.Tk()
        self.root.title("spintrack calibration")
        self.canvas = tk.Canvas(
            self.root, width=int(w * self.scale), height=int(h * self.scale)
        )
        self.canvas.pack()
        self.status = tk.StringVar(value=HELP)
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")
        self.sliders = {}
        frame = tk.Frame(self.root)
        frame.pack(fill="x")
        for name, lo, hi, init in (
            ("elevation", -90, 90, 30),
            ("azimuth", -180, 180, 0),
            ("twist", -180, 180, 0),
        ):
            var = tk.DoubleVar(value=init)
            tk.Label(frame, text=name).pack(side="left")
            tk.Scale(
                frame,
                variable=var,
                from_=lo,
                to=hi,
                orient="horizontal",
                resolution=0.5,
                length=260,
                command=lambda _v: self._angles_changed(),
            ).pack(side="left")
            self.sliders[name] = var
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Motion>", self._motion)
        self.root.bind("<Key>", self._key)
        self.root.bind("<Return>", lambda _e: self._confirm())
        self._photo = None
        self.redraw()

    # ----- helpers -----
    def _img_xy(self, event) -> tuple[float, float]:
        return event.x / self.scale, event.y / self.scale

    def redraw(self, message: str | None = None) -> None:
        import cv2

        rgb = self.session.overlay()
        if self.scale != 1.0:
            rgb = cv2.resize(
                rgb, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
            )
        self._photo = _to_photo(self.tk, rgb)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.status.set(f"[{self.mode}] " + (message or HELP))

    # ----- events -----
    def _click(self, event) -> None:
        x, y = self._img_xy(event)
        s = self.session
        if self.mode == "circle":
            s.circle_points.append((x, y))
        elif self.mode == "ignore":
            s.current_polygon.append((x, y))
        elif self.mode == "square" and len(s.square_points) < 4:
            s.square_points.append((x, y))
        self.redraw()

    def _motion(self, event) -> None:
        x, y = self._img_xy(event)
        ang = self.session.cursor_angle(x, y)
        extra = (
            f"  cursor angle about ball center: {ang:+.1f} deg"
            if ang is not None
            else ""
        )
        self.status.set(f"[{self.mode}] {HELP}{extra}")

    def _confirm(self) -> None:
        s = self.session
        if self.mode == "circle":
            ok = s.fit_circle()
            msg = (
                f"ball: center {np.round(s.center, 4)} "
                f"half-angle {np.degrees(s.half_angle):.2f} deg"
                if ok
                else "need 3+ points"
            )
        elif self.mode == "ignore":
            msg = "ignore polygon added" if s.close_polygon() else "need 3+ vertices"
        elif self.mode == "square":
            msg = (
                "camera-to-lab solved from square"
                if s.solve_square()
                else "need exactly 4 corners"
            )
        else:
            msg = HELP
        self.redraw(msg)

    def _angles_changed(self) -> None:
        if self.mode == "angles":
            v = {k: var.get() for k, var in self.sliders.items()}
            self.session.set_angles(v["elevation"], v["azimuth"], v["twist"])
            self.redraw("camera-to-lab from angles")

    def _key(self, event) -> None:
        s = self.session
        k = event.char
        if k == "c":
            self.mode = "circle"
        elif k == "i":
            self.mode = "ignore"
        elif k == "s":
            self.mode = "square"
        elif k == "a":
            self.mode = "angles"
            self._angles_changed()
        elif k == "p":
            planes = ["xy", "yz", "xz"]
            s.square_plane = planes[(planes.index(s.square_plane) + 1) % 3]
            self.redraw(f"square plane: {s.square_plane}")
            return
        elif k == "u":
            if self.mode == "circle" and s.circle_points:
                s.circle_points.pop()
            elif self.mode == "ignore":
                if s.current_polygon:
                    s.current_polygon.pop()
                elif s.ignore_polygons:
                    s.ignore_polygons.pop()
            elif self.mode == "square" and s.square_points:
                s.square_points.pop()
        elif k == "w":
            path = s.save()
            self.redraw(f"wrote {path}")
            return
        elif k == "q":
            self.root.destroy()
            return
        self.redraw()

    def run(self) -> None:
        self.root.mainloop()


def calibrate(config_path: str | Path, src: str | None = None) -> int:
    """Open the calibration window for `config_path` (first frame of its source)."""
    config_path = Path(config_path)
    try:
        cfg = open_config(config_path, src)
        spec = cfg.source(src)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    source = open_source(spec)
    frame = source.read()
    source.close()
    if frame is None:
        raise SystemExit(f"could not read a frame from {spec}")
    session = CalibrationSession(cfg, frame.image, config_path)
    CalibrationApp(session).run()
    return 0
