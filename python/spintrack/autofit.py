"""Fill in the geometry a config leaves open, from the recording itself.

`prepare_config` detects the ball when the config does not describe one (and compares
the two when it does), and fits `vfov` when it is `auto`. It needs a source it can read
before tracking begins, so a live camera is refused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from dataclasses import replace as _replace

import numpy as np

from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.detect import (
    BallDetection,
    DetectionError,
    circle_points,
    detect_ball,
    fit_circle,
    sample_frames,
)
from spintrack.sphere import fit_ball, pixel_circle

log = logging.getLogger("spintrack")

CAMERA_SOURCE_MESSAGE = (
    "automatic geometry needs a seekable source: it looks at frames before tracking "
    "starts. Record a short clip and run `spintrack calibrate CLIP.txt --auto`, then "
    "use the config it writes."
)
# A detected radius this far from the config's is worth saying out loud: a relative
# radius error costs about twice as much of every in-plane rotation reported.
DISAGREEMENT_WARN = 0.03

VFOV_GRID = (1.0, 120.0, 9)  # log-spaced candidates, degrees
VFOV_REFINE = 6  # candidates between the best grid point's neighbors
MAX_HALF_ANGLE_DEG = 60.0
WARMUP_FRAMES = 20  # the map is still filling; those costs say nothing about the vfov
# The cost separates fields of view only once a new view is matched against surface
# seen under a different rotation, so the map has to fill first.
VFOV_FRAMES = 1000
# A minimum is believed only when its nearer neighbor costs this much more; a curve
# whose neighbors jump by more than that either way has noise, not a minimum.
MIN_DEPTH = 1.02
# Candidates within this factor of the minimum count as indistinguishable from it.
FLAT_TOL = 1.05
# Below this much accumulated rotation the recording says nothing about the geometry.
MIN_TURN_DEG = 90.0


@dataclass
class VfovFit:
    """The field of view the photometric cost prefers, and how much it prefers it."""

    vfov: float
    identifiable: bool
    spread: float  # max cost / min cost over the grid
    flat_range: tuple[float, float] | None  # vfovs whose cost is within FLAT_TOL
    turned_deg: float = 0.0  # rotation the ball showed while the curve was measured
    depth: float = 1.0  # cost at the minimum's nearer neighbor, over the minimum
    curve: list[tuple[float, float]] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "value": self.vfov,
            "source": "auto",
            "identifiable": self.identifiable,
            "spread": self.spread,
            "flat_range": list(self.flat_range) if self.flat_range else None,
            "turned_deg": self.turned_deg,
            "depth": self.depth,
            "curve": [list(point) for point in self.curve],
        }

    def line(self) -> str:
        if self.identifiable:
            return (
                f"{self.vfov:.4g} deg (fitted; the cost minimum is {self.depth:.2g}x "
                f"below its neighbors)"
            )
        lo, hi = self.flat_range or (float("nan"), float("nan"))
        return (
            f"{self.vfov:.4g} deg (fitted; not identifiable, the cost is flat over "
            f"{lo:.3g}-{hi:.3g} deg, and the rotation scale does not depend on it)"
        )


@dataclass
class Prepared:
    """What automatic preparation found, for the run summary and the sidecar."""

    detection: BallDetection | None = None
    ball_source: str = "config"  # "config" or "detected"
    config_radius_px: float | None = None
    radius_disagreement: float | None = None  # detected / config - 1
    vfov: VfovFit | None = None
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        """The `ball` entry of the sidecar's provenance."""
        out: dict = {"source": self.ball_source}
        if self.detection is not None:
            out.update(
                center_px=[self.detection.cx, self.detection.cy],
                radius_px=self.detection.r,
                confidence=self.detection.confidence,
                rim_fraction=self.detection.rim_fraction,
                residual_px=self.detection.residual_px,
                n_frames=self.detection.n_frames,
            )
        if self.config_radius_px is not None:
            out["config_radius_px"] = self.config_radius_px
        if self.radius_disagreement is not None:
            out["radius_disagreement"] = self.radius_disagreement
        return out

    def line(self) -> str:
        """One line for the terminal block."""
        if self.detection is None:
            return "from config (detection failed, not checked)"
        d = self.detection
        where = f"({d.cx:.1f}, {d.cy:.1f}) r {d.r:.1f} px"
        if self.ball_source == "detected":
            return f"detected at {where}, confidence {d.confidence:.2f}"
        if self.radius_disagreement is None:
            return f"from config; detection says {where}"
        return (
            f"from config; detection says {where}, "
            f"{100 * self.radius_disagreement:+.1f}% on the radius"
        )


def config_circle(cfg: Config, width: int, height: int) -> tuple[float, float, float]:
    """The ball's image circle as the config describes it; raises if it has none."""
    if len(cfg.roi_circ) >= 6:
        return fit_circle(cfg.roi_circ)
    if cfg.roi_c is not None and cfg.roi_r is not None:
        if cfg.vfov is None:
            raise ValueError(
                "roi_c/roi_r describe the ball in camera directions, which need a vfov "
                "to become pixels; give a vfov or use roi_circ"
            )
        camera = source_camera(width, height, cfg.vfov, cfg.fisheye)
        return pixel_circle(camera, cfg.roi_c, float(cfg.roi_r))
    raise ValueError("config defines no ball")


def _set_ball_from_points(cfg: Config, width: int, height: int) -> float:
    """Fill `roi_c`/`roi_r` from `roi_circ` at the config's current `vfov`."""
    camera = source_camera(width, height, cfg.vfov, cfg.fisheye)
    points = np.asarray(cfg.roi_circ, dtype=np.float64).reshape(-1, 2)
    center, half_angle = fit_ball(points, camera)
    cfg.roi_c = [float(v) for v in center]
    cfg.roi_r = float(half_angle)
    return half_angle


def prepare_config(
    cfg: Config,
    src_spec,
    *,
    n_frames: int = 100,
    span: int = 300,
    vfov_frames: int = VFOV_FRAMES,
    params=None,
) -> Prepared:
    """Detect the ball, fit `vfov` if it is `auto`, and report what was found.

    The detection is used only when the config has no ball of its own; when it has one,
    the two are compared and the config's is kept, and a failed detection is only
    reported. `vfov` is fitted with the pixel circle held fixed, since the two together
    set the ball's angular radius and the cost cannot separate them. Mutates `cfg`.
    """
    if str(src_spec).isdigit():
        raise ValueError(CAMERA_SOURCE_MESSAGE)
    frames = sample_frames(str(src_spec), n_frames, span)
    height, width = frames[0].shape
    prepared = Prepared()
    try:
        prepared.detection = detect_ball(frames, max_frames=n_frames)
    except DetectionError as exc:
        if not cfg.has_ball():
            raise
        log.warning("ball detection failed, keeping the config's ball: %s", exc)
    detection = prepared.detection

    if cfg.has_ball():
        try:
            radius = config_circle(cfg, width, height)[2]
        except ValueError:  # roi_c/roi_r with vfov auto: no pixel circle yet
            radius = None
        if radius is not None and detection is not None:
            prepared.config_radius_px = radius
            prepared.radius_disagreement = detection.r / radius - 1.0
            if abs(prepared.radius_disagreement) > DISAGREEMENT_WARN:
                message = (
                    f"detected ball radius {detection.r:.1f} px differs from the "
                    f"config's {radius:.1f} px by "
                    f"{100 * prepared.radius_disagreement:+.1f}%; that is about twice "
                    f"as much again on every in-plane rotation reported"
                )
                log.warning("%s", message)
                prepared.notes.append(message)
    else:
        prepared.ball_source = "detected"
        cfg.roi_circ = detection.rim_points(16)
        if cfg.vfov is not None:
            half_angle = _set_ball_from_points(cfg, width, height)
            prepared.notes.append(
                f"ball fitted from the recording: half-angle "
                f"{np.degrees(half_angle):.4f} deg at confidence "
                f"{detection.confidence:.2f}"
            )

    if cfg.vfov is None:
        if cfg.roi_circ:
            circle = fit_circle(cfg.roi_circ)
        elif detection is not None:
            circle = detection
        else:
            raise ValueError(
                "vfov is auto and the ball has no pixel circle to fit it from: give "
                "roi_circ, or a vfov for roi_c/roi_r"
            )
        prepared.vfov = fit_vfov(
            src_spec, cfg, circle, n_frames=vfov_frames, params=params
        )
        cfg.vfov = prepared.vfov.vfov
        if not cfg.roi_circ:
            cfg.roi_circ = _circle_points(circle)
        _set_ball_from_points(cfg, width, height)
    return prepared


def _circle_points(circle, n: int = 16) -> list[int]:
    """`roi_circ` points from a `BallDetection` or a plain `(cx, cy, r)`."""
    if isinstance(circle, BallDetection):
        return circle.rim_points(n)
    return circle_points(*circle, n)


def _costs_at(
    src_spec, cfg: Config, points, vfovs, n_frames: int, params
) -> tuple[np.ndarray, np.ndarray]:
    """Median cost and turned angle (deg) of tracking the first frames at each `vfov`.

    All candidates track in lockstep, so the video is decoded once. A candidate whose
    geometry is impossible (the rim points do not describe a ball there, or the ball
    would cover most of the sky) or whose tracker fails scores NaN.
    """
    from spintrack.engine import TrackParams
    from spintrack.io.sources import VideoSource
    from spintrack.tracker import Tracker

    # This search lives on the cost's sensitivity to the geometry, so what explains
    # away part of a wrong field of view's cost is off: the static illumination field
    # absorbs a systematic misregistration as shading, and the pre-filter blurs it.
    params = _replace(
        params or TrackParams(),
        prefilter=0.0,
        illum_bias=False,
        center_watch=False,
    )
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    source = VideoSource(src_spec)
    trackers: list = []
    costs: list[list[float]] = [[] for _ in vfovs]
    turned = np.zeros(len(vfovs))
    try:
        for vfov in vfovs:
            trial = _replace(
                cfg,
                vfov=float(vfov),
                roi_circ=[round(v) for v in pts.ravel()],
                roi_c=None,
                roi_r=None,
                c2a_r=[0.0, 0.0, 0.0],
            )
            try:
                camera = source_camera(source.width, source.height, vfov, cfg.fisheye)
                _, half = fit_ball(pts, camera)
                if not np.isfinite(half) or np.degrees(half) > MAX_HALF_ANGLE_DEG:
                    raise ValueError("impossible geometry")
                trackers.append(Tracker(trial, source.width, source.height, params))
            except (ValueError, np.linalg.LinAlgError):
                trackers.append(None)
        for index in range(n_frames):
            frame = source.read()
            if frame is None:
                break
            for k, tracker in enumerate(trackers):
                if tracker is None:
                    continue
                try:
                    result = tracker.process_frame(frame.image, frame.ts_ms)
                except (ValueError, np.linalg.LinAlgError):
                    trackers[k], costs[k], turned[k] = None, [], 0.0
                    continue
                if result is not None:
                    turned[k] += float(np.linalg.norm(result.w_cam))
                    if index >= WARMUP_FRAMES:
                        costs[k].append(result.step.cost)
    finally:
        source.close()
    out = []
    for values in costs:
        values = [c for c in values if np.isfinite(c)]
        out.append(float(np.median(values)) if len(values) >= 10 else float("nan"))
    return np.array(out), np.degrees(turned)


def fit_vfov(
    src_spec,
    cfg: Config,
    circle,
    *,
    n_frames: int = VFOV_FRAMES,
    grid=None,
    params=None,
) -> VfovFit:
    """Fit `vfov` from the photometric cost, holding the ball's pixel circle fixed.

    Where the ball is small in the frame the curve is flat and any value in the flat
    region tracks identically; the fit says so rather than pretending to a number.
    """
    lo, hi, count = grid or VFOV_GRID
    points = _circle_points(circle)
    values = np.geomspace(lo, hi, count)
    costs, turns = _costs_at(src_spec, cfg, points, values, n_frames, params)
    for value, cost in zip(values, costs, strict=True):
        log.info("vfov %.3g deg: cost %.5g", value, cost)
    curve = [(float(v), float(c)) for v, c in zip(values, costs, strict=True)]
    usable = np.isfinite(costs)
    if usable.sum() < 3:
        raise ValueError("could not track at any candidate field of view")
    turned_deg = float(np.median(turns[turns > 0])) if (turns > 0).any() else 0.0
    if turned_deg < MIN_TURN_DEG:
        raise ValueError(
            f"the ball turned only {turned_deg:.0f} deg over {n_frames} frames, too "
            f"little to fit the field of view from; set `vfov` from the lens instead"
        )
    values, costs = values[usable], costs[usable]
    best = int(np.argmin(costs))
    spread = float(costs.max() / costs.min())
    flat_range = _flat_run(values, costs, best)

    # An interior minimum clearly below both neighbors on a curve that is not flat end
    # to end. On a flat curve (the near-orthographic regime) which candidate comes out
    # lowest is noise.
    if spread > FLAT_TOL and 0 < best < len(costs) - 1:
        depth = float(min(costs[best - 1], costs[best + 1]) / costs[best])
        if depth >= MIN_DEPTH:
            inner = np.geomspace(values[best - 1], values[best + 1], VFOV_REFINE + 2)
            inner = inner[1:-1]
            more, _ = _costs_at(src_spec, cfg, points, inner, n_frames, params)
            for value, cost in zip(inner, more, strict=True):
                log.info("vfov %.3g deg: cost %.5g", value, cost)
            vfov = _parabola_minimum(
                np.r_[values, inner], np.r_[costs, more], values[[best - 1, best + 1]]
            )
            extra = [(float(v), float(c)) for v, c in zip(inner, more, strict=True)]
            return VfovFit(
                vfov, True, spread, flat_range, turned_deg, depth, curve + extra
            )

    # No believable interior minimum. Usable only where the cost is flat around the
    # best candidate (the near-orthographic regime); anywhere else the search failed.
    near = costs[max(best - 1, 0) : best + 2]
    if float(near.max() / near.min()) > FLAT_TOL:
        raise ValueError(
            f"the photometric cost has no clear minimum between {lo:g} and {hi:g} deg "
            f"(lowest at {values[best]:g} deg), so `vfov` cannot be fitted from this "
            f"recording; set it from the lens"
        )
    return VfovFit(
        float(np.sqrt(flat_range[0] * flat_range[1])),
        False,
        spread,
        flat_range,
        turned_deg,
        1.0,
        curve,
    )


def _parabola_minimum(values, costs, bracket) -> float:
    """Minimum of a parabola in `log(vfov)` through the best point and its neighbors."""
    ok = np.isfinite(costs)
    order = np.argsort(values[ok])
    x, c = np.log(values[ok][order]), costs[ok][order]
    best = int(np.argmin(c))
    if not 0 < best < len(c) - 1:
        return float(np.exp(x[best]))
    (x0, x1, x2), (c0, c1, c2) = x[best - 1 : best + 2], c[best - 1 : best + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (c1 - c0) + x1 * (c0 - c2) + x0 * (c2 - c1)) / denom
    b = (x2**2 * (c0 - c1) + x1**2 * (c2 - c0) + x0**2 * (c1 - c2)) / denom
    if a <= 0:
        return float(np.exp(x1))
    lo, hi = np.log(bracket)
    return float(np.exp(np.clip(-b / (2.0 * a), max(x0, lo), min(x2, hi))))


def _flat_run(values, costs, best) -> tuple[float, float]:
    """Fields of view around `best` whose cost is indistinguishable from the minimum.

    Contiguous on purpose: an isolated candidate elsewhere that comes within `FLAT_TOL`
    is a wobble, not evidence that the range between is flat.
    """
    limit = FLAT_TOL * costs[best]
    lo = hi = best
    while lo > 0 and costs[lo - 1] <= limit:
        lo -= 1
    while hi < len(costs) - 1 and costs[hi + 1] <= limit:
        hi += 1
    return float(values[lo]), float(values[hi])
