"""Fill in the geometry a config leaves open, from the recording itself.

`prepare_config` is the single hook `spintrack run` and `spintrack calibrate --auto`
share: it looks at frames from the start of the recording, detects the ball when the
config does not describe one, and reports what it found either way. Both need a source
they can open before tracking begins, so a live camera is refused with a pointer to the
recorded-clip path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.detect import BallDetection, detect_ball, sample_frames
from spintrack.sphere import fit_ball, pixel_circle, rotated_window

log = logging.getLogger("spintrack")

CAMERA_SOURCE_MESSAGE = (
    "automatic geometry needs a seekable source: it looks at frames before tracking "
    "starts. Record a short clip and run `spintrack calibrate CLIP.txt --auto`, then "
    "use the config it writes."
)
# A detected radius this far from the one in the config is worth saying out loud: a
# relative radius error costs about twice as much of the rotation reported about an axis
# in the image plane, so this threshold is already 6% of forward speed on a camera that
# looks at the animal from behind (`tests/test_scale.py`).
DISAGREEMENT_WARN = 0.03

VFOV_GRID = (1.0, 120.0, 9)  # log-spaced candidates, degrees
MAX_HALF_ANGLE_DEG = 60.0
WARMUP_FRAMES = 20  # the map is still filling; those costs say nothing about the vfov
# The cost separates fields of view only once the map is full enough that a new view is
# matched against surface seen under a different rotation. Over 300 frames of
# `clean_fly` the curve is flat to 1.5% and its minimum is noise; over 1000 it has a
# clean minimum at the true 30 deg, 1.3x below its neighbors.
VFOV_FRAMES = 1000
# A minimum is believed only when its nearer neighbor costs this much more. Measured:
# the benchmark scenes that identify their field of view sit at 1.04-4.06x, while real
# trial 003 (a 1.2 deg half-angle, near-orthographic) dips 0.2% below a neighbor on a
# curve whose neighboring points jump 20% either way - noise, not a minimum.
MIN_DEPTH = 1.02
# Candidates within this factor of the minimum count as indistinguishable from it.
FLAT_TOL = 1.05
# Below this much accumulated rotation the recording says nothing about the geometry.
MIN_TURN_DEG = 90.0


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
        """The `checks["ball"]` entry of the sidecar."""
        out: dict = {"source": self.ball_source}
        if self.detection is not None:
            out.update(
                centre_px=[self.detection.cx, self.detection.cy],
                radius_px=self.detection.r,
                confidence=self.detection.confidence,
                rim_fraction=self.detection.rim_fraction,
                residual_px=self.detection.residual_px,
                n_frames=self.detection.n_frames,
                method=self.detection.method,
            )
        if self.config_radius_px is not None:
            out["config_radius_px"] = self.config_radius_px
        if self.radius_disagreement is not None:
            out["radius_disagreement"] = self.radius_disagreement
        return out

    def line(self) -> str:
        """One line for the terminal block."""
        if self.detection is None:
            return "from config (not checked)"
        d = self.detection
        where = f"({d.cx:.1f}, {d.cy:.1f}) r {d.r:.1f} px"
        if self.ball_source == "detected":
            return f"detected at {where}, confidence {d.confidence:.2f}"
        return (
            f"from config; detection says {where}, "
            f"{100 * self.radius_disagreement:+.1f}% on the radius"
        )


def resolve_source(config_path, cfg: Config, override=None) -> str:
    """The source a config points at: a camera index, or a path relative to it."""
    from pathlib import Path

    spec = override if override is not None else cfg.src_fn
    if not spec:
        raise ValueError("no source: set src_fn in the config or pass --src")
    if str(spec).isdigit():
        return str(spec)
    path = Path(spec)
    return str(path if path.is_absolute() else Path(config_path).parent / path)


def _circle_from(roi_circ) -> tuple[float, float, float]:
    """Algebraic circle through a flat `roi_circ` point list."""
    pts = np.asarray(roi_circ, dtype=np.float64).reshape(-1, 2)
    a = np.stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))], 1)
    sol, *_ = np.linalg.lstsq(a, (pts**2).sum(axis=1), rcond=None)
    return (
        float(sol[0]),
        float(sol[1]),
        float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)),
    )


def config_circle(cfg: Config, width: int, height: int) -> tuple[float, float, float]:
    """The ball's image circle as the config describes it; raises if it has none."""
    if len(cfg.roi_circ) >= 6:
        return _circle_from(cfg.roi_circ)
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
    centre, half_angle = fit_ball(points, camera)
    cfg.roi_c = [float(v) for v in centre]
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
    the two are compared and the difference is reported, never applied. `vfov` is fitted
    with the pixel circle held fixed, since the two together set the ball's angular
    radius and the cost cannot separate them. Mutates `cfg`.
    """
    if str(src_spec).isdigit():
        raise ValueError(CAMERA_SOURCE_MESSAGE)
    frames = sample_frames(str(src_spec), n_frames, span)
    height, width = frames[0].shape
    prepared = Prepared(detection=detect_ball(frames, max_frames=n_frames))
    detection = prepared.detection

    if cfg.has_ball():
        radius = config_circle(cfg, width, height)[2]
        prepared.config_radius_px = radius
        prepared.radius_disagreement = detection.r / radius - 1.0
        if abs(prepared.radius_disagreement) > DISAGREEMENT_WARN:
            message = (
                f"detected ball radius {detection.r:.1f} px differs from the config's "
                f"{radius:.1f} px by {100 * prepared.radius_disagreement:+.1f}%; that "
                f"is about twice as much again on every in-plane rotation reported"
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
        circle = detection if not cfg.roi_circ else _circle_from(cfg.roi_circ)
        prepared.vfov = fit_vfov(
            src_spec, cfg, circle, n_frames=vfov_frames, params=params
        )
        cfg.vfov = prepared.vfov.vfov
        if not cfg.roi_circ:
            cfg.roi_circ = _circle_points(circle)
        _set_ball_from_points(cfg, width, height)
    return prepared


@dataclass
class VfovFit:
    """The field of view the photometric cost prefers, and how much it prefers it."""

    vfov: float
    identifiable: bool
    spread: float  # max cost / min cost over the grid
    flat_range: tuple[float, float] | None  # vfovs whose cost is within 5% of the best
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
                f"{self.vfov:.4g} deg (fitted; identifiable, the cost minimum is "
                f"{self.depth:.2g}x below its neighbors and varies {self.spread:.2g}x "
                f"over the search range)"
            )
        lo, hi = self.flat_range or (float("nan"), float("nan"))
        return (
            f"{self.vfov:.4g} deg (fitted; not identifiable, cost varies "
            f"{self.spread:.3g}x over {lo:.3g}-{hi:.3g} deg; the rotation scale does "
            f"not depend on this choice)"
        )


def _circle_points(circle, n: int = 16) -> list[int]:
    """`roi_circ` points from a `BallDetection` or a plain `(cx, cy, r)`."""
    if isinstance(circle, BallDetection):
        return circle.rim_points(n)
    cx, cy, r = circle
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return [round(float(v)) for v in xy.ravel()]


def _cost_at(
    src_spec, cfg: Config, points, vfov: float, n_frames: int, params
) -> tuple[float, float]:
    """Median cost and total turned angle from tracking the first frames at this `vfov`.

    The cost is NaN when the geometry is impossible at this `vfov` (the rim points do
    not describe a ball, or the ball would cover most of the sky).
    """
    from dataclasses import replace as _replace

    from spintrack.engine import TrackParams
    from spintrack.io.sources import VideoSource
    from spintrack.tracker import Tracker

    trial = _replace(
        cfg, vfov=float(vfov), roi_circ=list(points), roi_c=None, roi_r=None
    )
    # This search lives on the cost's sensitivity to the geometry, so the two things
    # that explain away part of a wrong field of view's cost are turned off: the static
    # illumination field, which absorbs a systematic misregistration residual as if it
    # were shading, and the window pre-filter, which blurs it. With the field learning,
    # the curve's depth on the 16-degree test ball falls from 2.2 to 1.96.
    params = _replace(
        params or TrackParams(), prefilter=0.0, illum_bias=False, illum_gain=False
    )
    source = VideoSource(src_spec)
    try:
        camera = source_camera(source.width, source.height, vfov, cfg.fisheye)
        _, half_angle = fit_ball(np.asarray(points, float).reshape(-1, 2), camera)
        if not np.isfinite(half_angle) or np.degrees(half_angle) > MAX_HALF_ANGLE_DEG:
            return float("nan"), 0.0
        tracker = Tracker(trial, source.width, source.height, params)
        costs, turned = [], 0.0
        for index in range(n_frames):
            frame = source.read()
            if frame is None:
                break
            result = tracker.process_frame(frame.image, frame.ts_ms)
            if result is not None:
                turned += float(np.linalg.norm(result.w_cam))
                if index >= WARMUP_FRAMES:
                    costs.append(result.step.cost)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), 0.0
    finally:
        source.close()
    costs = [c for c in costs if np.isfinite(c)]
    cost = float(np.median(costs)) if len(costs) >= 10 else float("nan")
    return cost, float(np.degrees(turned))


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

    The pixel circle and `vfov` together set the ball's angular radius, so fitting both
    from the same cost is meaningless; the circle comes from the detector and only the
    field of view is searched here. Where the ball is small in the frame the curve is
    flat and any value in the flat region tracks identically - the fit says so rather
    than pretending to a number.
    """
    lo, hi, count = grid or VFOV_GRID
    points = _circle_points(circle)
    curve, turns = [], []
    for value in np.geomspace(lo, hi, count):
        cost, turned = _cost_at(src_spec, cfg, points, value, n_frames, params)
        log.info("vfov %.3g deg: cost %.5g", value, cost)
        curve.append((float(value), cost))
        turns.append(turned)
    usable = [(v, c) for v, c in curve if np.isfinite(c)]
    if len(usable) < 3:
        raise ValueError("could not track at any candidate field of view")
    turned_deg = float(np.median([t for t in turns if t > 0] or [0.0]))
    if turned_deg < MIN_TURN_DEG:
        raise ValueError(
            f"the ball turned only {turned_deg:.0f} deg over {n_frames} frames, too "
            f"little to fit the field of view from; set `vfov` from the lens instead"
        )

    costs = np.array([c for _, c in usable])
    values = np.array([v for v, _ in usable])
    best = int(np.argmin(costs))
    spread = float(costs.max() / costs.min())
    flat_range = _flat_run(values, costs, best)

    # An interior minimum is most of the test: the scenes that fail all have their
    # minimum at an end of the grid. It is not the whole test, though. On a curve that
    # is flat to within `FLAT_TOL` from end to end - the near-orthographic regime, where
    # the cost genuinely does not depend on the field of view - which candidate comes
    # out lowest is noise, and an interior one lands there as easily as an end one. Flat
    # is flat: say so rather than reporting a fitted value from a 1% wobble.
    if spread > FLAT_TOL and 0 < best < len(costs) - 1:
        depth = float(min(costs[best - 1], costs[best + 1]) / costs[best])
        vfov, extra = _golden_section(
            src_spec, cfg, points, values[best - 1], values[best + 1], n_frames, params
        )
        return VfovFit(vfov, True, spread, flat_range, turned_deg, depth, curve + extra)

    # No interior minimum, or a flat curve. Usable only where the cost is flat around
    # the best candidate, which is the near-orthographic regime; anywhere else the
    # search has simply failed.
    near = costs[max(best - 1, 0) : best + 2]
    if float(near.max() / near.min()) > FLAT_TOL:
        raise ValueError(
            f"the photometric cost has no minimum between {lo:g} and {hi:g} deg (it "
            f"falls towards {values[best]:g} deg and keeps changing there), so `vfov` "
            f"cannot be fitted from this recording; set it from the lens"
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


def _flat_run(values, costs, best) -> tuple[float, float]:
    """Fields of view around `best` whose cost is indistinguishable from the minimum.

    Contiguous on purpose: an isolated candidate elsewhere on the curve that happens to
    come within `FLAT_TOL` is a wobble, not evidence that the range between is flat.
    """
    limit = FLAT_TOL * costs[best]
    lo = hi = best
    while lo > 0 and costs[lo - 1] <= limit:
        lo -= 1
    while hi < len(costs) - 1 and costs[hi + 1] <= limit:
        hi += 1
    return float(values[lo]), float(values[hi])


def _golden_section(src_spec, cfg, points, left, right, n_frames, params, evals=6):
    """Minimise the cost in `log(vfov)` between two bracketing candidates."""
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = np.log(left), np.log(right)
    c, d = b - phi * (b - a), a + phi * (b - a)
    extra = []

    def cost(x):
        value = float(np.exp(x))
        out, _ = _cost_at(src_spec, cfg, points, value, n_frames, params)
        extra.append((value, out))
        log.info("vfov %.3g deg: cost %.5g", value, out)
        return np.inf if not np.isfinite(out) else out

    fc, fd = cost(c), cost(d)
    for _ in range(evals - 2):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = cost(d)
    return float(np.exp(c if fc < fd else d)), extra


INNER_MAX = 0.55  # window pixels with theta below this fraction of alpha are "inner"
OUTER_MIN = 0.70  # and above this, "outer"
MIN_INPLANE_DEG = 0.2  # a smaller in-plane increment carries no usable signal
MIN_CHECKED = 50
# How the outer/inner ratio responds to a relative radius error, and what a *correct*
# radius reads. The two are separate because the reading at a correct radius is not 1
# and is not universal: it is set by the geometry (see `RATIO_ZERO`), while the response
# to a radius error is nearly the same everywhere.
#
# Both are measured on the ratio each recording's own control pair has divided out
# (`ScaleCheck.step`), which is what makes the response scene-independent. Measured over
# 18 synthetic scenes with `roi_r` scaled, a check every second frame (300-480 per run)
# and the shipped pre-filter: each entry is the mean of the per-scene ratio over that
# scene's own reading at a correct radius, and the scenes agree on it to 0.4-1.5%. The
# raw ratio does not behave that way - over the same scenes it reads 0.962 to 1.062 at a
# correct radius, which is 6 percentage points of radius, because how much of each
# region's motion the solver recovers depends on the recording's own texture and on
# which way the ball turns. With the control pair and these two tables the reading is
# within 0.65 pp of radius (rms over 126 scene-error pairs, each scene left out of the
# response in turn) and within 0.5 pp at a correct radius; the exception is `occluded`,
# where the animal's body and legs stand over the ball and the check reads 1.1-2.9% too
# large, because a pattern that does not turn with the ball holds the outer annulus back
# and the control pair, built from one frame, cannot see that. That is a job for
# `roi_ignr`, and `occluded`'s own polygon covers the body but not the six legs: scaled
# by 2.5 about its centre, which drops 22% of the window's pixels, it takes the bias at
# three planted errors from +1.8, +2.4 and +1.4 to +0.6, +0.8 and +0.6 pp. The holder
# shadow does the same thing more gently - `lab_big_shadow` and `lab_big_ball` differ in
# nothing else, and their corrected ratios differ by 1%, half a point of radius.
#
# The near-orthographic prediction (a point at normalized offset t sits at depth sqrt(1
# - t^2) and moves by omega times that depth, so a solver assuming radius 1 + eps
# reports omega sqrt(1 - t^2) / sqrt((1 + eps)^2 - t^2)) has the right shape but is far
# too steep: it wants a ratio of 1.6 at -10% where the measurement says 1.13. The window
# is resampled at the assumed radius, which flattens the difference between the two
# regions, so the calibration is measured rather than derived.
RATIO_RESPONSE = (
    (-0.10, 1.1326),
    (-0.05, 1.0719),
    (-0.02, 1.0282),
    (0.00, 1.0000),
    (0.02, 0.9650),
    (0.05, 0.9080),
    (0.10, 0.8056),
)
# The corrected ratio a correct radius produces, against the ball's half-angle in
# degrees (`lab_small_ball`, `lab_big_ball`, `offaxis`, `clean_fly`; interpolated in the
# log of the half-angle, held flat outside). It falls away from 1 as the ball gets
# smaller because the rim of a near-orthographic ball is where the outer annulus lives
# and where the surface is most foreshortened: at a 0.31 degree half-angle the outermost
# ring of the window sees the surface almost edge-on, at 11 degrees the silhouette is
# already 11 degrees short of edge-on. Between the calibrated half-angles the
# interpolation is worth about half a percentage point of radius; the window size moves
# it by another 0.2-0.5% of ratio between `q_factor` 6 and 24, which is not calibrated
# for.
RATIO_ZERO = (
    (0.31, 0.9665),
    (1.20, 0.9815),
    (8.00, 1.0024),
    (11.00, 1.0032),
)


def ratio_at_correct_radius(half_angle: float) -> float:
    """`RATIO_ZERO` at a half-angle in radians."""
    degrees = np.degrees(max(float(half_angle), 1e-9))
    knots = np.log(np.array([a for a, _ in RATIO_ZERO]))
    values = np.log(np.array([r for _, r in RATIO_ZERO]))
    return float(np.exp(np.interp(np.log(degrees), knots, values)))


def radius_error_from_ratio(ratio: float, half_angle: float) -> float:
    """Relative radius error implied by a corrected outer/inner rotation ratio.

    Interpolates `RATIO_RESPONSE` (monotone decreasing) against the ratio as a fraction
    of what a correct radius reads at this half-angle, extrapolating linearly in the log
    beyond the calibrated range so that a gross error still reads gross.
    """
    eps = np.array([e for e, _ in RATIO_RESPONSE])
    curve = np.log(np.array([r for _, r in RATIO_RESPONSE]))
    value = np.log(max(ratio, 1e-6) / ratio_at_correct_radius(half_angle))
    if value >= curve[0]:
        slope = (eps[1] - eps[0]) / (curve[1] - curve[0])
        return float(eps[0] + (value - curve[0]) * slope)
    if value <= curve[-1]:
        slope = (eps[-1] - eps[-2]) / (curve[-1] - curve[-2])
        return float(eps[-1] + (value - curve[-1]) * slope)
    return float(np.interp(value, curve[::-1], eps[::-1]))


@dataclass
class ScaleVerdict:
    """What the inner and outer parts of the window say about the assumed radius."""

    # Outer gain over inner gain, divided by what the same recording reads on a pair it
    # cannot disagree about; `RATIO_ZERO` at this half-angle when the radius is right.
    ratio: float
    radius_err_pct: float
    ci_pct: tuple[float, float]
    n_checked: int
    verdict: str  # "ok", "warn", "fail" or "insufficient motion"

    def report(self) -> dict:
        return {
            "ratio": self.ratio,
            "radius_err_pct": self.radius_err_pct,
            "ci_pct": list(self.ci_pct),
            "n_checked": self.n_checked,
            "verdict": self.verdict,
        }

    def line(self) -> str:
        if self.verdict == "insufficient motion":
            return f"not enough in-plane rotation to check (n={self.n_checked})"
        return (
            f"outer/inner ratio {self.ratio:.3f} against its control "
            f"(n={self.n_checked}) -> assumed radius {self.radius_err_pct:+.1f}% "
            f"[{self.ci_pct[0]:+.1f}, {self.ci_pct[1]:+.1f}], {self.verdict}"
        )


class ScaleCheck:
    """Independent check that the ball's assumed radius is right, run while tracking.

    Solving the same frame-to-frame increment on an inner disc and on an outer annulus
    of the tracking window gives two estimates of the same rotation. They agree exactly
    when the assumed radius is the true one, whatever that radius is, because the two
    regions see the surface at different depths and the depth is what the assumed radius
    sets. Rotation about the line of sight carries no depth information, so only the
    in-plane components are compared.

    They agree exactly only for a perfect solver. What the two regions actually recover
    of a given motion also depends on the recording - on how much of each region's
    texture the source resolves, on which way the ball turns, on what is fixed in the
    image and does not move with the ball - and over the benchmark scenes that alone
    moves the ratio by 5 percentage points of radius, more than the errors worth
    reporting. So every check also solves a *control* pair: the same source frame
    remapped twice, once as it is and once through `sphere.rotated_window`, which turns
    the surface by the increment the frame just measured. That pair differs by exactly
    the rotation the assumed geometry describes, so whatever the two regions disagree
    about on it is the solver and the recording, not the radius, and the reading is
    divided by it (`RATIO_RESPONSE`, `RATIO_ZERO`).
    """

    def __init__(
        self, geometry, params, half_angle: float, camera, observation, max_checks=1000
    ):
        from spintrack._core import Engine as _Engine

        size = geometry.size
        rows, cols = np.divmod(geometry.index, size)
        offset = np.hypot(cols + 0.5 - size / 2.0, rows + 0.5 - size / 2.0)
        theta = geometry.rad_per_pixel * offset / max(half_angle, 1e-12)
        self.geometry = geometry
        self.camera = camera
        # `TrackEngine.observation`: what the solver compares, normalization and
        # illumination correction included, so the control pair is prepared exactly as
        # the real one was.
        self.observation = observation
        self.half_angle = half_angle
        self.params = params
        self.max_checks = max_checks
        self.cores = {}
        for name, mask in (("in", theta < INNER_MAX), ("out", theta > OUTER_MIN)):
            if mask.sum() < 100:
                continue
            self.cores[name] = _Engine(
                np.ascontiguousarray(geometry.surface[mask], dtype=np.float32),
                np.ascontiguousarray(geometry.index[mask], dtype=np.int64),
                size,
                2 * round(params.map_scale * size),
                round(params.map_scale * size),
                1,
                None,
            )
        # One entry per checked frame: the window's own increment and each region's, for
        # the measured pair and for the control pair.
        self.rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.controls: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def enabled(self) -> bool:
        return len(self.cores) == 2

    def _solve_pair(self, first, second, r_prev, velocity) -> dict | None:
        """Each region's increment between two observations, None if a solve fails."""
        p = self.params
        out = {}
        for name, core in self.cores.items():
            core.update(
                first,
                r_prev,
                lambda_=1.0,
                w_max=1e6,
                forget_outside=False,
                margin=0,
                update_main=False,
            )
            res = core.solve(
                second,
                r_prev,
                [float(v) for v in velocity],
                use_prev=True,
                levels=1,
                max_iter=p.max_iter,
                tol=p.tol,
                huber=p.huber,
                tukey=p.tukey,
                w_min=p.w_min,
                w_sat=p.w_sat,
                damping=p.damping,
            )
            if not res.converged or res.inlier_frac < 0.5:
                return None
            out[name] = np.asarray(res.w, dtype=np.float64)
        return out

    def step(self, obs_prev, obs, r_prev, w_win, velocity, gray) -> None:
        """Re-solve one increment on each region, and the control pair's."""
        if not self.enabled or len(self.rows) >= self.max_checks:
            return
        if np.degrees(np.linalg.norm(w_win[:2])) < MIN_INPLANE_DEG:
            return
        measured = self._solve_pair(obs_prev, obs, r_prev, velocity)
        if measured is None:
            return
        turned = rotated_window(self.geometry, self.camera, self.half_angle, w_win)
        control = self._solve_pair(
            obs, self.observation(turned.remap(gray)), r_prev, velocity
        )
        if control is None:
            return
        w = np.asarray(w_win, dtype=np.float64)
        self.rows.append((w, measured["in"], measured["out"]))
        self.controls.append((control["in"], control["out"]))

    def result(self, rng=None) -> ScaleVerdict:
        """Robust gains, their ratio against the control's, and what that implies."""
        if len(self.rows) < MIN_CHECKED:
            return ScaleVerdict(
                float("nan"), float("nan"), (float("nan"),) * 2, len(self.rows),
                "insufficient motion",
            )  # fmt: skip
        full = np.array([r[0][:2] for r in self.rows])
        inner = np.array([r[1][:2] for r in self.rows])
        outer = np.array([r[2][:2] for r in self.rows])
        c_inner = np.array([c[0][:2] for c in self.controls])
        c_outer = np.array([c[1][:2] for c in self.controls])

        def corrected(pick):
            measured = _gain_ratio(full[pick], inner[pick], outer[pick])
            control = _gain_ratio(full[pick], c_inner[pick], c_outer[pick])
            return measured / control if control else float("nan")

        every = np.arange(len(full))
        ratio = corrected(every)
        rng = rng or np.random.default_rng(0)
        draws = [corrected(rng.integers(0, len(full), len(full))) for _ in range(200)]
        estimate = 100.0 * radius_error_from_ratio(ratio, self.half_angle)
        low, high = (
            100.0 * radius_error_from_ratio(float(v), self.half_angle)
            for v in np.nanpercentile(draws, [97.5, 2.5])
        )
        size = abs(estimate)
        verdict = "ok" if size < 3.0 else "warn" if size < 8.0 else "fail"
        return ScaleVerdict(ratio, estimate, (low, high), len(self.rows), verdict)


def _gain_ratio(full, inner, outer) -> float:
    """Outer/inner projection of the region increments onto the full-window one."""
    magnitude = np.linalg.norm(full, axis=1)
    keep = magnitude <= np.percentile(magnitude, 90)  # the largest are blur, not signal
    reference = full[keep]
    weight = float((reference * reference).sum())
    if weight <= 0:
        return float("nan")
    g_in = float((inner[keep] * reference).sum() / weight)
    g_out = float((outer[keep] * reference).sum() / weight)
    return g_out / g_in if g_in else float("nan")
