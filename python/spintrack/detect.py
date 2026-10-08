"""Find the ball's silhouette: from scratch, and from a nearby seed on every frame.

`detect_ball` needs no prior. A high per-pixel quantile over many frames erases the
rotating surface texture, the convex hull of the foreground removes what bites into the
disc, RANSAC removes what sticks out (the animal, the holder), and the rim is fitted
sub-pixel along 360 rays, taking the outermost strong edge of each, since interior blob
edges are strong too. `RimLook` re-measures a ball of known radius in one frame from a
seed a few pixels off; `spintrack.refit` follows the ball with it.

A detection that is not trustworthy raises `DetectionError` rather than returning a poor
circle: a relative error in the radius costs about twice as much of every rotation
reported about an axis in the image plane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("spintrack")

N_RAYS = 360
MIN_RIM_RAYS = 8  # fewer rays than a circle fit needs
RADIAL_STEP = 0.25  # px between samples along a ray
# Per-pixel temporal quantile the silhouette is measured on: high enough to erase the
# texture, below the maximum, which keeps every bright transient (legs, glints).
QUANTILE = 0.9
BORDER_FRACTION = 0.05  # width of the background band, as a fraction of the image
MASK_K_BACKGROUND = 5.0  # deviations from the background level for the foreground mask
MASK_K_MOTION = 5.0  # multiples of the temporal noise level, same purpose
# Tukey constants of the free rim fit, in units of the residual scale. Contamination is
# one-sided: a ray whose rim is hidden reads too small, never too large.
TUKEY_OUTSIDE = 4.685
TUKEY_INSIDE = 1.0
# How far from the fitted circle a ray may land and still count as rim: a fraction of
# the radius or a multiple of the fit's residual, so a small ball keeps its noisy rays.
RIM_TOL = 0.01
RIM_TOL_RESIDUALS = 3.0
# Inlier band of the hull RANSAC, relative with an absolute floor: the hull comes from a
# closed mask and stands a little off the rim, more so on a big ball.
HULL_TOL = 0.015
HULL_TOL_PX = 2.0
MIN_AREA_FRACTION = 0.002
MAX_AREA_FRACTION = 0.85
# Checking a circle proposed from a segmentation mask. The mask edge sits about 1%
# inside the rim on the quantile image (0.3-1.7% on the lab test set), so the rim is
# searched in a band about a slightly larger circle, and has to agree with the mask.
MASK_INSIDE = 1.01
MASK_BAND = 0.03  # half-width of the rim search, relative to the radius
MASK_AGREE = 0.04  # largest radius or center difference to the mask's circle
MIN_SUPPORT = 0.35  # of the mask's rim arc that the rim search has to confirm
MIN_MASK_ARC = 0.25  # directions about the center that carry a rim point
MAX_RESIDUAL = 0.01  # RMS radial residual, relative to the radius
MIN_CONTRAST = 3.0  # rim edge strength over the image's pixel-to-pixel noise


def _arc_bins(theta, bins: int = 72) -> np.ndarray:
    """Which of `bins` equal angular sectors hold at least one of `theta`."""
    hit = np.zeros(bins, bool)
    if theta is not None and len(theta):
        sector = np.asarray(theta) % (2 * np.pi) / (2 * np.pi) * bins
        hit[sector.astype(int) % bins] = True
    return hit


def _theta_coverage(theta, bins: int = 72) -> float:
    """Fraction of `bins` equal angular sectors that hold at least one of `theta`."""
    return float(_arc_bins(theta, bins).mean())


def circle_points(cx: float, cy: float, r: float, n: int = 16) -> list[int]:
    """`n` points of a circle as a FicTrac `roi_circ` flat list."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return [round(float(v)) for v in xy.ravel()]


class DetectionError(RuntimeError):
    """Raised when no ball could be found with enough confidence to be used."""


@dataclass
class BallDetection:
    """The ball's image circle, in the continuous pixel coordinates `camera.rays` takes.

    Pixel `(i, j)` of the array has its center at `(x, y) = (j + 0.5, i + 0.5)`, as in
    `spintrack.camera` and FicTrac's `roi_circ`; raw numpy indices would bias
    `fit_ball` by half a pixel.
    """

    cx: float
    cy: float
    r: float
    confidence: float  # 0..1
    rim_fraction: float  # accepted rays, over the rays that stay inside the image
    residual_px: float  # robust RMS radial residual of the accepted edge points
    n_frames: int
    ok: bool
    # The measured rim in polar form about `(cx, cy)`, one entry per accepted ray.
    rim_theta: np.ndarray = None
    rim_radius: np.ndarray = None
    # The segmentation model's score when a model proposed the ball, else None.
    model_score: float | None = None

    @property
    def arc_fraction(self) -> float:
        """Fraction of the directions around the center that carry a rim point."""
        return _theta_coverage(self.rim_theta) if self.rim_theta is not None else 0.0

    def rim_points(self, n: int = 16) -> list[int]:
        """Up to `n` measured rim points as a FicTrac `roi_circ` flat list.

        The points come from the measured rim, not from the fitted circle: off the
        optical axis a pinhole camera images a sphere as an ellipse, which the circle
        misses. Each point is the median of the accepted rays in its angular bin.
        """
        if self.rim_theta is None or len(self.rim_theta) < 3:
            return circle_points(self.cx, self.cy, self.r, n)
        edges = np.linspace(0.0, 2.0 * np.pi, n + 1)
        which = np.clip(np.digitize(self.rim_theta % (2 * np.pi), edges) - 1, 0, n - 1)
        out: list[int] = []
        for k in range(n):
            in_bin = which == k
            if not in_bin.any():
                continue
            theta = float(np.median(self.rim_theta[in_bin]))
            radius = float(np.median(self.rim_radius[in_bin]))
            out += [
                round(self.cx + radius * np.cos(theta)),
                round(self.cy + radius * np.sin(theta)),
            ]
        return out


def temporal_stats(
    frames, max_frames: int = 100, quantile: float = QUANTILE
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-pixel high quantile and standard deviation over the frames.

    Frames are held as uint8, so the cost is `max_frames` times the frame size, doubled
    while the quantile is taken. A single frame gives that frame and zero deviation.
    """
    kept: list[np.ndarray] = []
    mean = m2 = None
    for frame in frames:
        if len(kept) >= max_frames:
            break
        image = np.asarray(frame)
        if image.ndim != 2:
            raise DetectionError(
                f"frames must be 2-D grayscale, got shape {image.shape}"
            )
        kept.append(image.astype(np.uint8, copy=False))
        value = image.astype(np.float32)
        if mean is None:
            mean, m2 = value, np.zeros_like(value)
            continue
        delta = value - mean
        mean += delta / len(kept)
        m2 += delta * (value - mean)
    n = len(kept)
    if n == 0:
        raise DetectionError("no frames to detect the ball in")
    stack = np.stack(kept)
    k = round(quantile * (n - 1))
    hi = np.partition(stack, k, axis=0)[k].astype(np.float32)
    sd = np.sqrt(m2 / (n - 1)) if n > 1 else np.zeros_like(hi)
    return hi, sd, n


def _background_level(image: np.ndarray) -> tuple[float, float]:
    """Median and robust deviation over the outer border band of an image."""
    h, w = image.shape
    band = max(1, round(BORDER_FRACTION * min(h, w)))
    edge = np.concatenate(
        [
            image[:band].ravel(),
            image[-band:].ravel(),
            image[band:-band, :band].ravel(),
            image[band:-band, -band:].ravel(),
        ]
    )
    level = float(np.median(edge))
    return level, 1.4826 * float(np.median(np.abs(edge - level)))


def foreground_mask(hi: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest foreground component of the temporal statistics, and its spread.

    Foreground is anything far from the background level in `hi` or moving in `sd`; the
    second term finds a ball whose brightness matches the background. The spread is the
    pixel deviation of `hi`'s background band, which edge strength is judged against.
    """
    h, w = hi.shape
    level, spread = _background_level(hi)
    _, noise = _background_level(sd)
    noise = max(noise, float(np.median(sd)), 1e-3)
    mask = np.abs(hi - level) > MASK_K_BACKGROUND * max(spread, 1.0)
    mask |= sd > MASK_K_MOTION * noise
    k = max(3, round(0.01 * min(h, w)) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise DetectionError("no foreground: the frames look empty")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[best, cv2.CC_STAT_AREA] / float(h * w)
    if not MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION:
        raise DetectionError(
            f"largest foreground blob covers {100 * area:.2f}% of the image, "
            f"outside {100 * MIN_AREA_FRACTION:g}-{100 * MAX_AREA_FRACTION:g}%"
        )
    return (labels == best).astype(np.uint8), max(spread, 1.0)


def fit_circle(points: np.ndarray, weights: np.ndarray | None = None):
    """Algebraic (Kasa) circle fit; returns `(cx, cy, r)`."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    a = np.stack([2.0 * x, 2.0 * y, np.ones(len(x))], 1)
    b = x * x + y * y
    if weights is not None:
        a = a * weights[:, None]
        b = b * weights
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if not np.isfinite(r2) or r2 <= 0:
        raise DetectionError("degenerate circle fit")
    return float(sol[0]), float(sol[1]), float(np.sqrt(r2))


def _coverage_about(points: np.ndarray, cx: float, cy: float) -> float:
    return _theta_coverage(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))


def hull_circle(mask: np.ndarray, rng: np.random.Generator):
    """Coarse circle through the convex hull of the foreground, by RANSAC.

    The hull removes what bites into the disc (dark blobs, legs, the body) and RANSAC
    what sticks out (the animal, the holder). Hull points on the image border are
    dropped, since a ball cut off by the frame has no rim there.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DetectionError("foreground component has no contour")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :]
    pts = hull.astype(np.float64) + 0.5  # cv2 gives pixel indices, we want centers
    inside = (
        (pts[:, 0] > 3.5) & (pts[:, 0] < w - 3.5)
        & (pts[:, 1] > 3.5) & (pts[:, 1] < h - 3.5)
    )  # fmt: skip
    pts = pts[inside]
    if len(pts) < 5:
        raise DetectionError("ball outline touches the image border almost everywhere")

    best_inliers = None
    for _ in range(200):
        try:
            sample = pts[rng.choice(len(pts), 3, replace=False)]
            cx, cy, r = fit_circle(sample)
        except DetectionError, np.linalg.LinAlgError:
            continue
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        inliers = d < max(HULL_TOL_PX, HULL_TOL * r)
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        raise DetectionError("no circle fits the foreground outline")
    cx, cy, r = fit_circle(pts[best_inliers])
    # Judged against the directions the outline occupies, so a ball half out of frame
    # is judged on the part of its rim that is in frame.
    available = _coverage_about(pts, cx, cy)
    coverage = _coverage_about(pts[best_inliers], cx, cy) / max(available, 1e-6)
    if coverage < 0.5:
        raise DetectionError(
            f"only {100 * coverage:.0f}% of the outline agrees with a circle"
        )
    return cx, cy, r, coverage


def _polar_offsets(r: float, band: float, n_rays: int = N_RAYS):
    """Sample radii of the rim search, and the rays' unit vectors and sample offsets."""
    radii = np.arange(r - band, r + band + RADIAL_STEP, RADIAL_STEP)
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    dx = np.outer(cos, radii).astype(np.float32)
    return radii, cos, sin, dx, np.outer(sin, radii).astype(np.float32)


def _rays_inside(cx, cy, radii, cos, sin, shape) -> np.ndarray:
    """Rays whose samples all stay in the image (a segment's two ends decide)."""
    h, w = shape
    ends = radii[[0, -1]]
    ex = cx + np.outer(cos, ends)
    ey = cy + np.outer(sin, ends)
    return ((ex >= 0) & (ex <= w) & (ey >= 0) & (ey <= h)).all(axis=1)


def _sample(image, cx, cy, dx, dy) -> np.ndarray:
    """The image resampled onto a polar grid (rays x radii), as float32.

    `dx`, `dy` are the samples' float32 offsets from the center; pixel centers sit at
    half-integers, which `cv2.remap` does not know about.
    """
    return cv2.remap(
        image,
        dx + np.float32(cx - 0.5),
        dy + np.float32(cy - 0.5),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)


def _edges(profile, radii, polarity):
    """Sub-pixel rim radius of each ray of a polar profile, its strength, and validity.

    The rim is the outermost local maximum of the radial gradient worth half the ray's
    best edge, so an interior blob edge cannot win, and taking a maximum rather than a
    half-maximum crossing does not put every rim point outside the edge.
    """
    # A derivative of Gaussian one pixel wide: neighboring samples share pixels.
    t = np.arange(-3.0, 3.0 + RADIAL_STEP, RADIAL_STEP)
    kernel = (-t * np.exp(-0.5 * t * t)).astype(np.float32)
    grad = polarity * cv2.filter2D(profile, -1, kernel.reshape(1, -1))
    peak = grad.max(axis=1)
    strong = np.zeros(grad.shape, bool)
    inner = grad[:, 1:-1]
    strong[:, 1:-1] = (
        (inner >= grad[:, :-2]) & (inner > grad[:, 2:]) & (inner > 0.5 * peak[:, None])
    )
    last = grad.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)
    ok = strong.any(axis=1) & (peak > 0)
    rows = np.arange(len(grad))
    k = np.clip(last, 1, grad.shape[1] - 2)
    left, mid, right = grad[rows, k - 1], grad[rows, k], grad[rows, k + 1]
    denom = left - 2.0 * mid + right
    offset = np.zeros(len(grad))
    np.divide(0.5 * (left - right), denom, out=offset, where=np.abs(denom) > 1e-12)
    offset = np.clip(offset, -1.0, 1.0)
    return radii[k] + offset * RADIAL_STEP, grad[rows, k], ok


def _profile_polarity(profile) -> float:
    """+1 when the ball (the band's inner quarter) is brighter than its surround."""
    quarter = max(1, profile.shape[1] // 4)
    core = float(np.median(profile[:, :quarter]))
    around = float(np.median(profile[:, -quarter:]))
    return 1.0 if core >= around else -1.0


@dataclass
class RimFit:
    cx: float
    cy: float
    r: float
    rim_fraction: float  # accepted rays over the rays that stay inside the image
    residual_px: float
    strength: float  # median edge strength of the accepted rays
    theta: np.ndarray  # accepted ray angles about the fitted center
    radius: np.ndarray  # measured rim radius along each of those rays


def refine_rim(image, cx, cy, r, polarity, min_strength=0.3, band=None) -> RimFit:
    """Sub-pixel rim around a coarse circle and a robust free circle fit through it.

    The rim is searched within `band` pixels of the circle (default 5% of the radius,
    at least 8 px).
    """
    image = np.asarray(image, dtype=np.float32)
    band = max(8.0, 0.05 * r) if band is None else band
    radii, cos, sin, dx, dy = _polar_offsets(r, band)
    keep = _rays_inside(cx, cy, radii, cos, sin, image.shape)
    profile = _sample(image, cx, cy, dx[keep], dy[keep])
    edge_r, strength, ok = _edges(profile, radii, polarity)
    cos, sin = cos[keep], sin[keep]
    if ok.any():
        ok &= strength > min_strength * np.median(strength[ok])
    if ok.sum() < MIN_RIM_RAYS:
        raise DetectionError("too few rim points survived the edge search")
    pts = np.stack([cx + edge_r[ok] * cos[ok], cy + edge_r[ok] * sin[ok]], 1)
    # IRLS with Tukey weights on the radial residual, started from the RANSAC center
    # and the median measured radius: something bright across one sector of the rim
    # pulls an unweighted start far enough that its rays stop looking like outliers.
    r = float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    for _ in range(6):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
        # The scale comes from the outer side, which hidden rim cannot reach.
        outer = np.abs(d[d >= 0.0])
        scale = max(
            1.4826 * float(np.median(outer)) if outer.size > 5 else 0.0,
            1.4826 * float(np.median(np.abs(d - np.median(d)))),
            1e-3,
        )
        cut = np.where(d < 0.0, TUKEY_INSIDE, TUKEY_OUTSIDE) * scale
        weights = (1.0 - np.clip(d / cut, -1.0, 1.0) ** 2) ** 2
        cx, cy, r = fit_circle(pts, weights)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    # Rays far from the circle are not rim (hidden behind something dark, or covered by
    # something bright); drop them and refit, so the outline and the circle agree.
    residual = float(np.sqrt(np.average(d**2, weights=weights)))
    sel = np.abs(d) < max(RIM_TOL * r, RIM_TOL_RESIDUALS * residual)
    if sel.sum() < MIN_RIM_RAYS:
        raise DetectionError("too few rim points survived the robust fit")
    pts, weights = pts[sel], weights[sel]
    cx, cy, r = fit_circle(pts, weights)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    residual = float(np.sqrt(np.average(d**2, weights=weights)))
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return RimFit(
        cx=cx,
        cy=cy,
        r=r,
        rim_fraction=ok.sum() / max(len(cos), 1),
        residual_px=residual,
        strength=float(np.median(strength[ok][sel])),
        theta=np.arctan2(dy, dx),
        radius=np.hypot(dx, dy),
    )


def _fill_hull(mask: np.ndarray) -> np.ndarray:
    """The mask's convex hull, filled: closes seams, holes and what bites into it."""
    pts = cv2.findNonZero(np.asarray(mask, np.uint8))
    if pts is None:
        raise DetectionError("empty mask")
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out


def mask_circle(mask: np.ndarray, rng: np.random.Generator, n_iter: int = 400):
    """The circle explaining the widest arc of a mask's outline, by RANSAC.

    Points on the image border are dropped. Inliers lie within a band tied to the image
    size rather than the radius, so a huge circle along a straight edge gains nothing,
    and candidates are scored by the angular coverage of their inliers about the
    center: a ball's rim spans a wide arc, an occluder's straight edge a narrow one.
    Returns `(cx, cy, r, theta)`, `theta` the inliers' angles about the center.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DetectionError("empty mask")
    pts = np.concatenate([c[:, 0, :] for c in contours]).astype(np.float64) + 0.5
    inside = (
        (pts[:, 0] > 3.5) & (pts[:, 0] < w - 3.5)
        & (pts[:, 1] > 3.5) & (pts[:, 1] < h - 3.5)
    )  # fmt: skip
    pts = pts[inside]
    if len(pts) < 20:
        raise DetectionError("the mask's outline lies almost all on the image border")
    pts = pts[:: max(1, len(pts) // 1500)]
    tol = max(2.0, 0.004 * max(h, w))
    best, best_coverage = None, -1.0
    for _ in range(n_iter):
        try:
            cx, cy, r = fit_circle(pts[rng.choice(len(pts), 3, replace=False)])
        except DetectionError, np.linalg.LinAlgError:
            continue
        if not 0.02 * min(h, w) < r < 1.5 * max(h, w):
            continue
        near = pts[np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol]
        if len(near) < 10:
            continue
        coverage = _coverage_about(near, cx, cy)
        if coverage > best_coverage:
            best, best_coverage = (cx, cy, r), coverage
    if best is None:
        raise DetectionError("no circle fits the mask's outline")
    cx, cy, r = best
    for _ in range(2):
        near = pts[np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol]
        cx, cy, r = fit_circle(near)
    return cx, cy, r, np.arctan2(near[:, 1] - cy, near[:, 0] - cx)


def _polarity_about(image, cx: float, cy: float, r: float) -> float:
    """+1 when the band just inside the circle is brighter than the one outside."""
    h, w = image.shape
    x0, x1 = max(0, int(cx - 1.2 * r)), min(w, int(cx + 1.2 * r) + 1)
    y0, y1 = max(0, int(cy - 1.2 * r)), min(h, int(cy + 1.2 * r) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / r
    crop = image[y0:y1, x0:x1]
    inner, outer = crop[(d > 0.8) & (d < 0.95)], crop[(d > 1.05) & (d < 1.2)]
    if not inner.size or not outer.size:
        return 1.0
    return 1.0 if np.median(inner) >= np.median(outer) else -1.0


def ball_from_masks(image, masks, scores, *, n_frames: int = 1, seed: int = 0):
    """The ball in `image`, from segmentation masks of it and the model's scores.

    Each mask proposes a circle through its outline; the rim is then searched on the
    image in a narrow band about it and has to confirm enough of the arc the mask
    shows, fit a circle tightly, stand out of the noise and agree with the mask. Of
    the masks that pass, the model's score decides, because a round thing the model
    is less sure about (the animal's thorax) can pass too. Raises `DetectionError`
    when none passes.
    """
    image = np.asarray(image, dtype=np.float32)
    noise = max(1.4826 * float(np.median(np.abs(np.diff(image, axis=1)))), 1.0)
    best, reasons = None, []
    for mask, score in zip(masks, scores):
        try:
            mx, my, mr, mask_theta = mask_circle(
                _fill_hull(mask), np.random.default_rng(seed)
            )
            fit = refine_rim(
                image,
                mx,
                my,
                MASK_INSIDE * mr,
                _polarity_about(image, mx, my, mr),
                band=max(4.0, MASK_BAND * mr),
            )
        except DetectionError as exc:
            reasons.append(str(exc))
            continue
        mask_bins = _arc_bins(mask_theta)
        support = (_arc_bins(fit.theta) & mask_bins).sum() / max(mask_bins.sum(), 1)
        arc = _theta_coverage(fit.theta)
        apart = max(abs(fit.r / mr - 1), np.hypot(fit.cx - mx, fit.cy - my) / mr)
        residual = fit.residual_px / fit.r
        checks = {
            "rim confirms": (support, support >= MIN_SUPPORT),
            "arc": (arc, arc >= MIN_MASK_ARC),
            "residual": (residual, residual <= MAX_RESIDUAL),
            "off the mask": (apart, apart < MASK_AGREE),
            "contrast": (fit.strength / noise, fit.strength >= MIN_CONTRAST * noise),
        }
        failed = [f"{k} {v:.3g}" for k, (v, ok) in checks.items() if not ok]
        if failed:
            reasons.append(f"mask at score {score:.2f}: " + ", ".join(failed))
            continue
        if best is None or score > best[1]:
            best = (fit, float(score), float(support), arc)
    if best is None:
        detail = "; ".join(reasons[:3]) if reasons else "no mask proposed"
        raise DetectionError(f"no ball found ({detail})")
    fit, score, support, arc = best
    log.debug(
        "ball at (%.1f, %.1f) r %.1f px from a mask at score %.2f (rim confirms %.0f%% "
        "of its arc, residual %.2f px)",
        fit.cx, fit.cy, fit.r, score, 100 * support, fit.residual_px,
    )  # fmt: skip
    return BallDetection(
        cx=fit.cx,
        cy=fit.cy,
        r=fit.r,
        confidence=support,
        rim_fraction=fit.rim_fraction,
        residual_px=fit.residual_px,
        n_frames=n_frames,
        ok=True,
        rim_theta=fit.theta,
        rim_radius=fit.radius,
        model_score=score,
    )


@dataclass
class BallRelocation:
    """Where the ball sits in one frame, as `RimLook` measures it."""

    cx: float
    cy: float
    r: float  # the look's radius, or the fitted one with `free_radius`
    rim_fraction: float  # accepted rays, over the rays that stay inside the image
    arc_fraction: float  # directions about the center that carry an accepted ray
    residual_px: float
    polarity: float  # +1 when the ball is brighter than what surrounds it
    ok: bool


def _weighted_rms(d, weights) -> float:
    """RMS radial residual; the weights collapsing means no rim was found at all."""
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise DetectionError("no rim point is consistent with a circle of this radius")
    return float(np.sqrt(float(np.sum(weights * d * d)) / total))


def _fit_fixed_cut(pts, center, r, cut, band=None, free_radius=False, rounds=8):
    """Tukey-weighted circle fit with a fixed cut, the radius held unless `free_radius`.

    With the radius held, `sum w_i (|p_i - c| - r)^2` has the fixed point
    `c = mean_w(p_i - r u_i)`, `u_i` the unit vector from `c` to `p_i`. Given `band`,
    the cut anneals from twice the band down to `cut`, so a seed up to a band off still
    converges. Returns `(center, r, weights, residuals)`.
    """
    weights = np.ones(len(pts))
    d = np.zeros(len(pts))
    for k in range(rounds):
        delta = pts - center
        dist = np.hypot(delta[:, 0], delta[:, 1])
        d = dist - r
        cut = float(cut)
        cuts = max(cut, 2.0 * float(band) / 2**k if band else 0.0)
        weights = (1.0 - np.clip(d / cuts, -1.0, 1.0) ** 2) ** 2
        unit = delta / np.maximum(dist, 1e-9)[:, None]
        total = float(weights.sum())
        if total <= 0:
            break
        before = center
        if free_radius:
            cx, cy, r = fit_circle(pts, weights)
            center = np.array([cx, cy])
        else:
            center = (weights[:, None] * (pts - r * unit)).sum(0) / total
        if cuts == cut and np.hypot(*(center - before)) < 1e-6:
            break
    return center, r, weights, d


class RimLook:
    """The rim of a ball of known radius, measured in one frame about a nearby seed.

    The radius is held fixed: with only the sides of the rim in view (the lab rig cuts
    the ball off top and bottom) that pins the center far better than a free circle.
    The outlier cut is fixed too, because something bright past the rim (the animal)
    inflates a scale estimated from the residuals, and a look seeded on its own answer
    then climbs it. The rim has to fall inside `band` of the seed's circle.

    Rays through `roi_ignr` are deliberately kept: without the rim's upper arc, a
    circle held by its sides and bottom is free to climb.
    """

    def __init__(self, radius: float, band: float, cut: float):
        self.radius, self.band, self.cut = float(radius), float(band), float(cut)
        self.radii, self.cos, self.sin, self.dx, self.dy = _polar_offsets(radius, band)
        self._rays = None  # the last image-bound subset of rays and their offsets

    def __call__(
        self,
        image,
        cx: float,
        cy: float,
        polarity: float | None = None,
        *,
        free_radius: bool = False,
        min_rim_fraction: float = 0.3,
        min_arc_fraction: float = 0.25,
    ) -> BallRelocation:
        """Measure the rim in `image` (2-D) about the seed `(cx, cy)`.

        `polarity` is read off the profile when not given. `free_radius` fits the
        radius as well, seeded on the look's own.
        """
        image = np.asarray(image)
        if image.ndim != 2:
            raise DetectionError(f"frames must be 2-D grayscale, got {image.shape}")
        keep = _rays_inside(cx, cy, self.radii, self.cos, self.sin, image.shape)
        n_rays = int(keep.sum())
        if n_rays < MIN_RIM_RAYS:
            raise DetectionError("the ball's rim band lies outside the image")
        if self._rays is None or not np.array_equal(keep, self._rays[0]):
            self._rays = (keep, self.dx[keep], self.dy[keep])
        profile = _sample(image, cx, cy, *self._rays[1:])
        if polarity is None:
            polarity = _profile_polarity(profile)
        edge_r, strength, ok = _edges(profile, self.radii, polarity)
        if ok.any():
            ok &= strength > 0.3 * np.median(strength[ok])
        if ok.sum() < MIN_RIM_RAYS:
            raise DetectionError("too few rim points survived the edge search")
        cos, sin = self.cos[keep][ok], self.sin[keep][ok]
        pts = np.stack([cx + edge_r[ok] * cos, cy + edge_r[ok] * sin], 1)
        center, r, weights, d = _fit_fixed_cut(
            pts, np.array([cx, cy]), self.radius, self.cut, self.band, free_radius
        )
        residual = _weighted_rms(d, weights)
        sel = np.abs(d) < max(RIM_TOL * r, RIM_TOL_RESIDUALS * residual)
        if sel.sum() >= MIN_RIM_RAYS:
            pts = pts[sel]
            center, r, weights, d = _fit_fixed_cut(
                pts, center, r, self.cut, free_radius=free_radius
            )
            residual = _weighted_rms(d, weights)
        theta = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        rim_fraction = len(pts) / n_rays
        arc_fraction = _theta_coverage(theta)
        return BallRelocation(
            cx=float(center[0]),
            cy=float(center[1]),
            r=float(r),
            rim_fraction=float(rim_fraction),
            arc_fraction=arc_fraction,
            residual_px=residual,
            polarity=float(polarity),
            ok=rim_fraction >= min_rim_fraction and arc_fraction >= min_arc_fraction,
        )


def _confidence(
    fit: RimFit, arc_fraction: float, hull_r: float, spread: float
) -> float:
    """Product of clipped linear terms, each 0 at unusable and 1 at clearly good.

    The residual is judged relative to the radius and the edge strength against the
    background's pixel spread (a temporal one would reject a rig whose lighting
    varies). The hull circle runs a few percent large, so agreement with it is only a
    sanity bound.
    """

    def term(value, bad, good):
        return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))

    return (
        term(fit.rim_fraction, 0.3, 0.8)
        * term(arc_fraction, 0.15, 0.45)
        * term(fit.residual_px / max(fit.r, 1e-6), 0.02, 0.004)
        * term(abs(fit.r / hull_r - 1.0), 0.25, 0.10)
        * term(fit.strength / spread, 1.0, 5.0)
    )


def detect_ball(
    frames,
    *,
    max_frames: int = 100,
    min_rim_fraction: float = 0.5,
    min_confidence: float = 0.6,
    seed: int = 0,
) -> BallDetection:
    """Find the ball in a run of grayscale frames; raises `DetectionError` if unsure.

    A single frame works too, without the texture-erasing benefit of the quantile.
    """
    hi, sd, n = temporal_stats(frames, max_frames)
    mask, spread = foreground_mask(hi, sd)
    cx0, cy0, r0, _ = hull_circle(mask, np.random.default_rng(seed))
    level, _ = _background_level(hi)
    polarity = 1.0 if float(np.median(hi[mask > 0])) >= level else -1.0
    fit = refine_rim(hi, cx0, cy0, r0, polarity)
    arc_fraction = _theta_coverage(fit.theta)
    confidence = _confidence(fit, arc_fraction, r0, spread)
    ok = fit.rim_fraction >= min_rim_fraction and confidence >= min_confidence
    if not ok:
        raise DetectionError(
            f"ball detection not trustworthy: confidence {confidence:.2f}, rim "
            f"{100 * fit.rim_fraction:.0f}% of {100 * arc_fraction:.0f}% visible, "
            f"residual {fit.residual_px:.2f} px, hull agreement "
            f"{100 * abs(fit.r / r0 - 1):.1f}% (circle would be {fit.cx:.1f}, "
            f"{fit.cy:.1f}, r {fit.r:.1f})"
        )
    # Debug, not info: the moved-ball watch calls this while it recovers a lost ball.
    log.debug(
        "ball detected at (%.1f, %.1f) r %.1f px from %d frames "
        "(confidence %.2f, rim %.0f%%, residual %.2f px)",
        fit.cx, fit.cy, fit.r, n, confidence, 100 * fit.rim_fraction, fit.residual_px,
    )  # fmt: skip
    return BallDetection(
        cx=fit.cx,
        cy=fit.cy,
        r=fit.r,
        confidence=confidence,
        rim_fraction=fit.rim_fraction,
        residual_px=fit.residual_px,
        n_frames=n,
        ok=ok,
        rim_theta=fit.theta,
        rim_radius=fit.radius,
    )


def sample_frames(source_spec, n: int = 100, span: int = 300) -> list[np.ndarray]:
    """Every `span // n`-th frame of the first `span` frames of a video."""
    from spintrack.io.sources import VideoSource

    source = VideoSource(source_spec)
    stride = max(1, span // max(n, 1))
    out: list[np.ndarray] = []
    try:
        for index in range(span):
            frame = source.read()
            if frame is None:
                break
            if index % stride == 0:
                out.append(frame.image)
    finally:
        source.close()
    if not out:
        raise DetectionError(f"no frames read from {source_spec}")
    return out
