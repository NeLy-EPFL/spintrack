"""Find the ball in the recording, so the user does not have to click its rim.

The silhouette is the only circle in the image with background outside it at every angle,
but a brightness threshold does not find it: the ball is shaded towards the rim and dark
surface blobs touching the rim bite into any thresholded outline, which is why the
baselines in `notes/baseline_detectors.py` come out 18-28% small. Three things fix that:

1. the surface texture rotates while the silhouette does not, so a high per-pixel
   quantile over a hundred frames erases the blobs and leaves the shading envelope;
2. the convex hull of the foreground removes what still bites in, and RANSAC removes what
   sticks out (the animal, the holder);
3. the rim is then fitted sub-pixel by taking, along each of 360 rays, the *outermost*
   strong radial edge - interior blob edges are strong too, and that is where
   `radial_gradient` went wrong.

A detection that does not meet `min_rim_fraction` and `min_confidence` raises rather than
returning a poor circle: the ball's pixel radius sets the rotation scale slightly worse
than 1:1, so a silently wrong radius rescales every speed the tool reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("spintrack")

N_RAYS = 360
MIN_RIM_RAYS = 8  # fewer rays than a circle fit needs
# Per-pixel temporal quantile the silhouette is measured on; see `temporal_stats`.
QUANTILE = 0.9
BORDER_FRACTION = 0.05  # width of the background band, as a fraction of the image
MASK_K_BACKGROUND = 5.0  # deviations from the background level for the foreground mask
MASK_K_MOTION = 5.0  # multiples of the temporal noise level, same purpose
# Tukey constants of the robust rim fit, in units of the residual scale: standard outside
# the fitted circle, tight inside it. Contamination here is one-sided, because taking the
# outermost strong edge of a ray whose rim is hidden reads too small and never too large.
# Measured over the 17 scenes and the three real trials: 1.0 inside takes `static` from
# 0.88% to 0.57% and its confidence from 0.50 to 0.76, and tightens the agreement between
# the three trials of one rig from 0.28% to 0.20%.
TUKEY_OUTSIDE = 4.685
TUKEY_INSIDE = 1.0
# How far from the fitted circle a ray may land and still count as rim, as a fraction of
# the radius or as a multiple of the fit's own residual, whichever is larger. The residual
# term is what keeps a small ball, where 1% of the radius is a fraction of a pixel, from
# cutting mere noise.
RIM_TOL = 0.01
RIM_TOL_RESIDUALS = 3.0
# Inlier band of the coarse hull RANSAC, as a fraction of the candidate circle's radius
# and as an absolute floor. A fixed two pixels is far too tight for a big ball: the hull
# comes from a morphologically closed mask, so it stands a little off the rim, and on the
# real trials' 518 px ball that left RANSAC agreeing with only a third of the outline and
# refusing about half of all detections.
HULL_TOL = 0.015
HULL_TOL_PX = 2.0
MIN_AREA_FRACTION = 0.002
MAX_AREA_FRACTION = 0.85
# Half-width of `relocate_ball`'s rim search, as a fraction of the known radius. It is
# the capture range of the cheap look: the rim has to fall inside it.
RELOCATE_BAND = 0.05


def _theta_coverage(theta, bins: int = 72) -> float:
    """Fraction of `bins` equal angular sectors that hold at least one of `theta`."""
    if theta is None or len(theta) == 0:
        return 0.0
    hit = np.zeros(bins, bool)
    hit[(np.asarray(theta) % (2 * np.pi) / (2 * np.pi) * bins).astype(int) % bins] = (
        True
    )
    return float(hit.mean())


class DetectionError(RuntimeError):
    """Raised when no ball could be found with enough confidence to be used."""


@dataclass
class BallDetection:
    """The ball's image circle, in the continuous pixel coordinates `camera.rays` takes.

    Pixel `(i, j)` of the array has its centre at `(x, y) = (j + 0.5, i + 0.5)`, and the
    principal point sits at `(width / 2, height / 2)`. This is the convention of
    `spintrack.camera` and of FicTrac's `roi_circ`, and it differs by half a pixel from
    raw numpy indices: mixing the two biases `fit_ball`.
    """

    cx: float
    cy: float
    r: float
    confidence: float  # 0..1
    rim_fraction: float  # accepted rays, over the rays that stay inside the image
    residual_px: float  # robust RMS radial residual of the accepted edge points
    n_frames: int
    method: str
    ok: bool
    # Polar description of the measured rim about `(cx, cy)`, one entry per accepted ray.
    rim_theta: np.ndarray = None
    rim_radius: np.ndarray = None

    @property
    def arc_fraction(self) -> float:
        """Fraction of the directions around the centre that carry a rim point.

        Support spread all the way around conditions the fit; support on one short arc
        does not, whether the rest of the rim is out of frame or merely hidden.
        """
        return _theta_coverage(self.rim_theta) if self.rim_theta is not None else 0.0

    def rim_points(self, n: int = 16) -> list[int]:
        """Up to `n` measured rim points as a FicTrac `roi_circ` flat list.

        The points come from the measured rim, not from the fitted circle: under a pinhole
        camera the silhouette of a sphere is an ellipse, and off-axis that matters. On
        `offaxis` (8 deg half-angle, 16 deg off the axis, less than half the rim in frame)
        resampling the fitted circle costs 3.7% of radius and 0.46 deg of centre, while
        the measured rim is exact. Each point is the median of the accepted rays in its
        angular bin, so it is also quieter than any single ray.
        """
        if self.rim_theta is None or len(self.rim_theta) < 3:
            a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            xy = np.stack(
                [self.cx + self.r * np.cos(a), self.cy + self.r * np.sin(a)], 1
            )
            return [round(float(v)) for v in xy.ravel()]
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

    The quantile is what erases the rotating surface texture: over a hundred frames every
    point of the ball is bright in most of them, so a high quantile is the shading
    envelope with the dark blobs gone. It is deliberately not the maximum, which also
    keeps every bright transient - on the real trials the fly's legs and the glint at the
    rim inflate the maximum's radius by 1.4% and make the three trials of one rig disagree
    by 0.9%, against 0.2% at the 90th percentile.

    Frames are held as uint8, so the cost is `max_frames` times the frame size (160 MB for
    100 frames of 1600x1008, doubled while the quantile is taken). The deviation is
    accumulated with Welford, and a single frame gives that frame and zeros, which the
    rest of the module allows for.
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
    """Largest connected foreground component of the temporal statistics, and the spread.

    Foreground is anything far from the background level in `mx` or moving in `sd`; the
    second term is what finds a ball whose brightness matches the background. The returned
    spread is the per-pixel deviation of the background band of `hi`, which is what the
    rim's edge strength is judged against.
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


def _fit_circle(points: np.ndarray, weights: np.ndarray | None = None):
    """Algebraic (Kasa) circle fit; returns `(cx, cy, r)`."""
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


def _angular_coverage(
    points: np.ndarray, cx: float, cy: float, bins: int = 72
) -> float:
    """Fraction of directions about `(cx, cy)` that hold at least one point."""
    ang = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    hit = np.zeros(bins, bool)
    hit[((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins] = True
    return float(hit.mean())


def hull_circle(mask: np.ndarray, rng: np.random.Generator):
    """Coarse circle through the convex hull of the foreground, by RANSAC.

    Dark blobs, legs and the body bite *into* the disc, and the hull removes those bites;
    the animal and the holder stick *out*, and RANSAC treats them as outliers. Hull points
    on the image border are dropped, since a ball cut off by the frame has no rim there.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DetectionError("foreground component has no contour")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :]
    pts = hull.astype(np.float64) + 0.5  # cv2 gives pixel indices, we want centres
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
            cx, cy, r = _fit_circle(sample)
        except (DetectionError, np.linalg.LinAlgError):
            continue
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        inliers = d < max(HULL_TOL_PX, HULL_TOL * r)
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        raise DetectionError("no circle fits the foreground outline")
    cx, cy, r = _fit_circle(pts[best_inliers])
    # Measured against the directions the outline actually occupies, so a ball half out
    # of frame is judged on the part of its rim that is in frame (`offaxis` shows 46%).
    available = _angular_coverage(pts, cx, cy)
    coverage = _angular_coverage(pts[best_inliers], cx, cy) / max(available, 1e-6)
    if coverage < 0.5:
        raise DetectionError(
            f"only {100 * coverage:.0f}% of the outline agrees with a circle"
        )
    return cx, cy, r, coverage


def _polar_grid(cx, cy, r, band, step=0.25, shape=None):
    """The sampling grid of the rim search: radii, ray angles and their pixel positions.

    Given the image `shape`, rays that would leave it are left out instead of being
    sampled and rejected afterwards. That changes no answer - such rays are already
    excluded from the edge search, from the polarity and from `rim_fraction`'s
    denominator - and on the lab trials, where the ball is cut off top and bottom, a
    third of the rays never counted.
    """
    radii = np.arange(r - band, r + band + step, step)
    angles = np.linspace(0.0, 2.0 * np.pi, N_RAYS, endpoint=False)
    if shape is not None:
        h, w = shape
        # A ray's samples lie on a segment, so its two ends decide the whole ray.
        ends = radii[[0, -1]]
        ex = cx + np.outer(np.cos(angles), ends)
        ey = cy + np.outer(np.sin(angles), ends)
        angles = angles[((ex >= 0) & (ex <= w) & (ey >= 0) & (ey <= h)).all(axis=1)]
    xs = cx + np.outer(np.cos(angles), radii)
    ys = cy + np.outer(np.sin(angles), radii)
    return radii, angles, xs, ys


def _polar_sample(image, xs, ys) -> np.ndarray:
    """The image resampled onto a polar grid (rays x radii)."""
    return cv2.remap(
        image,
        (xs - 0.5).astype(np.float32),
        (ys - 0.5).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _radial_edges(image, cx, cy, r, polarity, step=0.25, band=None):
    """Sub-pixel rim radius per ray of `_polar_grid`, plus each ray's edge strength.

    Along every ray the *outermost* radius whose radial gradient reaches half that ray's
    maximum is taken, so a strong interior blob edge cannot win. Rays whose edge lands on
    a search bound, or whose samples leave the image, are rejected.
    """
    band = max(8.0, 0.05 * r) if band is None else band
    radii, angles, xs, ys = _polar_grid(cx, cy, r, band, step, shape=image.shape)
    profile = _polar_sample(image, xs, ys)
    return _edges_from_profile(
        profile, radii, angles, xs, ys, polarity, image.shape, step
    )


def _edges_from_profile(profile, radii, angles, xs, ys, polarity, shape, step=0.25):
    """The rim edge of each ray of an already-sampled polar profile."""
    h, w = shape
    # Differentiate with a derivative-of-Gaussian one pixel wide rather than between
    # neighbouring samples: the profile is bilinearly interpolated every 0.25 px, so
    # consecutive samples share pixels and a plain difference divides noise by 0.25.
    t = np.arange(-3.0, 3.0 + step, step)
    kernel = (-t * np.exp(-0.5 * t * t)).astype(np.float32)
    grad = polarity * cv2.filter2D(profile, -1, kernel.reshape(1, -1))
    inside = (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
    usable = inside.all(axis=1)

    # The outermost *local maximum* worth half the ray's best edge. Taking the outermost
    # half-maximum crossing instead puts every rim point outside the edge by construction,
    # which showed up as a systematic +1% radius on all 17 scenes.
    peak = grad.max(axis=1)
    strong = np.zeros(grad.shape, bool)
    inner = grad[:, 1:-1]
    strong[:, 1:-1] = (
        (inner >= grad[:, :-2]) & (inner > grad[:, 2:]) & (inner > 0.5 * peak[:, None])
    )
    last = grad.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)
    ok = usable & strong.any(axis=1) & (peak > 0)
    rows = np.arange(len(angles))
    k = np.clip(last, 1, grad.shape[1] - 2)
    left, mid, right = grad[rows, k - 1], grad[rows, k], grad[rows, k + 1]
    denom = left - 2.0 * mid + right
    offset = np.zeros(len(angles))
    np.divide(0.5 * (left - right), denom, out=offset, where=np.abs(denom) > 1e-12)
    offset = np.clip(offset, -1.0, 1.0)
    return radii[k] + offset * step, grad[rows, k], ok, usable, angles


@dataclass
class RimFit:
    cx: float
    cy: float
    r: float
    rim_fraction: float  # accepted rays over the rays that stay inside the image
    residual_px: float
    strength: float  # median edge strength of the accepted rays
    theta: np.ndarray  # accepted ray angles about the fitted centre
    radius: np.ndarray  # measured rim radius along each of those rays


def refine_rim(image, cx, cy, r, polarity, min_strength=0.3, rounds=1) -> RimFit:
    """Alternate sub-pixel rim sampling and a robust circle fit around the current one."""
    for _ in range(rounds):
        edge_r, strength, ok, usable, angles = _radial_edges(image, cx, cy, r, polarity)
        if ok.any():
            ok &= strength > min_strength * np.median(strength[ok])
        if ok.sum() < 8:
            raise DetectionError("too few rim points survived the edge search")
        keep_r, keep_a = edge_r[ok], angles[ok]
        pts = np.stack([cx + keep_r * np.cos(keep_a), cy + keep_r * np.sin(keep_a)], 1)
        # IRLS with Tukey weights on the radial residual, started from the RANSAC centre
        # and the median measured radius rather than from an unweighted fit: something
        # bright lying across one sector of the rim (a holder, a leg) pulls an unweighted
        # fit far enough that the offending rays stop looking like outliers, and the
        # iteration then keeps them. Both seeds tolerate half the points being wrong; the
        # RANSAC radius on its own does not, being a mask outline and so slightly large.
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
            cx, cy, r = _fit_circle(pts, weights)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    # Rays that land far from the fitted circle are not rim: too far inside, the rim is
    # hidden behind something dark; too far outside, something bright sits in front of it.
    # Both are dropped and the circle refitted, so the outline handed to `fit_ball` and
    # the circle itself rest on the same points.
    residual = float(np.sqrt(np.average(d**2, weights=weights)))
    keep = np.abs(d) < max(RIM_TOL * r, RIM_TOL_RESIDUALS * residual)
    if keep.sum() < 8:
        raise DetectionError("too few rim points survived the robust fit")
    pts, weights = pts[keep], weights[keep]
    cx, cy, r = _fit_circle(pts, weights)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    residual = float(np.sqrt(np.average(d**2, weights=weights)))
    # Re-express the kept rim about the final centre, which the polar grid was not on.
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return RimFit(
        cx=cx,
        cy=cy,
        r=r,
        rim_fraction=ok.sum() / max(usable.sum(), 1),
        residual_px=residual,
        strength=float(np.median(strength[ok][keep])),
        theta=np.arctan2(dy, dx),
        radius=np.hypot(dx, dy),
    )


@dataclass
class BallRelocation:
    """Where a ball of *known* radius sits now, measured from a short run of frames."""

    cx: float
    cy: float
    rim_fraction: float  # accepted rays, over the rays that stay inside the image
    arc_fraction: float  # directions about the centre that carry an accepted ray
    residual_px: float
    n_frames: int
    polarity: float  # +1 when the ball is brighter than what surrounds it
    ok: bool


def relocate_ball(
    frames,
    cx: float,
    cy: float,
    r: float,
    polarity: float | None = None,
    *,
    quantile: float = QUANTILE,
    band: float | None = None,
    cut: float | None = None,
    min_rim_fraction: float = 0.4,
    min_arc_fraction: float = 0.25,
) -> BallRelocation:
    """Re-measure the centre of a ball whose radius is already known.

    This is `detect_ball`'s job stripped to what following a moving ball needs, and it is
    two orders of magnitude cheaper: the temporal quantile is taken on the polar rim band
    alone (360 rays x a few hundred samples) instead of over whole frames, and the
    foreground mask and hull RANSAC are skipped, since `(cx, cy)` is already within a few
    pixels and `r` is known. The trade is capture range: the rim has to fall inside
    `band`, so a ball that has jumped needs `detect_ball`.

    Holding the radius fixed is not only cheaper but better conditioned. On the real
    trials the ball is cut off at the top and bottom of the frame, so the rim is two side
    arcs; those pin a free circle's centre badly in y (4.3 px of scatter) and a
    known-radius circle's well (1.4-2.5 px).

    `polarity` may be left out, in which case it is read off the profile: the rim band
    is centred on the silhouette, so its inner quarter is ball and its outer quarter is
    whatever surrounds it. `cut` fixes the fit's outlier cut in pixels (see `_fit_centre`).

    Rays through the config's `roi_ignr` are deliberately *not* left out. With the animal
    at the top of the ball that removes the rim's upper arc, and a circle of fixed radius
    held only by its two sides and its bottom is free to climb: on the synthetic
    `lab_small_ball` a look seeded on its own previous answer then walked 80 px off the
    ball. The occluder's pull (3-8 px toward the body) is the lesser evil, and
    `spintrack.refit` is what has to live with it.
    """
    band = max(8.0, RELOCATE_BAND * r) if band is None else band
    images = [np.asarray(frame) for frame in frames]
    if not images:
        raise DetectionError("no frames to relocate the ball in")
    for image in images:
        if image.ndim != 2:
            raise DetectionError(
                f"frames must be 2-D grayscale, got shape {image.shape}"
            )
    shape = images[0].shape
    radii, angles, xs, ys = _polar_grid(cx, cy, r, band, shape=shape)
    if len(angles) < MIN_RIM_RAYS:
        raise DetectionError("the ball's rim band lies outside the image")
    stack = [_polar_sample(image, xs, ys) for image in images]
    k = round(quantile * (len(stack) - 1))
    profile = np.partition(np.stack(stack), k, axis=0)[k].astype(np.float32)
    if polarity is None:
        polarity = _profile_polarity(profile, xs, ys, shape)
    edge_r, strength, ok, usable, angles = _edges_from_profile(
        profile, radii, angles, xs, ys, polarity, shape
    )
    if ok.any():
        ok &= strength > 0.3 * np.median(strength[ok])
    if ok.sum() < 8:
        raise DetectionError("too few rim points survived the edge search")
    keep_r, keep_a = edge_r[ok], angles[ok]
    pts = np.stack([cx + keep_r * np.cos(keep_a), cy + keep_r * np.sin(keep_a)], 1)
    centre, weights, d = _fit_centre(pts, np.array([cx, cy]), r, cut=cut, band=band)
    residual = _weighted_rms(d, weights)
    keep = np.abs(d) < max(RIM_TOL * r, RIM_TOL_RESIDUALS * residual)
    if keep.sum() >= 8:
        centre, weights, d = _fit_centre(pts[keep], centre, r, cut=cut)
        pts = pts[keep]
        residual = _weighted_rms(d, weights)
    theta = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    rim_fraction = len(pts) / max(usable.sum(), 1)
    arc_fraction = _theta_coverage(theta)
    return BallRelocation(
        cx=float(centre[0]),
        cy=float(centre[1]),
        rim_fraction=float(rim_fraction),
        arc_fraction=arc_fraction,
        residual_px=residual,
        n_frames=len(stack),
        polarity=float(polarity),
        ok=rim_fraction >= min_rim_fraction and arc_fraction >= min_arc_fraction,
    )


def _profile_polarity(profile, xs, ys, shape) -> float:
    """Whether the ball is brighter (+1) or darker (-1) than what surrounds it.

    Only rays that stay inside the image are asked: the rest are extended by border
    replication, which says nothing about the background.
    """
    h, w = shape
    inside = (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
    rays = inside.all(axis=1)
    if not rays.any():
        rays = np.ones(len(profile), bool)
    quarter = max(1, profile.shape[1] // 4)
    core = float(np.median(profile[rays, :quarter]))
    around = float(np.median(profile[rays, -quarter:]))
    return 1.0 if core >= around else -1.0


def _weighted_rms(d, weights) -> float:
    """RMS radial residual; the weights collapsing means no rim was found at all."""
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise DetectionError("no rim point is consistent with a circle of this radius")
    return float(np.sqrt(float(np.sum(weights * d * d)) / total))


def _fit_centre(pts, centre, r, rounds: int = 8, cut=None, band=None):
    """Robust least-squares center of a circle of known radius `r` through `pts`.

    Minimizing `sum w_i (|p_i - c| - r)^2` over `c` alone has the fixed point
    `c = mean_w(p_i - r u_i)` with `u_i` the unit vector from `c` to `p_i`, which is what
    this iterates, re-weighting Tukey-style on the radial residual as `refine_rim` does.

    With `cut` (px) the Tukey cut is that fixed value on both sides, reached by halving
    from twice `band` so that a seed well off the center still converges (a seed `band`
    off puts genuine rim points `band` from the circle); without it the scale is estimated
    from the residuals, which something bright standing past the rim (the animal)
    inflates. See `spintrack.refit`.
    """
    weights = np.ones(len(pts))
    d = np.zeros(len(pts))
    for k in range(rounds):
        delta = pts - centre
        dist = np.hypot(delta[:, 0], delta[:, 1])
        d = dist - r
        if cut is not None:
            cuts = max(float(cut), 2.0 * float(band) / 2**k if band else 0.0)
        else:
            outer = np.abs(d[d >= 0.0])
            scale = max(
                1.4826 * float(np.median(outer)) if outer.size > 5 else 0.0,
                1.4826 * float(np.median(np.abs(d - np.median(d)))),
                1e-3,
            )
            cuts = np.where(d < 0.0, TUKEY_INSIDE, TUKEY_OUTSIDE) * scale
        weights = (1.0 - np.clip(d / cuts, -1.0, 1.0) ** 2) ** 2
        unit = delta / np.maximum(dist, 1e-9)[:, None]
        total = float(weights.sum())
        if total <= 0:
            break
        centre = (weights[:, None] * (pts - r * unit)).sum(0) / total
    return centre, weights, d


def _confidence(
    fit: RimFit, arc_fraction: float, hull_r: float, spread: float
) -> float:
    """Product of clipped linear terms, each 0 at unusable and 1 at clearly good.

    The residual is judged relative to the radius (0.5 px is tight on a 500 px ball and
    hopeless on a 20 px one) and the edge strength against the background pixel spread of
    the same image, not against the temporal noise: on `lighting` the illumination itself
    varies, and a temporal denominator would reject a perfectly good rim. The agreement
    with the coarse hull circle is only a sanity bound, not a quality measure: the hull
    comes from a morphologically closed mask and runs a few percent large, more on a
    compressed video, so it is asked only not to disagree wildly.
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
    """Find the ball in a sequence of grayscale frames; raises `DetectionError` if unsure.

    A single frame works too, with the blob-erasing benefit of the temporal maximum lost.
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
    detection = BallDetection(
        cx=fit.cx,
        cy=fit.cy,
        r=fit.r,
        confidence=confidence,
        rim_fraction=fit.rim_fraction,
        residual_px=fit.residual_px,
        n_frames=n,
        method="temporal quantile, hull RANSAC, radial edge",
        ok=ok,
        rim_theta=fit.theta,
        rim_radius=fit.radius,
    )
    if not ok:
        raise DetectionError(
            f"ball detection not trustworthy: confidence {confidence:.2f}, rim "
            f"{100 * fit.rim_fraction:.0f}% of {100 * arc_fraction:.0f}% visible, "
            f"residual {fit.residual_px:.2f} px, hull agreement "
            f"{100 * abs(fit.r / r0 - 1):.1f}% (circle would be {fit.cx:.1f}, "
            f"{fit.cy:.1f}, r {fit.r:.1f})"
        )
    # Debug, not info: `autofit.Prepared.line` reports this to the user once, while the
    # moved-ball watch calls this every few tens of frames while it is following.
    log.debug(
        "ball detected at (%.1f, %.1f) r %.1f px from %d frames "
        "(confidence %.2f, rim %.0f%%, residual %.2f px)",
        fit.cx, fit.cy, fit.r, n, confidence, 100 * fit.rim_fraction, fit.residual_px,
    )  # fmt: skip
    return detection


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
