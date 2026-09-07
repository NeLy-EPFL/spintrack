"""Ball geometry: fit the ball outline, mask it, and build the tracking window.

The tracking window is the view of a virtual fisheye camera pointed at the ball centre.
Every window pixel that sees the ball gets a unit vector from the ball centre to the
surface point it observes (`WindowGeometry.surface`, in the window frame); rotating the
ball rotates these vectors, which is what the solver estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spintrack.camera import Camera, EquidistantCamera, pixel_centres
from spintrack.geometry import normalize, rotation_between

# Anti-aliasing of the window remap, as the standard deviation of a Gaussian in units of
# the decimation (source pixels per window pixel: 5.8 on the synthetic scenes, 8.6 on the
# lab recordings). Bilinear interpolation at that decimation samples one source pixel in
# thirty and aliases the surface texture. Pre-filtering the source halves the per-frame
# error on the synthetic scenes (clean_fly 0.036 -> 0.019 deg, sparse 0.061 -> 0.026) and
# cuts trial 003's disagreement with an optical-flow cross-check by 13-31%; 0.35 and 0.7
# both measure worse. Below `PREFILTER_MIN_DECIMATION` there is nothing to alias, and the
# blur costs 1-4% on the lab-like scenes, so the filter is skipped. It is one `cv2.pyrDown`
# (a binomial blur of variance one in source pixels, then a halving) followed by a Gaussian
# on the half-size image: as accurate as the Gaussian on the full frame and a quarter of
# its cost. A second halving is not - it loses 5-15% of the gain.
PREFILTER_SIGMA = 0.5
PREFILTER_MIN_DECIMATION = 2.0


def tangent_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to `axis` and to each other."""
    axis = normalize(axis)
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = normalize(np.cross(axis, helper))
    e2 = np.cross(axis, e1)
    return e1, e2


def fit_ball(
    points_xy, camera: Camera, iterations: int = 20
) -> tuple[np.ndarray, float]:
    """Fit the ball outline: unit direction to the centre and angular radius (radians).

    Rays through the clicked rim points lie on a cone about the ball centre. The axis is
    initialised from the plane through the ray tips (SVD) and both axis and half-angle are
    refined by Gauss-Newton on the residuals `angle(ray_i, axis) - half_angle`.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("at least three rim points are needed to fit the ball")
    rays = camera.rays(pts[:, 0], pts[:, 1])
    _, _, vt = np.linalg.svd(rays - rays.mean(axis=0))
    axis = vt[-1]
    if axis @ rays.mean(axis=0) < 0:
        axis = -axis
    half = float(np.mean(np.arccos(np.clip(rays @ axis, -1.0, 1.0))))
    for _ in range(iterations):
        e1, e2 = tangent_basis(axis)
        cos_a = np.clip(rays @ axis, -1.0, 1.0)
        ang = np.arccos(cos_a)
        sin_a = np.maximum(np.sqrt(1.0 - cos_a * cos_a), 1e-9)
        residual = ang - half
        jac = np.stack(
            [-(rays @ e1) / sin_a, -(rays @ e2) / sin_a, -np.ones(len(rays))], 1
        )
        delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        axis = normalize(axis + delta[0] * e1 + delta[1] * e2)
        half += float(delta[2])
        if np.linalg.norm(delta) < 1e-13:
            break
    return axis, half


def ball_outline(
    camera: Camera, centre, half_angle: float, n_points: int = 180, shrink: float = 1.0
) -> np.ndarray:
    """Image polygon (K, 2) of the ball outline at `shrink * half_angle` from the centre."""
    c = normalize(np.asarray(centre, dtype=np.float64))
    e1, e2 = tangent_basis(c)
    a = shrink * half_angle
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    dirs = np.cos(a) * c + np.sin(a) * (
        np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2)
    )
    x, y, _ = camera.project(dirs)
    return np.stack([x, y], axis=1)


def pixel_circle(
    camera: Camera, centre, half_angle: float
) -> tuple[float, float, float]:
    """Image circle `(cx, cy, r)` of the ball outline, in continuous pixel coordinates.

    The silhouette of a sphere under a pinhole camera is exactly a circle, so this is the
    inverse of `fit_ball` there; under a fisheye it is the best-fit circle of the outline.
    The circle is fitted rather than summarised: for a ball far off the optical axis the
    outline points crowd one side, and their centroid misses the centre by a few pixels.
    """
    outline = ball_outline(camera, centre, half_angle, 180)
    a = np.stack([2.0 * outline[:, 0], 2.0 * outline[:, 1], np.ones(len(outline))], 1)
    sol, *_ = np.linalg.lstsq(a, (outline**2).sum(axis=1), rcond=None)
    return (
        float(sol[0]),
        float(sol[1]),
        float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)),
    )


def centre_from_pixel_circle(
    camera: Camera, target_px, half_angle: float, seed, iterations: int = 3
) -> np.ndarray:
    """The ball direction whose `pixel_circle` sits at `target_px`; inverts that function.

    The centre of the silhouette is not the projection of the ball's centre. It sits
    farther from the principal point, and by more the farther off axis the ball is, so
    reading a fitted circle's centre as a direction under-reports how far a ball has
    moved: on `ball_drop` (11 degree ball, 71 px of movement) by 5% of the movement, and
    on the real trials (1.2 degree ball) by 0.2 px over 284 px, which is nothing.

    Two or three fixed-point steps are exact to well under a pixel, because the
    correction moves with the projected direction almost one for one.
    """
    target = np.asarray(target_px, dtype=np.float64)
    centre = normalize(np.asarray(seed, dtype=np.float64))
    for _ in range(iterations):
        cx, cy, _ = pixel_circle(camera, centre, half_angle)
        px, py, _ = camera.project(centre)
        centre = normalize(
            camera.rays(float(px) + target[0] - cx, float(py) + target[1] - cy)
        )
    return centre


def source_mask(
    camera: Camera, centre, half_angle: float, ignore_polygons=(), shrink: float = 0.975
) -> np.ndarray:
    """uint8 mask of the source image: 255 on the ball (slightly shrunk), 0 elsewhere.

    `ignore_polygons` are FicTrac `roi_ignr` polygons (flat `x1, y1, x2, y2, ...` lists)
    covering the animal and other occluders; they are cut out of the mask.
    """
    mask = np.zeros((camera.height, camera.width), np.uint8)
    outline = ball_outline(camera, centre, half_angle, shrink=shrink)
    cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 255)
    for poly in ignore_polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 0)
    return mask


@dataclass(frozen=True)
class WindowGeometry:
    """Fixed per-configuration geometry of the tracking window."""

    size: int
    rad_per_pixel: float
    to_camera: np.ndarray  # (3, 3) rotation from the window frame to the camera frame
    map_x: np.ndarray  # (n, n) float32 source x for cv2.remap
    map_y: np.ndarray  # (n, n) float32 source y for cv2.remap
    mask: np.ndarray  # (n, n) bool: window pixels that see un-ignored ball surface
    surface: (
        np.ndarray
    )  # (N, 3) float32 unit ball-centre-to-surface vectors, window frame
    index: np.ndarray  # (N,) int64 flat row-major indices of the masked pixels
    decimation: float = 1.0  # source pixels per window pixel, median over the window
    # The pre-filter (see `PREFILTER_SIGMA`): `cv2.pyrDown` halvings, then a Gaussian of
    # `top_up_sigma` (pixels of the reduced image; 0 for none), then the remap through
    # `map_x_small`, `map_y_small`, the maps in the reduced image's coordinates.
    levels: int = 0
    top_up_sigma: float = 0.0
    map_x_small: np.ndarray | None = None
    map_y_small: np.ndarray | None = None

    @property
    def n_valid(self) -> int:
        return int(self.index.shape[0])

    def remap(self, image: np.ndarray) -> np.ndarray:
        """Resample a source image (2-D) into the window with bilinear interpolation."""
        if self.levels == 0 and self.top_up_sigma == 0.0:
            return cv2.remap(
                image,
                self.map_x,
                self.map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        small = image
        for _ in range(self.levels):
            small = cv2.pyrDown(small)
        if self.top_up_sigma > 0.0:
            small = cv2.GaussianBlur(small, (0, 0), self.top_up_sigma)
        return cv2.remap(
            small,
            self.map_x_small,
            self.map_y_small,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def prefilter_plan(decimation: float, sigma: float) -> tuple[int, float]:
    """Pyramid levels and top-up Gaussian (reduced-image px) for a blur of `sigma`.

    `sigma` is in units of the decimation. One `cv2.pyrDown` contributes a variance of one
    source pixel squared; the remainder of the target variance is a Gaussian on the
    half-size image. The pyramid's alignment is exact: output pixel `j` is centered on
    input pixel `2 j`, so a map in source pixel indices is divided by two.
    """
    if sigma <= 0.0 or decimation < PREFILTER_MIN_DECIMATION:
        return 0, 0.0
    target = (sigma * decimation) ** 2
    levels = 1 if target >= 1.0 else 0
    remainder = np.sqrt(max(target - levels, 0.0)) / 2.0**levels
    return levels, float(remainder) if remainder >= 0.3 else 0.0


def window_geometry(
    camera: Camera,
    centre,
    half_angle: float,
    size: int,
    mask: np.ndarray,
    prefilter: float = PREFILTER_SIGMA,
) -> WindowGeometry:
    """Build the tracking window for a ball at `centre` (unit) with `half_angle` (rad).

    The window frame has +z along `centre`; the ball centre sits at distance 1 and the
    ball radius is `sin(half_angle)`. `mask` is the source-image mask from `source_mask`.
    `prefilter` is the anti-aliasing blur in units of the decimation (0 for none).
    """
    c = normalize(np.asarray(centre, dtype=np.float64))
    to_camera = rotation_between(np.array([0.0, 0.0, 1.0]), c)
    window_cam = EquidistantCamera.from_extent(size, 2.0 * half_angle)
    xs, ys = pixel_centres(size, size)
    dirs_w = window_cam.rays(xs, ys)
    dirs_c = dirs_w @ to_camera.T
    x, y, valid = camera.project(dirs_c)
    map_x = (x - 0.5).astype(np.float32)
    map_y = (y - 0.5).astype(np.float32)

    radius = np.sin(half_angle)
    dz = dirs_w[..., 2]
    disc = dz * dz - (1.0 - radius * radius)
    hit = disc >= 0.0
    t = dz - np.sqrt(np.where(hit, disc, 0.0))
    surface = (t[..., None] * dirs_w - np.array([0.0, 0.0, 1.0])) / radius

    seen = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    window_mask = valid & hit & (seen > 0)
    index = np.flatnonzero(window_mask.ravel())
    step = np.hypot(np.gradient(map_x, axis=1), np.gradient(map_y, axis=1))
    decimation = float(np.median(step[window_mask])) if window_mask.any() else 1.0
    levels, top_up = prefilter_plan(decimation, prefilter)
    scale = np.float32(2.0**levels)
    return WindowGeometry(
        size=size,
        rad_per_pixel=window_cam.rad_per_pixel,
        to_camera=to_camera,
        map_x=map_x,
        map_y=map_y,
        mask=window_mask,
        surface=np.ascontiguousarray(surface.reshape(-1, 3)[index], dtype=np.float32),
        index=index,
        decimation=decimation,
        levels=levels,
        top_up_sigma=top_up,
        map_x_small=map_x / scale,
        map_y_small=map_y / scale,
    )
