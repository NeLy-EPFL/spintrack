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

    @property
    def n_valid(self) -> int:
        return int(self.index.shape[0])

    def remap(self, image: np.ndarray) -> np.ndarray:
        """Resample a source image (2-D) into the window with bilinear interpolation."""
        return cv2.remap(
            image,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def window_geometry(
    camera: Camera, centre, half_angle: float, size: int, mask: np.ndarray
) -> WindowGeometry:
    """Build the tracking window for a ball at `centre` (unit) with `half_angle` (rad).

    The window frame has +z along `centre`; the ball centre sits at distance 1 and the
    ball radius is `sin(half_angle)`. `mask` is the source-image mask from `source_mask`.
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
    return WindowGeometry(
        size=size,
        rad_per_pixel=window_cam.rad_per_pixel,
        to_camera=to_camera,
        map_x=map_x,
        map_y=map_y,
        mask=window_mask,
        surface=np.ascontiguousarray(surface.reshape(-1, 3)[index], dtype=np.float32),
        index=index,
    )
