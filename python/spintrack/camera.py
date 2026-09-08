"""Central camera models mapping between pixels and view directions.

Conventions (shared with FicTrac so its configs stay valid): camera axes x right, y
down, z forward; pixel coordinates are continuous with the centre of pixel `(i, j)` at
`(i + 0.5, j + 0.5)`. All methods are vectorized over leading dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spintrack.geometry import normalize


@dataclass(frozen=True)
class PinholeCamera:
    """Rectilinear lens described by the image size and vertical field of view (deg)."""

    width: int
    height: int
    vfov_deg: float

    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / np.tan(np.radians(self.vfov_deg) / 2.0)

    @property
    def centre(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    def rays(self, x, y) -> np.ndarray:
        """Unit view directions (..., 3) for continuous pixel coordinates."""
        cx, cy = self.centre
        f = self.focal_px
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        d = np.stack([(x - cx) / f, (y - cy) / f, np.ones_like(x)], axis=-1)
        return normalize(d)

    def project(self, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Directions (..., 3) -> continuous pixel coords `(x, y, valid)`."""
        v = np.asarray(v, dtype=np.float64)
        cx, cy = self.centre
        z = v[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = cx + self.focal_px * v[..., 0] / z
            y = cy + self.focal_px * v[..., 1] / z
        valid = (z > 0) & _inside(x, y, self.width, self.height)
        return x, y, valid


@dataclass(frozen=True)
class EquidistantCamera:
    """Ideal f-theta fisheye lens: image radius grows linearly with the view angle."""

    width: int
    height: int
    rad_per_pixel: float

    @classmethod
    def from_vfov(cls, width: int, height: int, vfov_deg: float) -> EquidistantCamera:
        """Fisheye whose image height spans `vfov_deg` degrees."""
        return cls(width, height, np.radians(vfov_deg) / height)

    @classmethod
    def from_extent(cls, size: int, extent_rad: float) -> EquidistantCamera:
        """Square fisheye whose side spans `extent_rad` radians."""
        return cls(size, size, extent_rad / size)

    @property
    def centre(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    def rays(self, x, y) -> np.ndarray:
        cx, cy = self.centre
        dx = np.asarray(x, dtype=np.float64) - cx
        dy = np.asarray(y, dtype=np.float64) - cy
        r = np.hypot(dx, dy)
        theta = r * self.rad_per_pixel
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r > 0, np.sin(theta) / r, 1.0)
        return np.stack([dx * scale, dy * scale, np.cos(theta)], axis=-1)

    def project(self, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = normalize(v)
        cx, cy = self.centre
        theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
        rxy = np.hypot(v[..., 0], v[..., 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rxy > 0, theta / self.rad_per_pixel / rxy, 0.0)
        x = cx + v[..., 0] * scale
        y = cy + v[..., 1] * scale
        return x, y, _inside(x, y, self.width, self.height)


Camera = PinholeCamera | EquidistantCamera


def _inside(x, y, width: int, height: int) -> np.ndarray:
    return (x >= 0) & (x <= width) & (y >= 0) & (y <= height)


def pixel_centres(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous coordinates `(x, y)` of every pixel centre, each of shape (h, w)."""
    ys, xs = np.mgrid[0:height, 0:width]
    return xs + 0.5, ys + 0.5


def source_camera(width: int, height: int, vfov_deg: float, fisheye: bool) -> Camera:
    """The camera model FicTrac configs describe with `vfov` and `fisheye`."""
    if fisheye:
        return EquidistantCamera.from_vfov(width, height, vfov_deg)
    return PinholeCamera(width, height, vfov_deg)
