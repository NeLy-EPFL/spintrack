"""Albedo textures on the unit sphere, stored as equirectangular maps.

A texture is painted once into a float32 map and then sampled per frame with bilinear
interpolation; `Texture.sample(dirs)` takes unit directions in the ball's body frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sample_sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def poisson_disk_sphere(
    rng: np.random.Generator, n: int, min_sep: float, max_tries: int = 20000
) -> np.ndarray:
    """Up to `n` unit vectors at least `min_sep` radians apart (dart throwing)."""
    kept: list[np.ndarray] = []
    cos_sep = np.cos(min_sep)
    for cand in sample_sphere(rng, max_tries):
        if not kept or np.max(np.asarray(kept) @ cand) < cos_sep:
            kept.append(cand)
            if len(kept) == n:
                break
    return np.asarray(kept)


def dirs_to_lonlat(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longitude in (-pi, pi] about +y and latitude in [-pi/2, pi/2]; y is the pole axis."""
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    return lon, lat


def lonlat_to_dirs(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon, lat = np.broadcast_arrays(lon, lat)
    c = np.cos(lat)
    return np.stack([c * np.sin(lon), np.sin(lat), c * np.cos(lon)], axis=-1)


@dataclass
class Texture:
    """Equirectangular albedo map in [0, 1] with bilinear, longitude-wrapping sampling."""

    data: np.ndarray  # (H, W) float32

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.data.shape
        lon = (np.arange(w) + 0.5) / w * 2.0 * np.pi - np.pi
        lat = (np.arange(h) + 0.5) / h * np.pi - np.pi / 2.0
        return lon, lat

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        h, w = self.data.shape
        lon, lat = dirs_to_lonlat(dirs)
        u = (lon + np.pi) / (2.0 * np.pi) * w - 0.5
        v = (lat + np.pi / 2.0) / np.pi * h - 0.5
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        fu = (u - u0).astype(np.float32)
        fv = (v - v0).astype(np.float32)
        u1 = (u0 + 1) % w
        u0 %= w
        v1 = np.clip(v0 + 1, 0, h - 1)
        v0 = np.clip(v0, 0, h - 1)
        d = self.data
        top = d[v0, u0] * (1 - fu) + d[v0, u1] * fu
        bot = d[v1, u0] * (1 - fu) + d[v1, u1] * fu
        return top * (1 - fv) + bot * fv

    def paint_cap(
        self, centre: np.ndarray, radius: float, darkness: float, soft: float
    ):
        """Multiply the albedo by `1 - darkness` inside a soft-edged spherical cap."""
        _h, w = self.data.shape
        lon_g, lat_g = self._grid()
        lon_c, lat_c = dirs_to_lonlat(centre)
        outer = radius + soft
        lat_lo = max(lat_c - outer, -np.pi / 2)
        lat_hi = min(lat_c + outer, np.pi / 2)
        rows = np.flatnonzero((lat_g >= lat_lo) & (lat_g <= lat_hi))
        if rows.size == 0:
            return
        if abs(lat_c) + outer >= np.pi / 2 - 1e-6:
            cols = np.arange(w)
        else:
            half_lon = outer / np.cos(min(abs(lat_c) + outer, np.pi / 2 - 1e-6))
            dlon = (lon_g - lon_c + np.pi) % (2 * np.pi) - np.pi
            cols = np.flatnonzero(np.abs(dlon) <= half_lon)
        if cols.size == 0:
            return
        sub = lonlat_to_dirs(lon_g[cols][None, :], lat_g[rows][:, None])
        theta = np.arccos(np.clip(sub @ centre, -1.0, 1.0))
        # 1 inside radius - soft, 0 outside radius + soft, smooth in between.
        x = np.clip((radius + soft - theta) / (2.0 * soft), 0.0, 1.0)
        weight = x * x * (3.0 - 2.0 * x)
        self.data[np.ix_(rows, cols)] *= (1.0 - darkness * weight).astype(np.float32)


@dataclass
class BlobTextureSpec:
    """Dark spots on a bright ball, the usual hand-painted foam trackball."""

    kind: str = "blobs"
    n_blobs: int = 150
    radius_deg: tuple[float, float] = (4.0, 10.0)
    darkness: tuple[float, float] = (0.8, 1.0)
    soft_deg: float = 0.8
    base_albedo: float = 0.9
    shading_blobs: int = 12  # large faint patches giving the base a mottled look
    shading_darkness: float = 0.12
    map_size: tuple[int, int] = field(default=(1024, 2048))

    def build(self, rng: np.random.Generator) -> Texture:
        tex = Texture(np.full(self.map_size, self.base_albedo, np.float32))
        for c in sample_sphere(rng, self.shading_blobs):
            tex.paint_cap(
                c, np.radians(rng.uniform(15, 40)), self.shading_darkness, 0.3
            )
        r_lo, r_hi = np.radians(self.radius_deg)
        centres = poisson_disk_sphere(rng, self.n_blobs, 0.9 * (r_lo + r_hi))
        for c in centres:
            radius = rng.uniform(r_lo, r_hi)
            darkness = rng.uniform(*self.darkness)
            tex.paint_cap(c, radius, darkness, np.radians(self.soft_deg))
        return tex


TEXTURE_PRESETS: dict[str, BlobTextureSpec] = {
    "blobs": BlobTextureSpec(),
    "low_contrast": BlobTextureSpec(darkness=(0.25, 0.35), base_albedo=0.7),
    "speckle": BlobTextureSpec(n_blobs=1500, radius_deg=(1.0, 2.5), soft_deg=0.4),
    "sparse": BlobTextureSpec(n_blobs=40, radius_deg=(8.0, 16.0)),
}
