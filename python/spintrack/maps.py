"""Save and load surface maps, including conversion of FicTrac sphere-map templates.

spintrack's native format is an `.npz` with `mean` and `weight` arrays (float32, shape
`(map_h, map_w)`) plus the window size they were built for. Version 3 also names the
projection, though the shape gives it away: an equal-area map is `(h, 2h)` and a cube
map `(6 face, face)`, so `projection_of` can read it off an array of either. Version 2
may also carry the static illumination fields (`illum_bias`, `illum_gain`, `illum_wt`,
shaped like the window): those describe the rig rather than the ball, and on the lab
recordings they come out near-identical from one trial to the next, so a run can start
from a measured field instead of learning it again. FicTrac templates are PNG images
whose pixels are 0 (dark), 255 (bright) or 128 (unseen) on an equal-area grid that is
mirrored both ways relative to spintrack's, so they are flipped and rescaled on import.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

FORMAT_VERSION = 3
ILLUM_KEYS = ("illum_bias", "illum_gain", "illum_wt")
UNSEEN = 128  # the mid-gray FicTrac uses for a tile it has never looked at
CONTRAST = 40.0  # normalized intensity per gray level, shared by every map rendering

# The six cube faces the map is stored on, as (forward, right, up): `forward` is the
# face center, `right` and `up` span it. The poles of the equal-area grid are at +/-y,
# so they land in the middle of two faces.
_FACES = {
    "-x": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "+z": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "+x": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "-z": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "+y": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "-y": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}
# The order the faces are stacked into a `(6 face, face)` array; `FACES` in `map.rs`.
_FACE_ORDER = ("-x", "+z", "+x", "-z", "+y", "-y")
# The unfolded dice `render_map` draws, as (row, col) -> (forward, right, up): a band of
# four faces around the window's horizontal great circle, with the top and bottom of the
# ball above and below the face the camera looks at. `right` runs along a tile's columns
# and `up` against its rows, so every tile is seen from outside the ball (right x up =
# forward), the middle tile is the near face the way the camera sees it (window frame: x
# right, y down, z away from the camera) and every seam of the net is a seam of the
# sphere. These are display frames, not the storage faces above.
_NET_TILES = {
    (0, 1): ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    (1, 0): ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
    (1, 1): ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    (1, 2): ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    (1, 3): ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
    (2, 1): ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
}
NET_SHAPE = (3, 4)  # tiles down, tiles across
NET_LABELS = {
    (0, 1): "top",
    (1, 0): "left",
    (1, 1): "near",
    (1, 2): "right",
    (1, 3): "far",
    (2, 1): "bottom",
}
_FORWARD = np.array([_FACES[f][0] for f in _FACE_ORDER], dtype=np.float64)
_RIGHT = np.array([_FACES[f][1] for f in _FACE_ORDER], dtype=np.float64)
_UP = np.array([_FACES[f][2] for f in _FACE_ORDER], dtype=np.float64)


def projection_of(shape: tuple[int, int]) -> str:
    """`"cube"` for a `(6 face, face)` stack of faces, `"equal_area"` otherwise.

    The two shapes cannot collide: an equal-area map is always twice as wide as tall.
    """
    h, w = shape
    return "cube" if h == 6 * w else "equal_area"


def map_directions(shape: tuple[int, int]) -> np.ndarray:
    """Unit vector of every map cell, in the ball's body frame, shaped `(h, w, 3)`.

    Inverse of the projection the solver uses: for an equal-area map, longitude about
    the window y axis and latitude by equal area, so every cell covers the same solid
    angle; for a cube map, the six faces of `cube_directions` stacked in `_FACE_ORDER`.
    """
    h, w = shape
    if projection_of(shape) == "cube":
        faces = cube_directions(w)
        return np.concatenate([faces[name] for name in _FACE_ORDER], axis=0)
    lon = (np.arange(w) + 0.5) / w * 2.0 * np.pi - np.pi
    y = 1.0 - 2.0 * (np.arange(h) + 0.5) / h
    lon, y = np.meshgrid(lon, y)
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    return np.stack([r * np.sin(lon), y, r * np.cos(lon)], axis=-1)


def save_map(path: str | Path, mean: np.ndarray, weight: np.ndarray, **meta) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    np.savez_compressed(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        format_version=FORMAT_VERSION,
        projection=projection_of(np.shape(mean)),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return path


def load_illumination(
    path: str | Path, size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """`(fields, geometry)` stored alongside a native map, if any and if they fit.

    The fields live in window pixels, so a map saved for a different window size cannot
    supply them; that is not an error, the run just learns its own. `geometry` carries
    the `center` and `half_angle` they were measured at, which the caller needs in order
    to resample them into its own window.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return {}, {}
    with np.load(path) as z:
        fields = {k: z[k].astype(np.float32) for k in ILLUM_KEYS if k in z.files}
        geometry = {k: z[k] for k in ("center", "half_angle") if k in z.files}
    fields = {
        k.removeprefix("illum_"): v
        for k, v in fields.items()
        if v.shape == (size, size)
    }
    return fields, geometry


def load_map(path: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load a native `.npz` map or a FicTrac PNG template, resampled to `shape` (h, w).

    The resampling goes through directions rather than pixels, so a map saved on one
    grid loads onto another - including a different projection, which is what lets a map
    from before the cube existed, or a FicTrac template, still be used.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            mean = z["mean"].astype(np.float32)
            weight = z["weight"].astype(np.float32)
        if mean.shape != tuple(shape):
            mean, weight = resample_map(mean, weight, shape)
        return np.ascontiguousarray(mean), np.ascontiguousarray(weight)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"could not read map template {path}")
    return fictrac_template_to_map(img, shape)


def fictrac_template_to_map(
    img: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a FicTrac sphere-map image to spintrack `(mean, weight)` arrays.

    FicTrac stores 128 for never-seen tiles and pushes seen tiles toward 0 or 255. Its
    grid runs the opposite way in both longitude and latitude, hence the double flip.
    Dark and bright tiles become -1 and +1 in normalized-intensity units, a coarse but
    usable prior. The conversion happens on the image's own equal-area grid; `shape` is
    reached from there by `resample_map`, the only way to land on a cube map.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flipped = img[::-1, ::-1].astype(np.float32)
    seen = flipped != UNSEEN
    mean = np.where(seen, np.clip((flipped - 128.0) / 127.0, -1.0, 1.0), 0.0).astype(
        np.float32
    )
    weight = seen.astype(np.float32)
    if mean.shape != tuple(shape):
        mean, weight = resample_map(mean, weight, shape)
    return np.ascontiguousarray(mean), np.ascontiguousarray(weight)


def _tap_cells(
    shape: tuple[int, int], dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The four cells a bilinear read along `dirs` draws on, flat, with their shares."""
    h, w = shape
    if projection_of(shape) == "cube":
        # The face `dirs` is closest to, then equi-angular coordinates on it. Taps that
        # fall off a face are clamped rather than followed across the seam: the solver
        # does follow them (`Map::cube_cell`), but this reader only ever resamples or
        # renders a map, where half a cell along twelve edges does not matter.
        n = w
        a = np.abs(dirs)
        face = np.where(
            (a[..., 0] >= a[..., 1]) & (a[..., 0] >= a[..., 2]),
            np.where(dirs[..., 0] > 0, 2, 0),
            np.where(
                a[..., 1] >= a[..., 2],
                np.where(dirs[..., 1] > 0, 4, 5),
                np.where(dirs[..., 2] > 0, 1, 3),
            ),
        )
        depth = np.sum(dirs * _FORWARD[face], axis=-1)
        s = (4.0 / np.pi) * np.arctan(np.sum(dirs * _RIGHT[face], axis=-1) / depth)
        t = (4.0 / np.pi) * np.arctan(np.sum(dirs * _UP[face], axis=-1) / depth)
        u = 0.5 * (s + 1.0) * n - 0.5
        v = 0.5 * (1.0 - t) * n - 0.5
        u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        base = face * n * n
        columns = [np.clip(u0 + d, 0, n - 1) for d in (0, 1)]
        rows = [base + n * np.clip(v0 + d, 0, n - 1) for d in (0, 1)]
    else:
        u = (np.arctan2(dirs[..., 0], dirs[..., 2]) + np.pi) * w / (2.0 * np.pi) - 0.5
        v = (1.0 - dirs[..., 1]) * 0.5 * h - 0.5
        u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        columns = [(u0 + d) % w for d in (0, 1)]
        rows = [w * np.clip(v0 + d, 0, h - 1) for d in (0, 1)]
    fu, fv = u - u0, v - v0
    flat = np.stack([r + c for r in rows for c in columns], axis=0)
    share = np.stack([
        (1.0 - fu) * (1.0 - fv), fu * (1.0 - fv), (1.0 - fu) * fv, fu * fv
    ])  # fmt: skip
    return flat, share


def _taps(
    mean: np.ndarray, weight: np.ndarray, dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_tap_cells` plus the weights of the cells it lands on."""
    flat, share = _tap_cells(mean.shape, dirs)
    return flat, share, weight.reshape(-1)[flat]


def sample_map(
    mean: np.ndarray, weight: np.ndarray, dirs: np.ndarray, w_min: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Read the map along `dirs`, mirroring `Map::sample` in `rust/src/map.rs`.

    Plain bilinear in `mean`, so any array on the same grid can be passed in its place;
    a direction counts as seen only when all four cells it draws on are, which is the
    rule the solver uses to decide a pixel has anything to match against.
    """
    flat, share, taps = _taps(mean, weight, dirs)
    value = np.einsum("k...,k...->...", share, mean.reshape(-1)[flat])
    return value.astype(np.float32), np.all(taps >= w_min, axis=0)


def resample_map(
    mean: np.ndarray, weight: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Move a map onto a different grid, which may be a different projection.

    Weighted like `Map::coarse`, `sum(m w) / sum(w)`, so a cell straddling the frontier
    of what has been seen takes the value of the seen side instead of being dragged
    toward zero by its unseen neighbors. Unseen stays unseen: the weight carries over
    bilinearly.
    """
    flat, share, taps = _taps(mean, weight, _directions(tuple(shape)))
    total = np.einsum("k...,k...->...", share, taps)
    value = np.einsum("k...,k...,k...->...", share, taps, mean.reshape(-1)[flat])
    value = np.where(total > 1e-6, value / np.maximum(total, 1e-6), 0.0)
    return (
        np.ascontiguousarray(value, dtype=np.float32),
        np.ascontiguousarray(total, dtype=np.float32),
    )


def _tile_directions(face: int, forward, right, up) -> np.ndarray:
    """Unit vector of every texel of one equi-angular face, `(face, face, 3)`.

    Equi-angular (`s' = tan(pi s / 4)`) rather than the plain gnomonic `s'= s`: it costs
    one `tan` and brings the solid angle per texel from a 5.2:1 spread between face
    center and corner down to 1.41:1, with near-square texels throughout.
    """
    s = np.tan(0.25 * np.pi * (2.0 * (np.arange(face) + 0.5) / face - 1.0))
    right_s, up_s = np.meshgrid(s, -s)
    v = (
        np.asarray(forward, dtype=np.float64)
        + right_s[..., None] * np.asarray(right, dtype=np.float64)
        + up_s[..., None] * np.asarray(up, dtype=np.float64)
    )
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cube_directions(face: int) -> dict[str, np.ndarray]:
    """Unit vector of every texel of the stored equi-angular cube, one `(face, face, 3)`
    array per face."""
    return {name: _tile_directions(face, *frame) for name, frame in _FACES.items()}


@lru_cache(maxsize=8)
def _directions(shape: tuple[int, int]) -> np.ndarray:
    """`map_directions` memoized. The result is read-only because callers share it."""
    dirs = map_directions(shape)
    dirs.flags.writeable = False
    return dirs


@lru_cache(maxsize=8)
def _render_plan(shape: tuple[int, int], layout: str, face: int | None):
    """Where a picture of a map shaped `shape` reads the map: the bilinear taps of every
    output pixel, and for the cube net the `(row, col)` tile each block of them fills.

    Cached because it depends on the grid and the layout, not on the map, and the debug
    video draws the map on every frame.
    """
    if layout == "grid":
        h = round(np.sqrt(shape[0] * shape[1] / 2.0))
        dirs, tiles = _directions((h, 2 * h)), None
    else:
        tiles = tuple(_NET_TILES)
        dirs = np.stack([_tile_directions(face, *_NET_TILES[rc]) for rc in tiles])
    flat, share = _tap_cells(shape, dirs)
    flat.flags.writeable = False
    share.flags.writeable = False
    return flat, share, tiles


def _face_size(shape: tuple[int, int]) -> int:
    """Cube face side that matches the cell count of a map shaped `shape`."""
    h, w = shape
    return w if projection_of(shape) == "cube" else max(round(np.sqrt(h * w / 6.0)), 1)


def render_map(
    mean: np.ndarray,
    weight: np.ndarray,
    w_min: float = 0.1,
    layout: str = "grid",
    face: int | None = None,
) -> np.ndarray:
    """uint8 picture of a surface map, unseen cells mid-gray.

    The layout is the projection to draw in, not the one the map is stored in: `grid` is
    a Lambert equal-area rectangle, on the same scale as FicTrac's sphere-map PNGs, and
    `cube` is an unfolded dice (`NET_LABELS` names its tiles) centered on the face the
    camera looks at and oriented like the image, so at `R = I` its middle tile is the
    tracking window. A map is resampled if it is not already on that grid. The cube is
    the honest way to look at the poles of an equal-area map: a single row of 0.1-degree
    slivers in the rectangle, a square face here.
    """
    if layout not in ("grid", "cube"):
        raise ValueError(f"unknown map layout {layout!r}")
    if layout == "grid" and projection_of(mean.shape) == "equal_area":
        value, seen = np.asarray(mean, dtype=np.float32), weight >= w_min
    else:
        if layout == "cube":
            face = int(face) if face is not None else _face_size(mean.shape)
        flat, share, tiles = _render_plan(tuple(mean.shape), layout, face)
        value = np.einsum("k...,k...->...", share, mean.reshape(-1)[flat])
        seen = np.all(weight.reshape(-1)[flat] >= w_min, axis=0)
        if tiles is not None:
            # The used tiles into the 4x3 net; the six unused ones stay unseen.
            rows, cols = NET_SHAPE
            net_value = np.zeros((rows * face, cols * face), np.float32)
            net_seen = np.zeros(net_value.shape, dtype=bool)
            for k, (r, c) in enumerate(tiles):
                tile = (
                    slice(r * face, (r + 1) * face),
                    slice(c * face, (c + 1) * face),
                )
                net_value[tile], net_seen[tile] = value[k], seen[k]
            value, seen = net_value, net_seen
    img = np.clip(UNSEEN + CONTRAST * value, 0, 255).astype(np.uint8)
    img[~seen] = UNSEEN
    return img
