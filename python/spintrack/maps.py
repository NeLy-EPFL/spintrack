"""Save and load surface maps, including conversion of FicTrac sphere-map templates.

spintrack's native format is an `.npz` with `mean` and `weight` arrays (float32, shape
`(map_h, map_w)`) plus the window size they were built for. Version 2 may also carry the
static illumination fields (`illum_bias`, `illum_gain`, `illum_wt`, shaped like the
window): those describe the rig rather than the ball, and on the lab recordings they come
out near-identical from one trial to the next, so a run can start from a measured field
instead of learning it again. FicTrac templates are PNG images whose pixels are 0 (dark),
255 (bright) or 128 (unseen) on an equal-area grid that is mirrored both ways relative to
spintrack's, so they are flipped and rescaled on import.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FORMAT_VERSION = 2
ILLUM_KEYS = ("illum_bias", "illum_gain", "illum_wt")


def map_directions(shape: tuple[int, int]) -> np.ndarray:
    """Unit vector of every map cell, in the ball's body frame, shaped `(h, w, 3)`.

    Inverse of the projection the solver uses: longitude about the window y axis, and
    latitude by equal area, so every cell covers the same solid angle.
    """
    h, w = shape
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
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return path


def load_illumination(
    path: str | Path, size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """`(fields, geometry)` stored alongside a native map, if any and if they fit.

    The fields live in window pixels, so a map saved for a different window size cannot
    supply them; that is not an error, the run just learns its own. `geometry` carries the
    `centre` and `half_angle` they were measured at, which the caller needs in order to
    resample them into its own window.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return {}, {}
    with np.load(path) as z:
        fields = {k: z[k].astype(np.float32) for k in ILLUM_KEYS if k in z.files}
        geometry = {k: z[k] for k in ("centre", "half_angle") if k in z.files}
    fields = {
        k.removeprefix("illum_"): v
        for k, v in fields.items()
        if v.shape == (size, size)
    }
    return fields, geometry


def load_map(path: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load a native `.npz` map or a FicTrac PNG template, resampled to `shape` (h, w)."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            mean = z["mean"].astype(np.float32)
            weight = z["weight"].astype(np.float32)
        if mean.shape != tuple(shape):
            mean = cv2.resize(
                mean, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
            )
            weight = cv2.resize(
                weight, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return np.ascontiguousarray(mean), np.ascontiguousarray(weight)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"could not read map template {path}")
    return fictrac_template_to_map(img, shape)


def fictrac_template_to_map(
    img: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a FicTrac sphere-map image to spintrack `(mean, weight)` arrays.

    FicTrac stores 128 for never-seen tiles and pushes seen tiles toward 0 or 255. Its grid
    runs the opposite way in both longitude and latitude, hence the double flip. Dark and
    bright tiles become -1 and +1 in normalized-intensity units, a coarse but usable prior.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != tuple(shape):
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    flipped = img[::-1, ::-1].astype(np.float32)
    seen = flipped != 128
    mean = np.where(seen, np.clip((flipped - 128.0) / 127.0, -1.0, 1.0), 0.0).astype(
        np.float32
    )
    weight = seen.astype(np.float32)
    return np.ascontiguousarray(mean), np.ascontiguousarray(weight)
