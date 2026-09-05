"""Save and load surface maps, including conversion of FicTrac sphere-map templates.

spintrack's native format is an `.npz` with `mean` and `weight` arrays (float32, shape
`(map_h, map_w)`) plus the window size they were built for. FicTrac templates are PNG images
whose pixels are 0 (dark), 255 (bright) or 128 (unseen) on an equal-area grid that is
mirrored both ways relative to spintrack's, so they are flipped and rescaled on import.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FORMAT_VERSION = 1


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
