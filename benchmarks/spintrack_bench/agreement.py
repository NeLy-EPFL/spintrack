"""Agreement between two trackers on a real video, where no ground truth exists.

Compares per-frame lab-frame rotation increments (what downstream analyses consume), the
integrated heading and the fictive path. Frames missing from either output are skipped.

Correlation is not the whole story - two trackers whose speeds differ by a constant 5%
correlate at 1.000 - so `scale_*` compares amplitudes (`block_scale`). FicTrac's
lab-frame columns must be put in the true lab frame first (`fictrac_lab_frame`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spintrack.path import integrate_path
from spintrack_bench.metrics import frame_errors_deg


@dataclass
class Agreement:
    n_common: int
    missing_a: int
    missing_b: int
    median_deg: float  # per-frame angle between the two lab-frame increments
    p95_deg: float
    speed_corr: float  # Pearson correlation of |w| per frame
    forward_corr: float  # correlation of the forward component (dr_lab y)
    turn_corr: float  # correlation of the turning component (dr_lab z)
    heading_diff_deg: float  # final heading difference, wrapped
    endpoint_diff_pct: float  # endpoint distance / mean path length
    # Scale of A relative to B, per component (`block_scale`).
    scale_forward: float
    scale_turn: float
    scale_side: float

    def as_dict(self) -> dict:
        return asdict(self)


def align(
    dat_a: np.ndarray, dat_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Rows of both `.dat` arrays for frames present in both (matched on column 0)."""
    fa = dat_a[:, 0].astype(int)
    fb = dat_b[:, 0].astype(int)
    common = np.intersect1d(fa, fb)
    ia = np.searchsorted(fa, common)
    ib = np.searchsorted(fb, common)
    return dat_a[ia], dat_b[ib], len(fa) - len(common), len(fb) - len(common)


# Frames per block for `block_scale`: 1 s at 100 fps.
SCALE_BLOCK = 100


def block_scale(a: np.ndarray, b: np.ndarray, block: int = SCALE_BLOCK) -> float:
    """`a` as a multiple of `b`, from their sums over `block`-frame blocks.

    Summing cancels errors that only move motion between neighboring frames (FicTrac
    smears each frame's rotation into the next), and regressing `b` on `a` assumes `a`
    is the far less noisy series (spintrack's per-frame error is a tenth of FicTrac's),
    so the regression is not diluted by `b`'s noise.
    """
    n = len(a) // block * block
    if n < 10 * block:
        return np.nan
    sa = np.asarray(a[:n], dtype=np.float64).reshape(-1, block).sum(axis=1)
    sb = np.asarray(b[:n], dtype=np.float64).reshape(-1, block).sum(axis=1)
    cross = float(sa @ sb)
    return float(sa @ sa) / cross if cross > 0 else np.nan


def compare(dat_a: np.ndarray, dat_b: np.ndarray) -> Agreement:
    a, b, miss_a, miss_b = align(dat_a, dat_b)
    wa = a[:, 5:8]
    wb = b[:, 5:8]
    err = frame_errors_deg(wa, wb)
    sa = np.linalg.norm(wa, axis=1)
    sb = np.linalg.norm(wb, axis=1)

    def corr(x, y):
        return (
            float(np.corrcoef(x, y)[0, 1])
            if len(x) > 2 and x.std() > 0 and y.std() > 0
            else np.nan
        )

    pa = integrate_path(wa)
    pb = integrate_path(wb)
    dh = (pa[-1, 2] - pb[-1, 2] + np.pi) % (2 * np.pi) - np.pi
    length = 0.5 * (
        np.hypot(wa[:, 1], wa[:, 0]).sum() + np.hypot(wb[:, 1], wb[:, 0]).sum()
    )
    endpoint = float(np.hypot(*(pa[-1, :2] - pb[-1, :2])))
    return Agreement(
        n_common=len(a),
        missing_a=miss_a,
        missing_b=miss_b,
        median_deg=float(np.nanmedian(err)),
        p95_deg=float(np.nanpercentile(err, 95)),
        speed_corr=corr(sa, sb),
        forward_corr=corr(wa[:, 1], wb[:, 1]),
        turn_corr=corr(wa[:, 2], wb[:, 2]),
        heading_diff_deg=float(np.degrees(dh)),
        endpoint_diff_pct=100.0 * endpoint / length if length > 0 else np.nan,
        scale_forward=block_scale(wa[:, 1], wb[:, 1]),
        scale_turn=block_scale(wa[:, 2], wb[:, 2]),
        scale_side=block_scale(wa[:, 0], wb[:, 0]),
    )
