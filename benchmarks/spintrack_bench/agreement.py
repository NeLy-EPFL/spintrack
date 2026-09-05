"""Agreement between two trackers on a real video, where no ground truth exists.

Compares per-frame lab-frame rotation increments (what downstream analyses consume), the
integrated heading and the fictive path. Frames missing from either output are skipped.
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
    )
