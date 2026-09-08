"""Agreement between two trackers on a real video, where no ground truth exists.

Compares per-frame lab-frame rotation increments (what downstream analyses consume), the
integrated heading and the fictive path. Frames missing from either output are skipped.

Correlation is deliberately not the whole story: two trackers whose reported speeds
differ by a constant 5% correlate at 1.000. `scale_*` is the ratio of the two, which is
the one comparison that bears on absolute calibration - the trackers are independent
implementations of the same geometry reading the same `config.txt`, so a ratio of 1 says
the disagreement is not in the code, and whatever is left is common to both (the ball
circle in the config, the camera model matching the real lens).

Measuring that ratio needs care, because both series are noisy estimates of the same
rotation and their noise is nowhere near equal - FicTrac's per-frame error is an order
of magnitude larger. Regressing either on the other is pulled toward zero by the noise
in the regressor, and the ratio of their standard deviations is pulled toward zero by
the noisier one's extra variance: on the synthetic scenes, where each system's true
gain is known, that ratio reads 0.78 to 0.99 where the answer is 1.00. `scale_ratio`
uses the lagged instrument instead (see the function), which reads those same scenes to
within 0.3%.
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
    # Scale of A relative to B, per component (`scale_ratio`); NaN where the component
    # does not vary enough to instrument.
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


SCALE_LAG = (
    3  # frames; measured over 8 synthetic scenes, lag 1 to 8 all read within 0.5%
)
SCALE_MIN_ACOV = 0.05  # share of the variance the lagged autocovariance has to carry


def _lagged(x: np.ndarray, y: np.ndarray, k: int) -> float:
    return float(((x[:-k] - x.mean()) * (y[k:] - y.mean())).mean())


def scale_ratio(a: np.ndarray, b: np.ndarray, k: int = SCALE_LAG) -> float:
    """`a` as a multiple of `b`, using each series' own past as the instrument.

    With `a = alpha s + e_a` and `b = beta s + e_b` over the same unknown rotation `s`
    and independent white errors, the errors contribute nothing at any nonzero lag, so
    `cov(a_t, a_t+k) = alpha^2 c_s(k)` and `cov(a_t, b_t+k) = alpha beta c_s(k)`, whose
    ratio is `alpha / beta` however large either error is. The cross term is symmetrized
    over the two orderings, since neither series leads the other.

    Returns NaN unless both series carry real low-frequency signal: a nearly constant or
    nearly white component leaves `c_s(k)` at zero and the ratio is then noise over
    noise (on `constant_forward`, whose truth never changes, it reads 0.03 to 0.26).

    Unbiased is not the same as accurate. The cross term is itself estimated, and a very
    noisy `b` makes it noisy, which biases the ratio up: on a planted series whose `b`
    carries two and a half times the signal's own amplitude in noise, ratios of 0.80,
    1.00 and 1.25 read 0.82, 1.05 and 1.34. It was validated at the noise these two
    trackers actually have - FicTrac's per-frame error is an order of magnitude larger
    than spintrack's but well under the signal - where it reads the synthetic scenes'
    known ratios within 0.3%. Treat a trial whose correlations are poor with the same
    suspicion you would treat its other columns.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 4 * k:
        return np.nan
    var_a, var_b = a.var(), b.var()
    acov_a, acov_b = _lagged(a, a, k), _lagged(b, b, k)
    floor = SCALE_MIN_ACOV
    if min(var_a, var_b) <= 0 or acov_a < floor * var_a or acov_b < floor * var_b:
        return np.nan
    cross = 0.5 * (_lagged(a, b, k) + _lagged(b, a, k))
    return acov_a / cross if cross > 0 else np.nan


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
        scale_forward=scale_ratio(wa[:, 1], wb[:, 1]),
        scale_turn=scale_ratio(wa[:, 2], wb[:, 2]),
        scale_side=scale_ratio(wa[:, 0], wb[:, 0]),
    )
