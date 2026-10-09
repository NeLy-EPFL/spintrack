"""Accuracy metrics for per-frame rotation estimates against ground truth.

Estimates and truth are (N, 3) rotation vectors of the per-frame ball rotation in the
same frame; estimates may contain NaN rows for frames the tracker did not report.

The angle metrics (`median_deg` and friends) mix a scale error with random disagreement:
a tracker reporting every rotation 2% too small scores the same as one that is right on
average and noisy. `component_gains` separates them by regressing the estimate on truth,
which is noiseless here, so the slope is the factor the tracker applies to each reported
component and the residual is what is left over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spintrack.calibrate.sliders import FICTRAC_TO_ANIMAL
from spintrack.geometry import rotvec_to_matrix
from spintrack.path import integrate_path


def rotvecs_to_matrices(w: np.ndarray) -> np.ndarray:
    """Batched Rodrigues formula, (N, 3) -> (N, 3, 3)."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w, axis=1)
    k = np.zeros((len(w), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -w[:, 2], w[:, 1]
    k[:, 1, 0], k[:, 1, 2] = w[:, 2], -w[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -w[:, 1], w[:, 0]
    small = theta < 1e-4
    t2 = theta * theta
    a = np.where(small, 1 - t2 / 6, np.sin(theta) / np.where(small, 1, theta))
    b = np.where(small, 0.5 - t2 / 24, (1 - np.cos(theta)) / np.where(small, 1, t2))
    return np.eye(3)[None] + a[:, None, None] * k + b[:, None, None] * (k @ k)


def rotation_angles(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1))


def frame_errors_deg(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Angle (deg) of `exp(est) exp(truth)^T` per frame; NaN where `est` is missing."""
    out = np.full(len(truth), np.nan)
    ok = np.all(np.isfinite(est), axis=1)
    if ok.any():
        Re = rotvecs_to_matrices(est[ok])
        Rt = rotvecs_to_matrices(truth[ok])
        out[ok] = np.degrees(rotation_angles(Re @ np.swapaxes(Rt, 1, 2)))
    return out


def accumulated_error_deg(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Orientation error (deg) after composing all increments.

    Missing frames add nothing.
    """
    R_est = np.eye(3)
    R_true = np.eye(3)
    out = np.empty(len(truth))
    for i in range(len(truth)):
        R_true = rotvec_to_matrix(truth[i]) @ R_true
        if np.all(np.isfinite(est[i])):
            R_est = rotvec_to_matrix(est[i]) @ R_est
        out[i] = np.degrees(rotation_angles((R_est @ R_true.T)[None])[0])
    return out


# A fictive path shorter than this over a whole recording is a turn in place, and an
# endpoint error as a percentage of it says nothing: `constant_turn` sums 8e-16 rad of
# translation and used to report 2.1e14%, while every scene that goes anywhere sums at
# least 4.7 rad.
MIN_PATH_RAD = 1e-3

# Moving-block bootstrap for the standard error of a gain. Tracking residuals are
# strongly autocorrelated - a hard stretch is hard for many frames together - so the
# textbook regression standard error understates the uncertainty by two or three times,
# which is the difference between "sideslip reads 2% low" and "sideslip is consistent
# with 1.00". The blocks are long enough to carry that correlation.
GAIN_BLOCK = 50  # frames per block
GAIN_RESAMPLES = 200
# A component a scene does not turn about is not zero after the round trip through the
# camera frame, it is 1e-18 rad of floating-point residue, and dividing by its square
# gives a gain of 1e13. Both floors have to be cleared: an absolute one, and a share of
# the largest component, so that "small but real" and "not there" stay apart.
GAIN_MIN_RAD = 1e-6
GAIN_MIN_SHARE = 1e-3


def component_gains(est: np.ndarray, truth: np.ndarray, seed: int = 0) -> tuple:
    """Per-component `est = gain * truth` least squares, and its bootstrap error.

    Returns `(gain, standard_error)`, both (3,), NaN for a component the scene does not
    excite. Each component is fitted on its own rather than as one 3x3 map: the map's
    diagonal trades off against its off-diagonal entries, which for a motion whose
    components are correlated hides a per-component scale error in the cross terms.
    """
    ok = np.all(np.isfinite(est), axis=1)
    e, t = est[ok], truth[ok]
    gain = np.full(3, np.nan)
    err = np.full(3, np.nan)
    if len(e) < 3 * GAIN_BLOCK:
        return gain, err
    rng = np.random.default_rng(seed)
    n_blocks = len(e) // GAIN_BLOCK
    starts = rng.integers(0, len(e) - GAIN_BLOCK + 1, (GAIN_RESAMPLES, n_blocks))
    index = (starts[..., None] + np.arange(GAIN_BLOCK)).reshape(GAIN_RESAMPLES, -1)
    rms = np.sqrt((t**2).mean(axis=0))
    for i in range(3):
        if rms[i] < max(GAIN_MIN_RAD, GAIN_MIN_SHARE * rms.max()):
            continue
        denominator = float((t[:, i] ** 2).sum())
        gain[i] = float((e[:, i] * t[:, i]).sum() / denominator)
        ei, ti = e[index, i], t[index, i]
        draws = (ei * ti).sum(axis=1) / np.maximum((ti * ti).sum(axis=1), 1e-30)
        err[i] = float(draws.std())
    return gain, err


@dataclass
class Summary:
    n_frames: int
    reported_frac: float
    median_deg: float
    p95_deg: float
    rms_deg: float
    fail_frac: float  # frames missing or with error > 5 deg
    bias_deg: tuple[float, float, float]  # mean signed per-axis error, degrees
    speed_rel_err: (
        float  # median relative error of |w|, on frames with |w_true| > 0.005
    )
    acc_final_deg: float
    acc_max_deg: float
    drift_deg_per_min: float
    heading_drift_deg: float
    # World-frame endpoint error as % of true path length; NaN when the true path is
    # shorter than `MIN_PATH_RAD`, where the percentage is meaningless.
    endpoint_err_pct: float
    path_len_rad: float
    # Absolute scale, which the angle metrics above cannot see. `scale` is the factor
    # the estimate applies to the truth, pooled over all three camera-frame components;
    # `scale_path` is the same factor as it reaches the integrated path, so a purely
    # temporal effect (an exposure that averages the motion over the frame) pulls
    # `scale` below 1 and leaves `scale_path` at 1. `gain_*` are per-component factors
    # in the animal frame with their bootstrap standard errors (`component_gains`).
    scale: float
    scale_path: float
    gain_side: float
    gain_forward: float
    gain_turn: float
    gain_se_side: float
    gain_se_forward: float
    gain_se_turn: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bias_x_deg"], d["bias_y_deg"], d["bias_z_deg"] = d.pop("bias_deg")
        return d


def summarize(
    est_cam: np.ndarray,
    true_cam: np.ndarray,
    cam_to_lab: np.ndarray,
    fps: float,
    fail_deg: float = 5.0,
) -> Summary:
    # The scenes' lab frame is FicTrac's; spintrack's path integrates its own.
    cam_to_lab = FICTRAC_TO_ANIMAL @ np.asarray(cam_to_lab, dtype=np.float64)
    n = len(true_cam)
    errors = frame_errors_deg(est_cam, true_cam)
    present = np.isfinite(errors)
    valid = errors[present]
    acc = accumulated_error_deg(est_cam, true_cam)
    t_min = np.arange(n) / fps / 60.0
    slope = np.polyfit(t_min, acc, 1)[0] if n > 1 else 0.0
    diff = np.degrees(np.where(present[:, None], est_cam - true_cam, np.nan))
    speed_true = np.linalg.norm(true_cam, axis=1)
    moving = present & (speed_true > 0.005)
    speed_est = np.linalg.norm(np.where(present[:, None], est_cam, 0.0), axis=1)
    rel = np.abs(speed_est[moving] - speed_true[moving]) / speed_true[moving]

    est_lab = np.where(present[:, None], est_cam, np.nan) @ cam_to_lab.T
    gain, gain_se = component_gains(est_lab, true_cam @ cam_to_lab.T)
    moved = present[:, None] & (np.abs(true_cam) > 0)
    pooled = np.where(moved, est_cam, 0.0), np.where(moved, true_cam, 0.0)
    denominator = float((pooled[1] ** 2).sum())
    sum_true = pooled[1].sum(axis=0)

    est_filled = np.where(present[:, None], est_cam, 0.0)
    path_est = integrate_path(est_filled @ cam_to_lab.T)
    path_true = integrate_path(true_cam @ cam_to_lab.T)
    steps = np.hypot(true_cam @ cam_to_lab.T[:, 1], true_cam @ cam_to_lab.T[:, 0])
    path_len = float(steps.sum())
    endpoint = float(np.hypot(*(path_est[-1, :2] - path_true[-1, :2])))
    dh = (path_est[-1, 2] - path_true[-1, 2] + np.pi) % (2 * np.pi) - np.pi
    return Summary(
        n_frames=n,
        reported_frac=float(present.mean()),
        median_deg=float(np.median(valid)) if valid.size else np.nan,
        p95_deg=float(np.percentile(valid, 95)) if valid.size else np.nan,
        rms_deg=float(np.sqrt(np.mean(valid**2))) if valid.size else np.nan,
        fail_frac=float(
            np.mean(~present | (np.nan_to_num(errors, nan=np.inf) > fail_deg))
        ),
        bias_deg=tuple(float(v) for v in np.nanmean(diff, axis=0)),
        speed_rel_err=float(np.median(rel)) if rel.size else np.nan,
        acc_final_deg=float(acc[-1]),
        acc_max_deg=float(acc.max()),
        drift_deg_per_min=float(slope),
        heading_drift_deg=float(np.degrees(dh)),
        endpoint_err_pct=(
            100.0 * endpoint / path_len if path_len > MIN_PATH_RAD else np.nan
        ),
        path_len_rad=path_len,
        scale=float((pooled[0] * pooled[1]).sum() / denominator)
        if denominator > 0
        else np.nan,
        scale_path=float(
            np.linalg.norm(pooled[0].sum(axis=0)) / np.linalg.norm(sum_true)
        )
        if np.linalg.norm(sum_true) > 1e-9
        else np.nan,
        gain_side=gain[0],
        gain_forward=gain[1],
        gain_turn=gain[2],
        gain_se_side=gain_se[0],
        gain_se_forward=gain_se[1],
        gain_se_turn=gain_se[2],
    )
