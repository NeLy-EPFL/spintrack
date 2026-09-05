"""Accuracy metrics for per-frame rotation estimates against ground truth.

Estimates and truth are (N, 3) rotation vectors of the per-frame ball rotation in the same
frame; estimates may contain NaN rows for frames the tracker did not report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

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
    """Orientation error (deg) after composing all increments; missing frames add nothing."""
    R_est = np.eye(3)
    R_true = np.eye(3)
    out = np.empty(len(truth))
    for i in range(len(truth)):
        R_true = rotvec_to_matrix(truth[i]) @ R_true
        if np.all(np.isfinite(est[i])):
            R_est = rotvec_to_matrix(est[i]) @ R_est
        out[i] = np.degrees(rotation_angles((R_est @ R_true.T)[None])[0])
    return out


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
    endpoint_err_pct: float  # world-frame endpoint error as % of true path length
    path_len_rad: float

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
        endpoint_err_pct=100.0 * endpoint / path_len if path_len > 0 else np.nan,
        path_len_rad=path_len,
    )
