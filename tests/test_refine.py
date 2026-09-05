"""Offline refinement reduces the drift of frame-to-frame tracking on a synthetic sequence."""

import sys

import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.engine import TrackEngine, TrackParams
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.refine import increments, refine_orientations
from spintrack.sphere import source_mask, window_geometry

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_engine import make_texture, render_window

CAM = PinholeCamera(320, 240, 40.0)
CENTRE = normalize(np.array([0.05, -0.03, 1.0]))
HALF = 0.25


def _accumulated_error(R_est, R_true):
    return max(
        np.degrees(np.linalg.norm(matrix_to_rotvec(a @ b.T)))
        for a, b in zip(R_est, R_true)
        if a is not None
    )


def test_refinement_reduces_accumulated_error_and_recovers_dropped_frames():
    rng = np.random.default_rng(7)
    geom = window_geometry(CAM, CENTRE, HALF, 60, source_mask(CAM, CENTRE, HALF))
    texture = make_texture(rng)
    # Frame-to-frame mode with heavy noise drifts; refinement against the full map should not.
    engine = TrackEngine(geom, TrackParams(forget_outside_view=True))
    noise = np.random.default_rng(8)
    R_true, windows, R_online = [], [], []
    R = np.eye(3)
    for i in range(80):
        if i > 0:
            R = rotvec_to_matrix([0.03, 0.02 * np.sin(i / 7), 0.01]) @ R
        win = render_window(geom, texture, R, noise)
        win = np.clip(
            win.astype(np.float32) + noise.normal(0, 12, win.shape), 0, 255
        ).astype(np.uint8)
        res = engine.step(win)
        R_true.append(R.copy())
        windows.append(engine.last_obs.astype(np.float16))
        R_online.append(res.R_win.copy() if res.ok else None)
    R_online[40] = None  # pretend one frame was dropped online
    err_online = _accumulated_error(R_online, R_true)
    refined, stats = refine_orientations(engine, windows, R_online, sweeps=2)
    err_refined = _accumulated_error(refined, R_true)
    assert refined[40] is not None and stats["recovered"] >= 1
    assert err_refined < err_online, (err_online, err_refined)
    assert err_refined < 0.5, err_refined
    inc = increments(refined)
    assert inc[0] is not None and np.allclose(inc[0], 0) and len(inc) == 80
