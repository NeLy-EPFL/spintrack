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


def _drifting_run(projection: str, seed: int = 7):
    """80 frames of frame-to-frame tracking under heavy noise, with one frame dropped.

    Returns `(engine, windows, online, truth)`. Frame-to-frame mode (`forget_outside_view`)
    integrates a random walk, which is the drift refinement exists to undo.
    """
    rng = np.random.default_rng(seed)
    geom = window_geometry(CAM, CENTRE, HALF, 60, source_mask(CAM, CENTRE, HALF))
    texture = make_texture(rng)
    engine = TrackEngine(
        geom, TrackParams(forget_outside_view=True, map_projection=projection)
    )
    noise = np.random.default_rng(seed + 1)
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
    return engine, windows, R_online, R_true


def test_refinement_reduces_accumulated_error_and_recovers_dropped_frames():
    """Pinned to the equal-area grid, which is where this inequality still has room.

    Over texture seeds 7, 11, 13 and 17 the refinement takes the worst-frame error down
    by 5-13% there (seed 19 is the exception, +10%). On the cubemap the online run is
    already better than the equal-area run manages after refinement, and refining it
    changes the figure by a few percent either way, so the same assertion would be
    measuring noise. `test_refinement_leaves_a_cube_run_alone` covers the default.
    """
    engine, windows, online, truth = _drifting_run("equal_area")
    err_online = _accumulated_error(online, truth)
    refined, stats = refine_orientations(engine, windows, online, sweeps=2)
    err_refined = _accumulated_error(refined, truth)
    assert refined[40] is not None and stats["recovered"] >= 1
    assert err_refined < err_online, (err_online, err_refined)
    assert err_refined < 0.5, err_refined
    inc = increments(refined)
    assert inc[0] is not None and np.allclose(inc[0], 0) and len(inc) == 80


def test_refinement_leaves_a_cube_run_alone():
    """On the default map there is little drift left to remove, and none to add.

    The point of the test is the second half: a sweep that rebuilds the map and re-solves
    every frame must not make a run that was already good worse. `_accumulated_error` is a
    worst-frame figure under heavy noise: over seeds 7, 11, 13, 17 and 19 the refined
    figure is 0.98-1.05 of the online one whatever the map's temporal window, so the
    margin here is that noise, and not regressing beyond it is the claim that holds.
    """
    engine, windows, online, truth = _drifting_run("cube")
    err_online = _accumulated_error(online, truth)
    refined, stats = refine_orientations(engine, windows, online, sweeps=2)
    err_refined = _accumulated_error(refined, truth)
    assert refined[40] is not None and stats["recovered"] >= 1
    assert err_refined < 1.06 * err_online, (err_online, err_refined)
    assert err_refined < 0.5, err_refined
