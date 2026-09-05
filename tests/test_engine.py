"""Known-rotation recovery on a procedurally textured ball, without any video."""

import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.engine import TrackEngine, TrackParams
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.sphere import source_mask, window_geometry

CAM = PinholeCamera(320, 240, 40.0)
CENTRE = normalize(np.array([0.05, -0.03, 1.0]))
HALF = 0.25


def make_texture(rng, n_blobs=120):
    centres = rng.normal(size=(n_blobs, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    radii = rng.uniform(0.06, 0.16, n_blobs)

    def sample(dirs):
        cos = dirs @ centres.T
        ang = np.arccos(np.clip(cos, -1, 1))
        dark = np.clip((radii + 0.02 - ang) / 0.04, 0, 1)
        return 0.92 * np.prod(1 - 0.9 * dark, axis=1)

    return sample


def render_window(geom, texture, R, rng):
    dirs = geom.surface.astype(np.float64) @ R  # R^T applied row-wise
    vals = np.zeros(geom.size * geom.size)
    vals[geom.index] = texture(dirs) * 230 + 10
    img = vals.reshape(geom.size, geom.size) + rng.normal(
        0, 2.0, (geom.size, geom.size)
    )
    return np.clip(img, 0, 255).astype(np.uint8)


def test_engine_recovers_known_rotations():
    rng = np.random.default_rng(0)
    mask = source_mask(CAM, CENTRE, HALF)
    geom = window_geometry(CAM, CENTRE, HALF, 60, mask)
    texture = make_texture(rng)
    engine = TrackEngine(geom, TrackParams())
    R = np.eye(3)
    engine.step(render_window(geom, texture, R, rng))
    errors = []
    for i in range(40):
        w_true = np.array([0.02, -0.015, 0.01]) * (1 + 0.3 * np.sin(i / 3))
        if i == 20:
            w_true = np.array([0.0, 0.15, 0.05])  # a saccade-sized jump
        R = rotvec_to_matrix(w_true) @ R
        res = engine.step(render_window(geom, texture, R, rng))
        assert res.ok, f"frame {i} lost: {res}"
        err = np.linalg.norm(
            matrix_to_rotvec(rotvec_to_matrix(res.w_win) @ rotvec_to_matrix(w_true).T)
        )
        errors.append(np.degrees(err))
    errors = np.array(errors)
    assert np.median(errors) < 0.05, errors
    assert errors.max() < 0.5, errors
    # Absolute orientation stays locked to the truth (no drift beyond the per-frame noise).
    drift = np.degrees(np.linalg.norm(matrix_to_rotvec(engine.R @ R.T)))
    assert drift < 0.5, drift


def test_engine_forget_outside_view_tracks_too():
    rng = np.random.default_rng(1)
    mask = source_mask(CAM, CENTRE, HALF)
    geom = window_geometry(CAM, CENTRE, HALF, 60, mask)
    texture = make_texture(rng)
    engine = TrackEngine(geom, TrackParams(forget_outside_view=True))
    R = np.eye(3)
    engine.step(render_window(geom, texture, R, rng))
    for _ in range(10):
        R = rotvec_to_matrix([0.03, 0.0, 0.0]) @ R
        res = engine.step(render_window(geom, texture, R, rng))
        assert (
            res.ok
            and abs(np.degrees(np.linalg.norm(res.w_win)) - np.degrees(0.03)) < 0.1
        )
    weight = engine.core.map_weight()
    assert (weight > 0).mean() < 0.6  # only the current view (plus margin) is kept


def test_global_search_relocalises_after_a_large_jump():
    rng = np.random.default_rng(2)
    mask = source_mask(CAM, CENTRE, HALF)
    geom = window_geometry(CAM, CENTRE, HALF, 60, mask)
    texture = make_texture(rng)
    engine = TrackEngine(geom, TrackParams(global_search=True, max_step=0.3))
    R = np.eye(3)
    engine.step(render_window(geom, texture, R, rng))
    # Sweep the ball so the map covers a broad region.
    for _ in range(30):
        R = rotvec_to_matrix([0.04, 0.02, 0.0]) @ R
        assert engine.step(render_window(geom, texture, R, rng)).ok
    # A jump far beyond max_step: the local solves must fail and the global search recover.
    R_jump = rotvec_to_matrix([-0.6, -0.3, 0.1]) @ R
    res = engine.step(render_window(geom, texture, R_jump, rng))
    assert res.ok and res.source == "global", res
    err = np.degrees(np.linalg.norm(matrix_to_rotvec(engine.R @ R_jump.T)))
    assert err < 0.5, err
