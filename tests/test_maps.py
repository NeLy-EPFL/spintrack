"""Saving, loading and localising against surface-map templates."""

import sys

import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.engine import TrackEngine, TrackParams
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.maps import fictrac_template_to_map, load_map, save_map
from spintrack.sphere import source_mask, window_geometry

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_engine import make_texture, render_window

CAM = PinholeCamera(320, 240, 40.0)
CENTRE = normalize(np.array([0.05, -0.03, 1.0]))
HALF = 0.25


def test_fictrac_template_conversion_flips_and_scales():
    img = np.full((6, 12), 128, np.uint8)
    img[0, 0] = 0  # dark tile at FicTrac's top-left ...
    img[5, 11] = 255  # ... and bright at its bottom-right
    mean, weight = fictrac_template_to_map(img, (6, 12))
    assert mean.shape == (6, 12) and weight.sum() == 2
    assert mean[5, 11] == -1.0 and weight[5, 11] == 1.0  # flipped both ways
    assert mean[0, 0] == 1.0 and weight[0, 0] == 1.0
    assert weight[3, 3] == 0.0 and mean[3, 3] == 0.0


def test_saved_map_localises_a_fresh_engine(tmp_path):
    rng = np.random.default_rng(4)
    geom = window_geometry(CAM, CENTRE, HALF, 60, source_mask(CAM, CENTRE, HALF))
    texture = make_texture(rng)
    a = TrackEngine(geom, TrackParams())
    R = np.eye(3)
    a.step(render_window(geom, texture, R, rng))
    for _ in range(40):
        R = rotvec_to_matrix([0.03, 0.02, 0.0]) @ R
        assert a.step(render_window(geom, texture, R, rng)).ok
    mean, weight = a.export_map()
    path = save_map(tmp_path / "ball", mean, weight, window_size=60)
    mean2, weight2 = load_map(path, a.map_shape)
    assert np.array_equal(mean2, mean) and np.array_equal(weight2, weight)

    # A new engine with the saved map finds the absolute orientation of a mid-sequence frame.
    b = TrackEngine(geom, TrackParams(global_search=True))
    b.load_map(mean2, weight2, frozen=True)
    R_mid = rotvec_to_matrix([0.03 * 20, 0.02 * 20, 0.0])
    res = b.step(render_window(geom, texture, R_mid, rng))
    assert res.ok and res.source == "global", res
    err = np.degrees(np.linalg.norm(matrix_to_rotvec(b.R @ R_mid.T)))
    assert err < 0.5, err
    # Frozen: the map is untouched by tracking; tracking continues normally.
    res2 = b.step(
        render_window(geom, texture, rotvec_to_matrix([0.02, 0.0, 0.0]) @ R_mid, rng)
    )
    assert res2.ok and res2.source == "map"
    assert np.array_equal(b.export_map()[0], mean2)
