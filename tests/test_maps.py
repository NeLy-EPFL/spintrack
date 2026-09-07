"""Saving, loading and localising against surface-map templates."""

import sys

import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.engine import TrackEngine, TrackParams
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.maps import (
    cube_directions,
    fictrac_template_to_map,
    load_map,
    map_directions,
    projection_of,
    render_map,
    resample_map,
    sample_map,
    save_map,
)
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


def test_sample_map_reads_cell_centres_exactly():
    """`sample_map` inverts `map_directions`, so both agree with `Map::project` in Rust."""
    rng = np.random.default_rng(0)
    mean = rng.standard_normal((18, 36)).astype(np.float32)
    weight = np.ones_like(mean)
    value, seen = sample_map(mean, weight, map_directions(mean.shape))
    assert np.allclose(value, mean, atol=1e-5), np.abs(value - mean).max()
    assert seen.all()


def test_cube_faces_meet_at_their_seams():
    """The net is an unfolded dice: neighbouring faces have to line up along their edge."""
    n = 32
    faces = cube_directions(n)
    texel = 0.5 * np.pi / n  # angular size of a face-centre texel
    band = ["-x", "+z", "+x", "-z"]
    for left, right in zip(band, band[1:] + band[:1], strict=True):
        gap = np.arccos(
            np.clip(np.sum(faces[left][:, -1] * faces[right][:, 0], axis=-1), -1, 1)
        )
        assert gap.max() < texel, (left, right, np.degrees(gap.max()))
    for pole, row in (("+y", 0), ("-y", -1)):
        edge = faces[pole][-1] if pole == "+y" else faces[pole][0]
        gap = np.arccos(np.clip(np.sum(edge * faces["+z"][row], axis=-1), -1, 1))
        assert gap.max() < texel, (pole, np.degrees(gap.max()))


def test_cube_and_equal_area_shapes_are_told_apart():
    assert projection_of((90, 180)) == "equal_area"
    assert projection_of((312, 52)) == "cube"


def test_resample_survives_a_change_of_projection():
    """A map saved before the cube existed, or a FicTrac template, has to keep loading."""
    dirs = map_directions((90, 180))
    mean = (dirs[..., 0] * dirs[..., 1] + 0.5 * dirs[..., 2]).astype(np.float32)
    weight = np.full(mean.shape, 5.0, dtype=np.float32)
    unseen = dirs[..., 2] < -0.3  # a patch of the ball this run never looked at
    weight[unseen] = 0.0
    cube, cube_weight = resample_map(mean, weight, (312, 52))
    back, back_weight = resample_map(cube, cube_weight, mean.shape)
    seen = (weight > 1.0) & (back_weight > 1.0)
    assert seen.mean() > 0.6, seen.mean()
    assert np.corrcoef(mean[seen], back[seen])[0, 1] > 0.99
    # Unseen stays unseen rather than being invented on the way through the cube; only
    # the frontier bleeds, by the couple of cells two bilinear resamples reach.
    deep = dirs[..., 2] < -0.5
    assert back_weight[deep].max() < 0.1, back_weight[deep].max()


def test_a_cube_map_renders_on_either_grid():
    mean = np.random.default_rng(0).standard_normal((312, 52)).astype(np.float32)
    weight = np.full(mean.shape, 5.0, dtype=np.float32)
    assert render_map(mean, weight, layout="grid").shape == (90, 180)
    assert render_map(mean, weight, layout="cube").shape == (156, 208)
