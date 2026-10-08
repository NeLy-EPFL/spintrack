"""Static illumination is separated from the ball's texture instead of entering the
map.
"""

import numpy as np
import pytest

from helpers import make_texture
from spintrack import photometry
from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.engine import TrackEngine, TrackParams
from spintrack.geometry import normalize, rotvec_to_matrix
from spintrack.maps import load_illumination, load_map, save_map
from spintrack.sphere import source_mask, window_geometry
from spintrack.tracker import Tracker

CAM = PinholeCamera(320, 240, 40.0)
CENTER = normalize(np.array([0.05, -0.03, 1.0]))
HALF = 0.25
SIZE = 60
# Fast enough to converge inside a test; the shipped constants are ten times slower.
FAST = {"WARMUP": 20, "UPDATE_EVERY": 10, "TAU": 60.0}


@pytest.fixture(autouse=True)
def fast_fields(monkeypatch):
    for name, value in FAST.items():
        monkeypatch.setattr(photometry, name, value)


def shading(size: int, start: float = 0.68, amp: float = 0.7, glow: float = 0.08):
    """A dark band across the bottom of the window, fixed in the camera frame.

    Shaped like the holder shadow on the lab rig: a sharp edge, a deep multiplicative
    darkening, and a little stray light that flattens what contrast is left.
    """
    rows = np.arange(size)[:, None] / size
    t = np.clip((rows - start) / 0.06, 0.0, 1.0) * np.ones((1, size))
    t = t * t * (3.0 - 2.0 * t)
    return 1.0 - amp * t, glow * t


def geometry():
    mask = source_mask(CAM, CENTER, HALF)
    return window_geometry(CAM, CENTER, HALF, SIZE, mask)


def render(geom, texture, R, rng, shade=None, glow=None):
    dirs = geom.surface.astype(np.float64) @ R  # R^T applied row-wise
    vals = np.zeros(geom.size * geom.size)
    vals[geom.index] = texture(dirs) * 230 + 10
    img = vals.reshape(geom.size, geom.size)
    if shade is not None:
        img = np.where(geom.mask, img * shade + 255 * glow, img)
    img = img + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def turns(n: int) -> list[np.ndarray]:
    """A rotation sequence that sweeps enough surface past the shadow to see it."""
    R, out = np.eye(3), []
    for i in range(n):
        w = np.array([0.02 * np.cos(i / 17), 0.05, 0.01 * np.sin(i / 23)])
        R = rotvec_to_matrix(w) @ R
        out.append(R)
    return out


def run(params, n=400, seed=0):
    rng = np.random.default_rng(seed)
    geom = geometry()
    texture = make_texture(np.random.default_rng(7))
    shade, glow = shading(SIZE)
    engine = TrackEngine(geom, params)
    for R in turns(n):
        engine.step(render(geom, texture, R, rng, shade, glow))
    return engine, geom, texture


def test_bias_field_recovers_the_shadow():
    fixed, geom, _ = run(TrackParams(illum_bias=True))

    # The field lands where the artifact actually is. Deep inside the shadow the local
    # normalization copes, because the whole neighborhood is dark; it is at the *edge*
    # that its box straddles bright and dark and drives the reading down. So the field
    # should be strongly negative in that band and flat well above it.
    rows = np.arange(SIZE)[:, None]
    edge = geom.mask & (rows >= 0.72 * SIZE) & (rows < 0.85 * SIZE)
    top = geom.mask & (rows < 0.5 * SIZE)
    bias = fixed.photometry.bias
    assert bias[edge].mean() < -0.2, bias[edge].mean()
    assert abs(bias[top].mean()) < 0.1, bias[top].mean()


def test_window_move_carries_the_field_with_the_window():
    engine, geom, _ = run(TrackParams(illum_bias=True), n=200)
    before = engine.photometry.bias.copy()
    rows = np.arange(SIZE)[:, None]
    edge = (rows >= 0.72 * SIZE) & (rows < 0.85 * SIZE)
    band_before = before[geom.mask & edge].mean()
    # A window aimed slightly differently: the field follows the window, so the shadow
    # must still sit at the bottom of it, not be rotated away with the map.
    moved = normalize(CENTER + np.array([0.0, 0.004, 0.0]))
    new_geom = window_geometry(CAM, moved, HALF, SIZE, source_mask(CAM, moved, HALF))
    Q = new_geom.to_camera.T @ geom.to_camera
    engine.rebuild(new_geom, Q)
    after = engine.photometry.bias
    assert after[new_geom.mask & edge].mean() == pytest.approx(band_before, abs=0.1)
    assert not np.array_equal(after, before)  # it did move


def test_a_ball_that_moved_in_its_holder_forgets_the_field():
    """A window nudge carries the field; a ball that actually moved invalidates it.

    Shading follows the surface normal, so translating the ball re-lights it: past a
    fraction of a radius the old field describes a shadow that is no longer there, and
    carrying it is worse than starting again.
    """
    engine, geom, _ = run(TrackParams(illum_bias=True), n=200)
    assert np.abs(engine.photometry.bias).max() > 0.1  # there is something to lose
    moved = normalize(CENTER + np.array([0.0, 0.05, 0.0]))  # ~0.05 rad, HALF is 0.25
    new_geom = window_geometry(CAM, moved, HALF, SIZE, source_mask(CAM, moved, HALF))
    engine.rebuild(new_geom, new_geom.to_camera.T @ geom.to_camera)
    assert np.array_equal(engine.photometry.bias, np.zeros((SIZE, SIZE)))
    assert engine.photometry.acc_n.max() == 0.0


def test_map_file_carries_the_illumination_field(tmp_path):
    engine, _, _ = run(TrackParams(illum_bias=True), n=200)
    mean, weight = engine.export_map()
    state = engine.photometry.state()
    path = save_map(
        tmp_path / "m.npz", mean, weight, window_size=SIZE, illum_bias=state["bias"]
    )
    back_mean, back_weight = load_map(path, mean.shape)
    assert np.array_equal(back_mean, mean) and np.array_equal(back_weight, weight)
    fields, _ = load_illumination(path, SIZE)
    assert np.array_equal(fields["bias"], state["bias"])
    # A map saved for a different window cannot supply window-shaped fields.
    assert load_illumination(path, SIZE + 2)[0] == {}


def test_illumination_fn_loads_the_fields_and_leaves_the_map_alone(tmp_path):
    """The point of `illumination_fn`: the rig's lighting without a previous ball.

    `sphere_map_fn` carries the fields too, but only along with the surface map they
    were saved beside, and localizing the first frame against it.
    """
    engine, _, _ = run(TrackParams(illum_bias=True), n=200)
    state = engine.photometry.state()
    mean, weight = engine.export_map()
    path = save_map(
        tmp_path / "m.npz", mean, weight, window_size=SIZE, center=CENTER,
        half_angle=HALF, illum_bias=state["bias"],
    )  # fmt: skip
    cfg = Config(vfov=CAM.vfov_deg, q_factor=SIZE // 10, roi_c=list(CENTER), roi_r=HALF)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.illumination_fn = str(path)
    tracker = Tracker(cfg, CAM.width, CAM.height, TrackParams(center_watch=False))
    photo = tracker.engine.photometry
    got, want = photo.bias[photo.mask], state["bias"][photo.mask]
    assert np.corrcoef(got, want)[0, 1] > 0.99
    assert tracker.engine.map_coverage() == 0.0
    assert not tracker.params.global_search


def test_a_loaded_field_survives_the_first_refresh():
    """A loaded field goes into the accumulators, not only into `bias`.

    Every refresh reads the field off the accumulators, so a field that was merely
    assigned is replaced by a handful of the new run's own samples `WARMUP` frames
    in, and the head start lasts exactly that long. Seeded `TAU` frames deep it is
    a prior instead, which the new frames pull away from over that time constant.
    """
    donor, geom, texture = run(TrackParams(illum_bias=True), n=400)
    fields = donor.photometry.state()
    shade, glow = shading(SIZE)

    def distance(prior_frames):
        engine = TrackEngine(geometry(), TrackParams(illum_bias=True))
        engine.photometry.load(**fields, prior_frames=prior_frames)
        rng = np.random.default_rng(11)
        for R in turns(FAST["WARMUP"] + FAST["UPDATE_EVERY"] + 5):
            engine.step(render(geom, texture, R, rng, shade, glow))
        photo = engine.photometry
        gap = photo.bias - fields["bias"]
        return float(np.sqrt((gap[photo.mask] ** 2).mean()))

    kept, dropped = distance(FAST["TAU"]), distance(0.0)
    assert kept < 0.5 * dropped, (kept, dropped)
