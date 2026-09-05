import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.geometry import normalize
from spintrack.sphere import ball_outline, fit_ball, source_mask, window_geometry

CAM = PinholeCamera(640, 480, 45.0)
CENTRE = normalize(np.array([0.12, -0.05, 1.0]))
HALF = 0.2


def test_fit_ball_recovers_synthetic_outline():
    rim = ball_outline(CAM, CENTRE, HALF, n_points=9)
    rng = np.random.default_rng(3)
    rim_noisy = rim + rng.normal(scale=0.3, size=rim.shape)
    c, a = fit_ball(rim, CAM)
    assert np.arccos(np.clip(c @ CENTRE, -1, 1)) < 1e-9 and abs(a - HALF) < 1e-9
    c, a = fit_ball(rim_noisy, CAM)
    assert (
        np.degrees(np.arccos(np.clip(c @ CENTRE, -1, 1))) < 0.05
        and abs(a - HALF) < 1e-3
    )


def test_source_mask_covers_ball_and_cuts_ignore_region():
    mask = source_mask(CAM, CENTRE, HALF)
    x, y, _ = CAM.project(CENTRE)
    assert mask[int(y), int(x)] == 255
    area = np.pi * (CAM.focal_px * np.tan(0.975 * HALF)) ** 2
    assert abs(mask.astype(bool).sum() / area - 1.0) < 0.05
    poly = [x - 20, y - 20, x + 20, y - 20, x + 20, y + 20, x - 20, y + 20]
    cut = source_mask(CAM, CENTRE, HALF, [poly])
    assert cut[int(y), int(x)] == 0 and cut.sum() < mask.sum()


def test_window_geometry_is_consistent():
    mask = source_mask(CAM, CENTRE, HALF)
    geom = window_geometry(CAM, CENTRE, HALF, 60, mask)
    assert geom.mask.shape == (60, 60) and geom.n_valid > 0.6 * 60 * 60 * np.pi / 4
    assert np.allclose(np.linalg.norm(geom.surface, axis=1), 1.0, atol=1e-5)
    assert (geom.surface[:, 2] < 0).all()  # visible cap faces the camera
    # The window centre (between the four central pixels) looks at the ball centre.
    x, y, _ = CAM.project(CENTRE)
    assert np.isclose(geom.map_x[29:31, 29:31].mean() + 0.5, x, atol=0.05)
    assert np.isclose(geom.map_y[29:31, 29:31].mean() + 0.5, y, atol=0.05)
    # Window frame +z is the ball direction in camera coordinates.
    assert np.allclose(geom.to_camera @ [0, 0, 1], CENTRE)
    # Remapping a constant image yields that constant on the ball.
    img = np.full((480, 640), 200, np.uint8)
    win = geom.remap(img)
    assert (win[geom.mask] == 200).all()
