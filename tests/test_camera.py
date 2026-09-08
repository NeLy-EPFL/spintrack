import numpy as np

from spintrack.camera import (
    EquidistantCamera,
    PinholeCamera,
    pixel_centers,
    source_camera,
)


def _round_trip(cam):
    rng = np.random.default_rng(2)
    x = rng.uniform(0, cam.width, 500)
    y = rng.uniform(0, cam.height, 500)
    px, py, valid = cam.project(cam.rays(x, y))
    assert valid.all()
    assert np.allclose(px, x, atol=1e-9) and np.allclose(py, y, atol=1e-9)


def test_pinhole_round_trip_and_fov():
    cam = PinholeCamera(640, 480, 45.0)
    _round_trip(cam)
    center = cam.rays(320.0, 240.0)
    assert np.allclose(center, [0, 0, 1])
    top = cam.rays(320.0, 0.0)
    assert np.isclose(np.degrees(np.arccos(top @ center)), 22.5)


def test_equidistant_round_trip_and_linear_angle():
    cam = EquidistantCamera.from_vfov(640, 480, 90.0)
    _round_trip(cam)
    for r in (10.0, 100.0, 200.0):
        d = cam.rays(320.0 + r, 240.0)
        assert np.isclose(np.arccos(d[2]), r * cam.rad_per_pixel)
    assert np.allclose(np.linalg.norm(cam.rays(*pixel_centers(4, 5)), axis=-1), 1.0)


def test_source_camera_selects_model():
    assert isinstance(source_camera(10, 10, 30.0, False), PinholeCamera)
    assert isinstance(source_camera(10, 10, 30.0, True), EquidistantCamera)
