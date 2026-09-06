"""Automatic preparation: ball into the config, vfov from the cost, and the refusals."""

import sys

import cv2
import numpy as np
import pytest

from spintrack.autofit import CAMERA_SOURCE_MESSAGE, fit_vfov, prepare_config
from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.geometry import normalize, rotvec_to_matrix
from spintrack.sphere import pixel_circle

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_detect import render
from test_engine import make_texture

CENTRE = normalize(np.array([0.0, 0.0, 1.0]))


def write_video(
    path, size, centre, half, n, vfov=40.0, seed=0, step=(0.02, 0.05, 0.01)
):
    """A rotating textured ball, as a temporary video file."""
    rng = np.random.default_rng(seed)
    texture = make_texture(rng, n_blobs=120)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50, (size[0], size[1])
    )
    R = np.eye(3)
    for i in range(n):
        if i > 0:
            R = rotvec_to_matrix(step) @ R
        frame = render(texture, R, rng, size, centre, half, occluders=False)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def base_config(**kwargs) -> Config:
    cfg = Config(q_factor=6, **kwargs)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    return cfg


def test_prepare_fills_a_config_that_has_no_ball(tmp_path):
    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTRE, half, 40)
    cfg = base_config(vfov=40.0)
    assert not cfg.has_ball()
    prepared = prepare_config(cfg, str(video))
    assert prepared.ball_source == "detected"
    assert cfg.has_ball() and len(cfg.roi_circ) >= 24
    assert abs(cfg.roi_r / half - 1) < 0.02, cfg.roi_r
    assert np.allclose(cfg.roi_c, CENTRE, atol=2e-3)


def test_prepare_reports_disagreement_but_keeps_the_config_ball(tmp_path):
    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTRE, half, 40)
    camera = PinholeCamera(size[0], size[1], 40.0)
    cx, cy, r = pixel_circle(camera, CENTRE, half)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    wrong = 0.85 * r  # a hand-fitted circle 15% too small
    cfg = base_config(vfov=40.0)
    cfg.roi_circ = [
        round(v)
        for a in angles
        for v in (cx + wrong * np.cos(a), cy + wrong * np.sin(a))
    ]
    before = list(cfg.roi_circ)
    prepared = prepare_config(cfg, str(video))
    assert cfg.roi_circ == before, "a config that describes a ball must be left alone"
    assert prepared.ball_source == "config"
    assert 0.15 < prepared.radius_disagreement < 0.22, prepared.radius_disagreement
    assert prepared.notes, "a disagreement this large must be said out loud"


def test_prepare_refuses_a_live_camera():
    with pytest.raises(ValueError, match="seekable"):
        prepare_config(base_config(vfov=40.0), "0")
    assert "calibrate" in CAMERA_SOURCE_MESSAGE


def test_vfov_is_fitted_when_the_ball_fills_the_frame(tmp_path):
    """A 16 deg half-angle: perspective across the ball pins the field of view down."""
    size, half, vfov = (160, 120), 0.28, 40.0
    video = write_video(tmp_path / "wide.mp4", size, CENTRE, half, 200, vfov=vfov)
    camera = PinholeCamera(size[0], size[1], vfov)
    fit = fit_vfov(
        str(video),
        base_config(vfov=vfov),
        pixel_circle(camera, CENTRE, half),
        n_frames=200,
        grid=(10.0, 120.0, 5),
    )
    assert fit.identifiable and fit.depth > 2.0
    assert abs(fit.vfov / vfov - 1) < 0.10, fit.vfov


def test_vfov_is_not_identifiable_on_a_near_orthographic_view(tmp_path):
    """A ball spanning a degree: any vfov in the flat region gives the same rotation."""
    size, half, vfov = (160, 120), 0.02, 3.0
    video = write_video(tmp_path / "small.mp4", size, CENTRE, half, 200, vfov=vfov)
    camera = PinholeCamera(size[0], size[1], vfov)
    fit = fit_vfov(
        str(video),
        base_config(vfov=vfov),
        pixel_circle(camera, CENTRE, half),
        n_frames=200,
        grid=(1.0, 30.0, 5),
    )
    assert not fit.identifiable
    assert fit.flat_range[1] / fit.flat_range[0] >= 4.0, fit.flat_range


def test_scale_check_does_not_change_tracking(tmp_path):
    """It only reads the engine's state, so the output must be bit-for-bit identical."""
    from spintrack.engine import TrackParams
    from spintrack.io.sources import VideoSource
    from spintrack.tracker import Tracker

    size, half = (160, 120), 0.28
    video = write_video(tmp_path / "ball.mp4", size, CENTRE, half, 60)
    cfg = base_config(vfov=40.0, roi_c=list(CENTRE), roi_r=half)
    runs = []
    for stride in (0, 5):
        source = VideoSource(video)
        tracker = Tracker(
            cfg, source.width, source.height, TrackParams(scale_check_stride=stride)
        )
        rows = []
        while (frame := source.read()) is not None:
            result = tracker.process_frame(frame.image, frame.ts_ms)
            rows.append(result.w_cam if result is not None else np.zeros(3))
        source.close()
        runs.append(np.array(rows))
        if stride:
            assert tracker.scale_check is not None
    assert np.array_equal(runs[0], runs[1])


def test_radius_error_inversion_is_monotone_and_zero_at_one():
    from spintrack.autofit import radius_error_from_ratio

    assert radius_error_from_ratio(1.0) == pytest.approx(0.0, abs=1e-6)
    ratios = np.linspace(0.6, 1.3, 40)
    estimates = [radius_error_from_ratio(float(r)) for r in ratios]
    assert np.all(np.diff(estimates) < 0), (
        "a larger outer/inner ratio is a smaller ball"
    )
    # The calibration points must come back as the errors they were measured at.
    assert radius_error_from_ratio(1.121) == pytest.approx(-0.10, abs=0.005)
    assert radius_error_from_ratio(0.750) == pytest.approx(0.10, abs=0.005)
