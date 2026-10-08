"""Automatic preparation: ball into the config, vfov from the cost, and the refusals."""

import cv2
import numpy as np
import pytest

from helpers import make_texture, render
from spintrack.autofit import CAMERA_SOURCE_MESSAGE, fit_vfov, prepare_config
from spintrack.camera import PinholeCamera
from spintrack.config import Config, leading_comments
from spintrack.geometry import normalize, rotvec_to_matrix
from spintrack.sphere import fit_ball, pixel_circle

CENTER = normalize(np.array([0.0, 0.0, 1.0]))


def write_video(
    path, size, center, half, n, vfov=40.0, seed=0, step=(0.02, 0.05, 0.01)
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
        frame = render(texture, R, rng, size, center, half, occluders=False)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def base_config(vfov: float) -> Config:
    return Config(camera={"vfov_deg": vfov, "rotation": (0.0, 0.0, 0.0)})


def test_prepare_fills_a_config_that_has_no_ball(tmp_path):
    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTER, half, 40)
    cfg = base_config(vfov=40.0)
    prepared = prepare_config(cfg, str(video))
    assert prepared.ball_source == "detected" and len(cfg.ball.rim) >= 12
    center, fitted = fit_ball(cfg.ball.rim, PinholeCamera(size[0], size[1], 40.0))
    assert abs(fitted / half - 1) < 0.02, fitted
    assert np.allclose(center, CENTER, atol=2e-3)


def test_prepare_reports_disagreement_but_keeps_the_config_ball(tmp_path):
    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTER, half, 40)
    camera = PinholeCamera(size[0], size[1], 40.0)
    cx, cy, r = pixel_circle(camera, CENTER, half)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    wrong = 0.85 * r  # a hand-fitted circle 15% too small
    cfg = base_config(vfov=40.0)
    cfg.ball.rim = [
        (round(cx + wrong * np.cos(a)), round(cy + wrong * np.sin(a))) for a in angles
    ]
    before = list(cfg.ball.rim)
    prepared = prepare_config(cfg, str(video))
    assert cfg.ball.rim == before, "a config that describes a ball must be left alone"
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
    video = write_video(tmp_path / "wide.mp4", size, CENTER, half, 200, vfov=vfov)
    camera = PinholeCamera(size[0], size[1], vfov)
    fit = fit_vfov(
        str(video),
        base_config(vfov=vfov),
        pixel_circle(camera, CENTER, half),
        n_frames=200,
        grid=(10.0, 120.0, 5),
    )
    assert fit.identifiable and fit.depth > 2.0
    assert abs(fit.vfov / vfov - 1) < 0.10, fit.vfov


def test_vfov_is_not_identifiable_on_a_near_orthographic_view(tmp_path):
    """A ball spanning a degree: any vfov in the flat region gives the same rotation."""
    size, half, vfov = (160, 120), 0.02, 3.0
    video = write_video(tmp_path / "small.mp4", size, CENTER, half, 200, vfov=vfov)
    camera = PinholeCamera(size[0], size[1], vfov)
    fit = fit_vfov(
        str(video),
        base_config(vfov=vfov),
        pixel_circle(camera, CENTER, half),
        n_frames=200,
        grid=(1.0, 30.0, 5),
    )
    assert not fit.identifiable
    assert fit.flat_range[1] / fit.flat_range[0] >= 4.0, fit.flat_range


def test_calibrate_auto_creates_the_config_and_run_refuses_an_unknown_vfov(tmp_path):
    """`calibrate --auto` writes a new config once per fit; `run` wants a fixed vfov."""
    from spintrack.cli import main

    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTER, half, 40)
    path = tmp_path / "config.toml"
    for _ in range(2):
        argv = ["calibrate", str(path), "--src", str(video), "--auto"]
        assert main([*argv, "--camera-position", "0", "180", "0"]) == 0
    cfg = Config.load(path)
    assert cfg.video == str(video) and cfg.ball.rim and cfg.camera.vfov_deg
    assert cfg.camera.position_deg == (0.0, 180.0, 0.0)
    assert 'video = "ball.mp4"' in path.read_text()  # relative to the config
    assert len(leading_comments(path)) == 1  # one note, not one per run
    cfg.camera.vfov_deg = None
    cfg.save(path)
    assert main(["run", str(path), "--max-frames", "5"]) == 2
