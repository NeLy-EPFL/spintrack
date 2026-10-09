"""Automatic preparation: ball into the config, vfov from the cost, and the refusals."""

import json

import cv2
import numpy as np
import pytest

from helpers import make_texture, render
from spintrack.autofit import CAMERA_SOURCE_MESSAGE, fit_vfov, prepare_config
from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.detect import fit_circle
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
    with pytest.raises(ValueError, match="live camera"):
        prepare_config(base_config(vfov=40.0), "0")
    assert "spintrack gui" in CAMERA_SOURCE_MESSAGE


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


def test_a_video_alone_needs_only_the_camera_azimuth(tmp_path, caplog):
    """A video cannot tell the animal's front from its back, so a run asks for the
    azimuth, before any slow work; given it, the ball and the vfov come from the
    recording, and the config the run writes says all three."""
    import logging

    from spintrack.cli import main

    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTER, half, 40)
    with caplog.at_level(logging.ERROR, logger="spintrack"):
        assert main([str(video), "--no-preview"]) == 2
    assert "camera.azimuth_deg=180" in caplog.records[-1].getMessage()
    assert not (tmp_path / "ball_spintrack").exists()
    argv = [str(video), "camera.azimuth_deg=180", "--max-frames", "5", "--no-preview"]
    assert main(argv) == 0
    out = tmp_path / "ball_spintrack"
    cfg = Config.load(out / "config.toml")
    assert cfg.video == str(video) and cfg.ball.rim and cfg.camera.vfov_deg
    assert cfg.camera.position_deg == (0.0, 180.0, 0.0)  # level: no silhouette here
    provenance = json.loads((out / "summary.json").read_text())["provenance"]
    assert provenance["camera_position"]["source"] == "estimated"
    assert provenance["ball"]["source"] == "detected"


@pytest.mark.parametrize(
    "position",
    [
        (0.0, 180.0, 0.0),
        (0.0, 180.0, -15.0),
        (35.0, 180.0, 10.0),
        (70.0, 90.0, -20.0),
        (82.0, 180.0, 0.0),  # nearly overhead: no twist read, none assumed
    ],
)
def test_the_camera_is_placed_where_the_animal_stands(monkeypatch, position):
    """An animal on the ball's top, seen from `position`: the fit reads it back."""
    from spintrack import segment
    from spintrack.autofit import place_camera
    from spintrack.calibrate.sliders import camera_to_lab_from_angles
    from spintrack.sphere import ball_outline

    size, half = (640, 480), 0.02  # a small ball in a narrow view, as on a rig
    camera = PinholeCamera(*size, 5.0)
    rim = ball_outline(camera, CENTER, half, 16)
    up = camera_to_lab_from_angles(*position).T @ np.array([0.0, 0.0, 1.0])
    x, y, _ = camera.project(normalize(CENTER + np.sin(half) * up))
    mask = np.zeros(size[::-1], np.uint8)
    cv2.circle(mask, (round(float(x)), round(float(y))), 4, 1, -1)
    # A patterned ball can be the best "insect", and so can a spot of its texture,
    # which moves from frame to frame: both are passed over.
    ball = np.zeros_like(mask)
    cx, cy, r = np.round(fit_circle(rim)).astype(int)
    cv2.circle(ball, (cx, cy), r, 1, -1)
    calls = iter(range(3))

    def animal_masks(image):
        spot = np.zeros_like(mask)
        cv2.circle(spot, (cx - r // 2 + next(calls) * r // 3, cy + r // 4), 6, 1, -1)
        return np.stack([ball, spot, mask]) > 0, np.array([0.95, 0.92, 0.9])

    monkeypatch.setattr(segment, "animal_masks", animal_masks)
    frames = [np.zeros(size[::-1], np.uint8)] * 3
    fit = place_camera(frames, rim, position[1])
    # Orthographic: the top's nearness to the camera costs about a degree at 35.
    assert np.allclose(fit.position_deg, position, atol=1.5), fit


def test_walking_off_forward_suggests_the_azimuth():
    """A camera behind the animal, configured at its right: forward walking reads as
    walking to the left, and the check points back to behind."""
    from spintrack.calibrate.sliders import camera_to_lab_from_angles
    from spintrack.quality import walking_check

    forward = np.tile([0.0, -0.01, 0.0], (2000, 1))  # 20 ball radii straight ahead
    w_cam = forward @ camera_to_lab_from_angles(0, 180, 0)  # into the true camera
    w_lab = w_cam @ camera_to_lab_from_angles(0, 90, 0).T  # out by the wrong one
    line = walking_check(w_lab, (0.0, 90.0, 0.0))
    assert "90 deg left of forward" in line and "azimuth near 180" in line
    assert "wrong" not in walking_check(forward, (0.0, 180.0, 0.0))


def test_a_shallow_wide_minimum_is_taken_for_a_narrow_lens(tmp_path, monkeypatch):
    """A cost that leans to a wide lens without a clear minimum is not evidence of
    one (a lab rig's 2 deg lens leaned to 26 deg and read 18% low): the narrow end is
    assumed, and the fit says how much smaller the rotations would read there."""
    from spintrack import autofit

    size, half = (160, 120), 0.2
    video = write_video(tmp_path / "v.mp4", size, CENTER, half, 5, vfov=30.0)
    circle = pixel_circle(PinholeCamera(size[0], size[1], 30.0), CENTER, half)
    values = np.geomspace(1.0, 120.0, 9)
    costs = np.array([1.30, 1.30, 1.25, 1.15, 1.08, 1.01, 1.00, 1.05, 1.20])

    turned = 600.0  # plenty of rotation: a lean without a clear minimum is a mismatch

    def costs_at(src, cfg, points, vfovs, n_frames, params, max_frames=None):
        cost = np.interp(np.log(vfovs), np.log(values), costs)
        return cost, turned * (1 - 0.1 * np.log10(vfovs))  # wider reads less

    monkeypatch.setattr(autofit, "_costs_at", costs_at)
    fit = fit_vfov(str(video), base_config(vfov=30.0), circle)
    assert not fit.identifiable and fit.leaning == pytest.approx(values[6])
    assert fit.flat_range == (1.0, values[2]) and fit.vfov < values[2]
    assert fit.scale_spread > 0.05 and "leans to" in fit.line()
    turned = 200.0  # too little rotation to tell: no guess
    with pytest.raises(ValueError, match="too little to tell"):
        fit_vfov(str(video), base_config(vfov=30.0), circle)


def test_a_plain_ball_is_told_from_a_patterned_one():
    """Plain polystyrene gives the tracker nothing to hold: the run says so."""
    from spintrack.autofit import MIN_TEXTURE, ball_texture

    rng = np.random.default_rng(0)
    plain = np.full((240, 320), 40, np.uint8)
    cv2.circle(plain, (160, 120), 100, 200, -1)
    plain = np.clip(plain + rng.normal(0, 1.5, plain.shape), 0, 255).astype(np.uint8)
    patterned = plain.copy()
    for x, y in rng.uniform(80, 240, (40, 2)):
        cv2.circle(patterned, (int(x), int(y) - 40), 8, 60, -1)
    assert ball_texture(plain, (160, 120, 100)) < MIN_TEXTURE
    assert ball_texture(patterned, (160, 120, 100)) > 3 * MIN_TEXTURE
