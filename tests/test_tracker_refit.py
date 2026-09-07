"""Moving the tracking window onto a ball that has moved, without losing the map."""

import sys

import numpy as np
import pytest

from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.geometry import normalize, rotvec_to_matrix
from spintrack.sphere import pixel_circle
from spintrack.tracker import Tracker

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_detect import render
from test_engine import make_texture

SIZE = (240, 180)
VFOV = 40.0
HALF = 0.15
CENTRE = normalize(np.array([0.0, 0.0, 1.0]))
CAMERA = PinholeCamera(SIZE[0], SIZE[1], VFOV)
STEP = (0.02, 0.05, 0.01)


def config() -> Config:
    cfg = Config(vfov=VFOV, q_factor=6, roi_c=list(CENTRE), roi_r=HALF)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    return cfg


def shifted(dy: float):
    cx, cy, _ = pixel_circle(CAMERA, CENTRE, HALF)
    return normalize(CAMERA.rays(cx, cy + dy))


def sequence(n, drift=None, seed=0):
    """`n` frames of a rotating ball, optionally with the ball itself moving."""
    rng = np.random.default_rng(seed)
    texture = make_texture(rng, n_blobs=120)
    R = np.eye(3)
    out = []
    for i in range(n):
        if i > 0:
            R = rotvec_to_matrix(STEP) @ R
        centre = CENTRE if drift is None else drift(i)
        out.append((render(texture, R, rng, SIZE, centre, HALF, occluders=False), R))
    return out, texture, rng


@pytest.mark.parametrize(
    ("projection", "shape"), [("cube", (312, 52)), ("equal_area", (90, 180))]
)
def test_refit_carries_the_map_projection(projection, shape):
    """A window move rebuilds the Rust core, which has to keep the map's projection.

    Getting this wrong reads the map on the wrong grid: it survives the move as an array
    and becomes noise as a map, and the tracker starts dropping frames a few frames later
    rather than failing where the mistake was made. Both directions are worth pinning,
    since a run can be either projection.
    """
    params = TrackParams(centre_watch=False, map_projection=projection)
    tracker = Tracker(config(), *SIZE, params)
    images, texture, rng = sequence(20)
    for image, _ in images:
        tracker.process_frame(image)
    assert tracker.engine.map_shape == shape, tracker.engine.map_shape
    tracker.refit_centre(shifted(5.0))
    # The same ball, moved 5 px and followed, keeps turning at the same rate.
    R = images[-1][1]
    for _ in range(5):
        R = rotvec_to_matrix(STEP) @ R
        moved = render(texture, R, rng, SIZE, shifted(5.0), HALF, occluders=False)
        result = tracker.process_frame(moved)
        assert result is not None, "lost the ball after the window moved"
        assert np.allclose(result.w_cam, STEP, atol=2e-3), result.w_cam


def test_refit_is_a_change_of_coordinates():
    """The map and the ball's orientation in the camera survive a window move exactly."""
    tracker = Tracker(config(), *SIZE, TrackParams(centre_watch=False))
    images, texture, rng = sequence(20)
    for image, _ in images:
        tracker.process_frame(image)
    orientation = tracker.R_wc @ tracker.engine.R  # body -> camera, the physical state
    mean, weight = (a.copy() for a in tracker.engine.export_map())
    cost_before = tracker.engine._cost_level

    tracker.refit_centre(shifted(5.0))
    after_mean, after_weight = tracker.engine.export_map()
    assert np.array_equal(mean, after_mean)
    assert np.array_equal(weight, after_weight)
    assert np.allclose(tracker.R_wc @ tracker.engine.R, orientation, atol=1e-9)

    # The same ball, moved 5 px and followed, is not a rotation.
    _, R_last = images[-1]
    moved = render(texture, R_last, rng, SIZE, shifted(5.0), HALF, occluders=False)
    result = tracker.process_frame(moved)
    assert result is not None, "the tracker should not lose a ball it just followed"
    assert np.degrees(np.linalg.norm(result.w_cam)) < 0.15, result.w_cam
    assert result.step.cost < 1.5 * cost_before


def test_watch_follows_a_moving_ball():
    """With the watch off the ball's movement becomes rotation; with it on, it does not."""
    move, start, over = 25.0, 220, 150

    def drift(i):
        return shifted(move * float(np.clip((i - start) / over, 0.0, 1.0)))

    images, _, _ = sequence(start + over + 20, drift=drift)
    errors = {}
    for watch in (False, True):
        tracker = Tracker(config(), *SIZE, TrackParams(centre_watch=watch))
        wrong = []
        for image, _ in images:
            result = tracker.process_frame(image)
            wrong.append(
                np.nan
                if result is None
                else np.degrees(np.linalg.norm(result.w_cam - np.array(STEP)))
            )
        during = np.array(wrong[start + 70 : start + over])
        errors[watch] = float(np.nanmedian(during))
        if watch:
            assert tracker.refits, "the watch should have noticed a 25 px move"
            assert tracker.refits[-1].max_shift_px > 10.0
    assert errors[True] < 0.5 * errors[False], errors


def test_a_re_fitted_frame_keeps_its_observation_and_its_window_frame():
    """The frame whose window moved is still an observation, in the frame it was tracked in.

    Dropping it cost the offline refinement 562 of trial 004's 3013 rows and left the
    debug video's window panel blank over the whole episode.
    """
    move, start, over = 25.0, 220, 150

    def drift(i):
        return shifted(move * float(np.clip((i - start) / over, 0.0, 1.0)))

    images, _, _ = sequence(start + over + 20, drift=drift)
    tracker = Tracker(config(), *SIZE, TrackParams(centre_watch=True))
    kept = []  # (R_win, the version it was tracked in, the camera-frame orientation)
    moved_frames = 0
    for image, _ in images:
        result = tracker.process_frame(image)
        assert tracker.engine.last_obs is not None, "every tracked frame has a window"
        moved_frames += tracker.geometry_version > tracker.tracked_version
        if result is not None:
            kept.append((result.step.R_win, tracker.tracked_version, result.R_cam))
    assert tracker.refits and moved_frames > 1, (
        "the watch should have followed the ball"
    )

    # Every kept orientation, brought into the final window frame, must still describe
    # the camera-frame orientation that was reported at the time.
    brought = tracker.orientations_in_current_window([(r, v) for r, v, _ in kept])
    for R_now, (_, _, R_cam) in zip(brought, kept, strict=True):
        assert np.allclose(tracker.R_wc @ R_now @ tracker.R_wc0.T, R_cam, atol=1e-9)
