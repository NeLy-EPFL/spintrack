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
CENTER = normalize(np.array([0.0, 0.0, 1.0]))
CAMERA = PinholeCamera(SIZE[0], SIZE[1], VFOV)
STEP = (0.02, 0.05, 0.01)


def config() -> Config:
    cfg = Config(vfov=VFOV, q_factor=6, roi_c=list(CENTER), roi_r=HALF)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    return cfg


def shifted(dy: float):
    cx, cy, _ = pixel_circle(CAMERA, CENTER, HALF)
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
        center = CENTER if drift is None else drift(i)
        out.append((render(texture, R, rng, SIZE, center, HALF, occluders=False), R))
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
    params = TrackParams(center_watch=False, map_projection=projection)
    tracker = Tracker(config(), *SIZE, params)
    images, texture, rng = sequence(20)
    for image, _ in images:
        tracker.process_frame(image)
    assert tracker.engine.map_shape == shape, tracker.engine.map_shape
    tracker.refit_center(shifted(5.0))
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
    tracker = Tracker(config(), *SIZE, TrackParams(center_watch=False))
    images, texture, rng = sequence(20)
    for image, _ in images:
        tracker.process_frame(image)
    orientation = tracker.R_wc @ tracker.engine.R  # body -> camera, the physical state
    mean, weight = (a.copy() for a in tracker.engine.export_map())
    cost_before = float(np.median(tracker.engine._costs))

    tracker.refit_center(shifted(5.0))
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
        tracker = Tracker(config(), *SIZE, TrackParams(center_watch=watch))
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


def test_watch_keeps_up_with_a_jerk():
    """A ball that drops in a few frames has the window back on it within a few frames.

    The follower used to decide that the ball had moved on a slow filter of its looks,
    which on trial 004 let the ball fall 45 px before the window moved, and then
    followed it with a lagging filter that overshot each jerk by 10 px. The ball here is
    scaled like 004's: a drop of a quarter of the radius over eight frames.
    """
    half, move, start, over = 0.3, 20.0, 160, 8
    cx, cy0, _ = pixel_circle(CAMERA, CENTER, half)

    def offset(i):
        phase = float(np.clip((i - start) / over, 0.0, 1.0))
        return move * 0.5 * (1.0 - np.cos(np.pi * phase))

    rng = np.random.default_rng(0)
    texture = make_texture(rng, n_blobs=120)
    cfg = Config(vfov=VFOV, q_factor=6, roi_c=list(CENTER), roi_r=half)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    tracker = Tracker(cfg, *SIZE, TrackParams(center_watch=True))
    R, error = np.eye(3), []
    for i in range(start + 40):
        R = rotvec_to_matrix(STEP) @ R
        center = normalize(CAMERA.rays(cx, cy0 + offset(i)))
        tracker.process_frame(render(texture, R, rng, SIZE, center, half, False))
        _, cy, _ = pixel_circle(CAMERA, tracker.center, half)
        error.append(abs(cy - cy0 - offset(i)))
    # The rim look itself reads this ball 1.3-1.8 px low once the frame's edge cuts it,
    # and just after the jerk the window may run past the ball by up to `T_MOVE`.
    late = np.array(error)[start + over :]
    assert late.max() < 8.0 and np.median(late) < 4.0, late


def test_watch_finds_a_resting_ball_that_jumped_out_of_reach():
    """A ball that moves farther in one frame than its look can reach is found again.

    No seed can follow such a jump, and the look that needs none - a detection on a
    downscaled buffer of frames - used to be fed only while the window was already
    following, so a resting ball that jumped was lost for the rest of the recording.
    """
    half, move, start = 0.3, 25.0, 160
    cx, cy0, _ = pixel_circle(CAMERA, CENTER, half)
    rng = np.random.default_rng(0)
    texture = make_texture(rng, n_blobs=120)
    cfg = Config(vfov=VFOV, q_factor=6, roi_c=list(CENTER), roi_r=half)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    tracker = Tracker(cfg, *SIZE, TrackParams(center_watch=True))
    R, error = np.eye(3), []
    for i in range(start + 80):
        R = rotvec_to_matrix(STEP) @ R
        offset = move if i >= start else 0.0
        center = normalize(CAMERA.rays(cx, cy0 + offset))
        tracker.process_frame(render(texture, R, rng, SIZE, center, half, False))
        _, cy, _ = pixel_circle(CAMERA, tracker.center, half)
        error.append(abs(cy - cy0 - offset))
    assert max(error[-20:]) < 3.0, np.round(error[start + 20 :], 1)


def test_a_re_fitted_frame_keeps_its_observation_and_its_window_frame():
    """The frame whose window moved is still an observation, in the frame it was tracked in.

    Dropping it cost the offline refinement 562 of trial 004's 3013 rows and left the
    debug video's window panel blank over the whole episode.
    """
    move, start, over = 25.0, 220, 150

    def drift(i):
        return shifted(move * float(np.clip((i - start) / over, 0.0, 1.0)))

    images, _, _ = sequence(start + over + 20, drift=drift)
    tracker = Tracker(config(), *SIZE, TrackParams(center_watch=True))
    kept = []  # (R_win, the version it was tracked in, the camera-frame orientation)
    moved_frames = 0
    for image, _ in images:
        before = tracker.geometry_version
        result = tracker.process_frame(image)
        assert tracker.engine.last_obs is not None, "every tracked frame has a window"
        moved_frames += tracker.geometry_version > before
        # The window moves onto the ball before the frame is tracked, not after.
        assert tracker.tracked_version == tracker.geometry_version
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
