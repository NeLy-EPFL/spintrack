"""Moving the tracking window onto a ball that has moved, without losing the map."""

import numpy as np

from helpers import ball_config, make_texture, render
from spintrack.camera import PinholeCamera
from spintrack.engine import TrackParams
from spintrack.geometry import normalize, rotvec_to_matrix
from spintrack.sphere import pixel_circle
from spintrack.tracker import Tracker

SIZE = (240, 180)
VFOV = 40.0
HALF = 0.15
CENTER = normalize(np.array([0.0, 0.0, 1.0]))
CAMERA = PinholeCamera(SIZE[0], SIZE[1], VFOV)
STEP = (0.02, 0.05, 0.01)


def config():
    return ball_config(SIZE, CENTER, HALF, VFOV)


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


def test_refit_carries_the_map():
    """A window move rebuilds the Rust core, which has to carry the map over intact.

    Getting this wrong survives the move as an array and becomes noise as a map, and the
    tracker starts dropping frames a few frames later rather than where the mistake was.
    """
    tracker = Tracker(config(), *SIZE, TrackParams(center_watch=False))
    images, texture, rng = sequence(20)
    for image, _ in images:
        tracker.process_frame(image)
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
    """The map and the ball's camera orientation survive a window move exactly."""
    tracker = Tracker(config(), *SIZE, TrackParams(center_watch=False))
    images, texture, rng = sequence(20)
    costs = [tracker.process_frame(image).step.cost for image, _ in images]
    orientation = tracker.R_wc @ tracker.engine.R  # body -> camera, the physical state
    mean, weight = (a.copy() for a in tracker.engine.export_map())
    cost_before = float(np.nanmedian(costs))

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
    """The ball's movement becomes rotation with the watch off, not with it on."""
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
            assert tracker.watch.episodes, "the watch should have noticed the move"
            assert tracker.watch.episodes[-1][2] > 10.0
            ball = tracker.ball_columns()
    assert errors[True] < 0.5 * errors[False], errors
    # The saved path is where the ball's circle really was, frame by frame.
    truth = [pixel_circle(CAMERA, drift(i), HALF)[:2] for i in range(len(ball))]
    truth = np.array(truth)
    seen = ball[:, 2] > 0.0
    miss = np.hypot(*(ball[seen, :2] - truth[seen]).T)
    assert seen.mean() > 0.8 and np.median(miss) < 0.5 and miss.max() < 2.0, (
        seen.mean(),
        np.median(miss),
        miss.max(),
    )


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
    cfg = ball_config(SIZE, CENTER, half, VFOV)
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
    cfg = ball_config(SIZE, CENTER, half, VFOV)
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


def test_a_re_fitted_frame_keeps_its_observation():
    """The frame whose window moved is still an observation.

    Dropping it left the debug video's window panel blank over the whole episode.
    """
    move, start, over = 25.0, 220, 150

    def drift(i):
        return shifted(move * float(np.clip((i - start) / over, 0.0, 1.0)))

    images, _, _ = sequence(start + over + 20, drift=drift)
    tracker = Tracker(config(), *SIZE, TrackParams(center_watch=True))
    moved_frames = 0
    for image, _ in images:
        before = tracker.geometry_version
        tracker.process_frame(image)
        assert tracker.engine.last_obs is not None, "every tracked frame has a window"
        moved_frames += tracker.geometry_version > before
    assert tracker.watch.episodes and moved_frames > 1, (
        "the watch should have followed the ball"
    )
