"""Mapping the ball once, then tracking it again from the finished map.

What the second pass buys: a map complete from frame 0 instead of a single visible cap,
the static illumination field from frame 0, and on a ball that moved the window planned
from the first pass's looks (the last test). It no longer buys velocity accuracy on the
scene below (seeds 0-3, 40 frames: the opening frames come out within a few percent of a
one-pass run either way), because the handed-over map is a prior whose weights are capped
at `map_prior_w_max`: on a lab recording it is stale (the lighting changed, and the first
pass's drift displaced it by about a degree), with its weights kept the second pass opened
at 100x the one-pass cost and matched an optical-flow cross-check worse than one pass for
300 frames, and with any cap that let stale and fresh content mix for a few frames the
first frames' increments wobbled by up to 1.8 deg. Capped just above `w_min`, the second
pass opens like a one-pass run frame for frame, and closes loops a little tighter than the
first pass; the drift that accumulates over a recording is not what it removes.
"""

import sys

import cv2
import numpy as np

from spintrack.cli import main
from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.geometry import rotvec_to_matrix
from spintrack.io.dat import read_dat
from spintrack.tracker import Tracker

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_cli import CENTER, HALF, H, W, render_frame
from test_engine import make_texture
from test_tracker_refit import SIZE, STEP, config, sequence, shifted

PARAMS = TrackParams(center_watch=False)


def track(images, primed=None):
    """`(tracker, per-frame velocity error in degrees, coverage after frame 0)`."""
    tracker = Tracker(config(), *SIZE, PARAMS)
    if primed is not None:
        tracker.prime_from(primed)
    errors, coverage = [], None
    for i, (image, _) in enumerate(images):
        result = tracker.process_frame(image)
        if i == 0:
            coverage = tracker.engine.map_coverage()
        errors.append(
            np.degrees(np.linalg.norm(result.w_cam - np.array(STEP)))
            if result is not None
            else np.nan
        )
    return tracker, np.array(errors), coverage


def test_prime_from_carries_the_map_exactly():
    """The hand-over is in memory, so nothing is resampled or rounded on the way; only
    the weights are capped, so that the second pass's frames overwrite a stale cell as
    fast as a lightly seen one."""
    images, _, _ = sequence(20)
    first, _, _ = track(images)
    second = Tracker(config(), *SIZE, PARAMS)
    second.prime_from(first)
    mean, weight = first.engine.export_map()
    mean_2, weight_2 = second.engine.export_map()
    assert np.array_equal(mean, mean_2)
    assert np.array_equal(weight_2, np.minimum(weight, PARAMS.map_prior_w_max))
    assert weight.max() > PARAMS.map_prior_w_max  # the cap did something
    # The map is in the body frame both passes share, so no global search is needed.
    assert np.array_equal(second.engine.R, np.eye(3))
    assert not second.engine._needs_localization


def test_first_frame_against_a_handed_over_map_reports_no_rotation():
    """Where the first frame lands on the stale map is an initial orientation, not a
    rotation: the .dat's frame 0 has no frame before it to have turned from."""
    images, _, _ = sequence(20)
    first, _, _ = track(images)
    second = Tracker(config(), *SIZE, PARAMS)
    second.prime_from(first)
    # Hand over a map displaced by two degrees, as the first pass's drift would leave it.
    mean, weight = second.engine.export_map()
    second.engine.core.set_map(mean, weight)
    second.engine.R = rotvec_to_matrix(np.array([0.035, 0.0, 0.0])) @ second.engine.R
    result = second.process_frame(images[0][0])
    assert result is not None and result.step.source == "map"
    assert np.array_equal(result.step.w_win, np.zeros(3))
    assert np.allclose(second.engine.velocity, 0.0)
    # The snap did happen: the orientation moved off the displaced start.
    assert np.degrees(np.linalg.norm(np.array([0.035, 0.0, 0.0]))) > 1.5
    assert not np.allclose(
        result.step.R_win, rotvec_to_matrix(np.array([0.035, 0, 0])), atol=1e-3
    )
    # And the next frame reports a rotation again.
    result = second.process_frame(images[1][0])
    assert result is not None and np.linalg.norm(result.step.w_win) > 0


def test_two_pass_starts_from_a_mapped_ball():
    images, _, _ = sequence(40)
    first, cold, cold_coverage = track(images)
    second, warm, warm_coverage = track(images, first)
    assert cold_coverage < 0.35, cold_coverage
    assert warm_coverage > 0.6, warm_coverage
    # The map is replaced at first sight, so the opening frames match a one-pass run
    # rather than beat it; what must not happen is the stale map making them worse.
    assert np.nanmean(warm[1:10]) < 1.1 * np.nanmean(cold[1:10]), (
        np.nanmean(cold[1:10]),
        np.nanmean(warm[1:10]),
    )
    assert second.engine.map_coverage() >= first.engine.map_coverage()


def test_cli_two_pass_runs_the_video_twice(tmp_path):
    rng = np.random.default_rng(0)
    texture = make_texture(rng, n_blobs=80)
    writer = cv2.VideoWriter(
        str(tmp_path / "ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (W, H)
    )
    R = np.eye(3)
    n = 30
    for i in range(n):
        if i > 0:
            R = rotvec_to_matrix([0.0, 0.03, 0.01]) @ R
        writer.write(cv2.cvtColor(render_frame(texture, R, rng), cv2.COLOR_GRAY2BGR))
    writer.release()

    cfg = Config(
        src_fn="ball.mp4", vfov=40.0, q_factor=6, roi_c=list(CENTER), roi_r=HALF
    )
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.save(tmp_path / "config.txt")
    out = tmp_path / "out.dat"
    argv = ["run", str(tmp_path / "config.txt"), "--out", str(out), "--two-pass"]
    assert main([*argv, "--save-map"]) == 0
    assert read_dat(out).shape[0] == n
    assert (tmp_path / "out-map.npz").exists()  # --save-map with no path
    summary = (tmp_path / "out-summary.json").read_text()
    assert "two-pass" in summary


def test_two_pass_needs_a_recording(tmp_path, caplog):
    """A live camera cannot be read twice, so the flag is refused rather than ignored."""
    cfg = Config(vfov=40.0, q_factor=6, roi_c=list(CENTER), roi_r=HALF)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.save(tmp_path / "config.txt")
    argv = ["run", str(tmp_path / "config.txt"), "--src", "0", "--two-pass"]
    assert main(argv) == 2
    assert "--two-pass" in caplog.text


def test_planned_window_ignores_the_animal_and_leaves_with_the_ball():
    """The rule that keeps a still ball's window still applies to the plan as well.

    Six pixels for sixty frames is the animal at the rim, and the window must not move;
    a fall is followed from the frame the ball leaves its resting place, not from the
    frame an online confirmation would have ended.
    """
    from spintrack.refit import plan_window_trajectory

    rng = np.random.default_rng(0)
    n, reference = 1500, np.array([300.0, 200.0])
    truth = np.zeros((n, 2))
    truth[300:360, 1] = 6.0
    t = np.arange(50) / 49
    truth[700:750, 1] = 120.0 * t**2
    truth[750:, 1] = 120.0
    looks = [
        (i, reference + truth[i] + rng.normal(0.0, 0.5, 2), 0.9)
        for i in range(n)
        if rng.random() > 0.05  # a look fails now and then
    ]
    window, rim, episodes = plan_window_trajectory(
        looks, n, reference, radius_px=200.0, scatter=0.5
    )
    assert np.array_equal(window[:690], np.tile(reference, (690, 1)))
    assert len(episodes) == 1, episodes
    start, stop, peak = episodes[0]
    assert 690 <= start <= 706 and 750 <= stop <= 800 and peak > 100.0, episodes
    error = np.hypot(*(window - reference - truth).T)
    assert np.median(error[700:900]) < 1.0, np.median(error[700:900])
    assert np.percentile(error[700:900], 95) < 6.0, np.percentile(error[700:900], 95)
    assert np.isfinite(rim).sum() == len(looks)


def test_second_pass_places_the_window_from_the_first_pass_looks():
    """Pass 2 knows where the ball went: its window leaves with the ball instead of
    after the online confirmation, and the opening of the move costs less rotation
    error."""
    from spintrack.refit import ScriptedWatch

    move, start, over = 25.0, 220, 150

    def drift(i):
        return shifted(move * float(np.clip((i - start) / over, 0.0, 1.0)))

    images, _, _ = sequence(start + over + 40, drift=drift)

    def run(tracker):
        errors, first_move = [], None
        for i, (image, _) in enumerate(images):
            version = tracker.geometry_version
            result = tracker.process_frame(image)
            if first_move is None and tracker.geometry_version != version:
                first_move = i
            errors.append(
                np.nan
                if result is None
                else np.degrees(np.linalg.norm(result.w_cam - np.array(STEP)))
            )
        return np.array(errors), first_move

    first = Tracker(config(), *SIZE, TrackParams(center_watch=True))
    online, moved_online = run(first)
    second = Tracker(config(), *SIZE, TrackParams(center_watch=True))
    second.prime_from(first)
    assert isinstance(second.watch, ScriptedWatch)
    assert len(second.watch.episodes) == 1, second.watch.episodes
    planned, moved_planned = run(second)
    assert moved_planned < moved_online, (moved_planned, moved_online)
    early = slice(start + 10, start + 80)
    assert np.nanmedian(planned[early]) < np.nanmedian(online[early]), (
        np.nanmedian(online[early]),
        np.nanmedian(planned[early]),
    )


def test_planned_window_keeps_up_with_a_jerk():
    """A ball that drops in a few frames is followed through the drop, not smeared.

    Trial 004's ball fell in jerks of about 50 px over ten frames; a fixed six-frame
    Gaussian over the looks put the planned window up to 24 px behind it mid-jerk and
    moved it before the ball did.
    """
    from spintrack.refit import plan_window_trajectory

    rng = np.random.default_rng(1)
    n, reference = 900, np.array([400.0, 300.0])
    truth = np.zeros((n, 2))
    for start, length, size in ((300, 10, 50.0), (340, 8, 60.0), (600, 300, -110.0)):
        phase = np.clip((np.arange(n) - start) / length, 0.0, 1.0)
        truth[:, 1] += size * 0.5 * (1.0 - np.cos(np.pi * phase))
    looks = [
        (i, reference + truth[i] + rng.normal(0.0, 0.7, 2), 0.9)
        for i in range(n)
        if rng.random() > 0.05
    ]
    window, _, _ = plan_window_trajectory(
        looks, n, reference, radius_px=500.0, scatter=0.7
    )
    error = np.hypot(*(window - reference - truth).T)
    jerks = slice(295, 355)
    assert np.percentile(error[jerks], 95) < 3.0, np.percentile(error[jerks], 95)
    assert np.array_equal(window[:280], np.tile(reference, (280, 1)))
