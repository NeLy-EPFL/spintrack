"""Mapping the ball once, then tracking it again from the finished map.

What the second pass buys, measured on the scene below (seeds 0-3, 40 frames): the
per-frame velocity error drops by 10-15% overall and 15-25% over the opening frames,
and the map is complete from frame 0 instead of a single visible cap. What it does not
buy is less accumulated drift - the second pass inherits the first pass's drift baked
into the map, and the absolute-orientation error comes out a wash. That is what
`--refine` is for, and the two compose.
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
from test_cli import CENTRE, HALF, H, W, render_frame
from test_engine import make_texture
from test_tracker_refit import SIZE, STEP, config, sequence

PARAMS = TrackParams(centre_watch=False)


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
    """The hand-over is in memory, so nothing is resampled or rounded on the way."""
    images, _, _ = sequence(20)
    first, _, _ = track(images)
    second = Tracker(config(), *SIZE, PARAMS)
    second.prime_from(first)
    for a, b in zip(first.engine.export_map(), second.engine.export_map(), strict=True):
        assert np.array_equal(a, b)
    # The map is in the body frame both passes share, so no global search is needed.
    assert np.array_equal(second.engine.R, np.eye(3))
    assert not second.engine._needs_localisation


def test_two_pass_starts_from_a_mapped_ball():
    images, _, _ = sequence(40)
    first, cold, cold_coverage = track(images)
    second, warm, warm_coverage = track(images, first)
    assert cold_coverage < 0.35, cold_coverage
    assert warm_coverage > 0.6, warm_coverage
    # The opening frames are the ones a cold map costs.
    assert np.nanmean(warm[1:10]) < 0.9 * np.nanmean(cold[1:10]), (
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
        src_fn="ball.mp4", vfov=40.0, q_factor=6, roi_c=list(CENTRE), roi_r=HALF
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
    cfg = Config(vfov=40.0, q_factor=6, roi_c=list(CENTRE), roi_r=HALF)
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.save(tmp_path / "config.txt")
    argv = ["run", str(tmp_path / "config.txt"), "--src", "0", "--two-pass"]
    assert main(argv) == 2
    assert "--two-pass" in caplog.text
