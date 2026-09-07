"""Ball detection against the synthetic benchmark scenes' ground truth.

Skipped unless `benchmarks/data/` has been generated
(`uv run --group bench python benchmarks/bench.py synth`).
"""

from pathlib import Path

import numpy as np
import pytest

from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.detect import DetectionError, detect_ball, sample_frames
from spintrack.sphere import fit_ball

# The detector measures every scene to within 1.24% of radius and 0.13 deg of centre; the
# worst of both is `offaxis`, whose silhouette is an ellipse with only half its rim in
# frame. The bar is set from that worst case with room to spare, since a radius error goes
# into the reported rotation speed slightly worse than 1:1.
RADIUS_TOL = 0.02
CENTRE_TOL = 0.2  # degrees

# `holder_shadow_cut` is held out of the sweep below and asserted separately: the holder
# shadow leaves the bottom of the ball as dark as the background, so the silhouette the
# detector works from is not there to be found. Refusing is the right answer, and the rig
# this scene copies hand-annotates the ball (`roi_circ`) rather than detecting it.
UNDETECTABLE = "holder_shadow_cut"
ALL_SCENES = sorted(p.parent for p in Path("benchmarks/data").glob("*/truth.npz"))
SCENES = [p for p in ALL_SCENES if p.name != UNDETECTABLE]


@pytest.mark.skipif(not SCENES, reason="benchmarks/data not generated")
@pytest.mark.parametrize("scene", SCENES, ids=lambda p: p.name)
def test_detect_matches_ground_truth(scene):
    truth = np.load(scene / "truth.npz")
    cfg = Config.load(scene / "config.txt")
    frames = sample_frames(str(scene / "video.mp4"), 100, 300)
    height, width = frames[0].shape
    detection = detect_ball(frames)
    camera = source_camera(width, height, cfg.vfov, cfg.fisheye)
    points = np.asarray(detection.rim_points(16), dtype=float).reshape(-1, 2)
    centre, half_angle = fit_ball(points, camera)

    radius_error = half_angle / float(truth["half_angle"]) - 1.0
    centre_error = np.degrees(np.arccos(np.clip(centre @ truth["centre"], -1.0, 1.0)))
    assert abs(radius_error) < RADIUS_TOL, f"radius {100 * radius_error:+.2f}%"
    assert centre_error < CENTRE_TOL, f"centre {centre_error:.3f} deg"


@pytest.mark.skipif(
    not any(p.name == UNDETECTABLE for p in ALL_SCENES),
    reason=f"{UNDETECTABLE} not generated",
)
def test_detector_refuses_a_ball_the_holder_shadow_has_eaten():
    """The same geometry without the shadow detects to 0.2% of radius; with it there is
    no silhouette to fit, and the detector must say so rather than guess."""
    scene = Path("benchmarks/data") / UNDETECTABLE
    frames = sample_frames(str(scene / "video.mp4"), 100, 300)
    with pytest.raises(DetectionError):
        detect_ball(frames)
