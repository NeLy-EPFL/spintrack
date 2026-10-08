"""Ball detection on rendered frames: accuracy, occlusion, partial views, refusal."""

import numpy as np
import pytest

from helpers import make_texture, render
from spintrack.camera import PinholeCamera
from spintrack.detect import DetectionError, detect_ball
from spintrack.geometry import normalize, rotvec_to_matrix


def ball_circle(cam, center, half):
    from spintrack.sphere import pixel_circle

    return pixel_circle(cam, center, half)


def sequence(size, center, half, n=12, seed=0, occluders=True, spin=True):
    rng = np.random.default_rng(seed)
    texture = make_texture(rng, n_blobs=120)
    R = np.eye(3)
    out = []
    for i in range(n):
        if spin and i > 0:
            R = rotvec_to_matrix([0.02, 0.05, 0.01]) @ R
        out.append(render(texture, R, rng, size, center, half, occluders))
    return out


CASES = [
    ((320, 240), normalize(np.array([0.0, 0.0, 1.0])), 0.122),  # radius ~40 px
    ((480, 360), normalize(np.array([0.05, -0.03, 1.0])), 0.21),  # radius ~120 px
]


@pytest.mark.parametrize("size, center, half", CASES)
def test_detects_the_ball_through_occluders(size, center, half):
    frames = sequence(size, center, half)
    det = detect_ball(frames)
    cx, cy, r = ball_circle(PinholeCamera(size[0], size[1], 40.0), center, half)
    assert abs(det.r / r - 1) < 0.01, (det.r, r)
    assert np.hypot(det.cx - cx, det.cy - cy) < 0.5, (det.cx, det.cy, cx, cy)


def test_single_frame_still_works():
    size, center, half = CASES[1]
    det = detect_ball(sequence(size, center, half, n=1))
    _, _, r = ball_circle(PinholeCamera(size[0], size[1], 40.0), center, half)
    assert abs(det.r / r - 1) < 0.02, (det.r, r)


def test_ball_partly_out_of_frame():
    """40% of the rim leaves the image; the visible arc still fixes the ball.

    Checked through `fit_ball`, the way the detection is actually consumed: this far off
    axis the silhouette is an ellipse, so no circle describes it to better than 2.6 px
    and a pixel-radius comparison would be measuring the reference, not the detector.
    """
    from spintrack.sphere import fit_ball

    size, half = (480, 360), 0.21
    center = normalize(np.array([0.0, 0.30, 1.0]))
    cam = PinholeCamera(size[0], size[1], 40.0)
    _, cy, r = ball_circle(cam, center, half)
    assert cy + r > size[1], "the ball should be cut off at the bottom"
    det = detect_ball(sequence(size, center, half))
    points = np.asarray(det.rim_points(16), dtype=float).reshape(-1, 2)
    found, found_half = fit_ball(points, cam)
    assert abs(found_half / half - 1) < 0.015, (found_half, half)
    assert np.degrees(np.arccos(np.clip(found @ center, -1, 1))) < 0.2


def test_refuses_frames_without_a_ball():
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 60, (240, 320), dtype=np.uint8) for _ in range(12)]
    with pytest.raises(DetectionError):
        detect_ball(frames)


def test_rim_points_are_a_fictrac_roi_circ():
    size, center, half = CASES[1]
    det = detect_ball(sequence(size, center, half))
    points = det.rim_points(16)
    # Bins with no accepted ray are simply absent, so this is at most 16 points.
    assert 24 <= len(points) <= 32 and len(points) % 2 == 0
    assert all(isinstance(v, int) for v in points)
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    assert np.allclose(np.hypot(xy[:, 0] - det.cx, xy[:, 1] - det.cy), det.r, atol=2.0)


def test_rim_look_finds_a_known_ball_from_a_seed_a_few_pixels_off():
    """The look the moved-ball watch follows with: one frame, radius given.

    A seed most of a band off is pulled only part of the way in, so the look is
    repeated from its answer, as the watch does.
    """
    from spintrack.detect import RimLook

    size, center, half = CASES[1]
    frame = sequence(size, center, half, n=1)[0]
    cx, cy, r = ball_circle(PinholeCamera(size[0], size[1], 40.0), center, half)
    look = RimLook(r, max(8.0, 0.05 * r), max(2.0, 0.01 * r))
    for dx, dy in ((0.0, 0.0), (4.0, -3.0), (-5.0, 5.0)):
        found = look(frame, cx + dx, cy + dy)
        found = look(frame, found.cx, found.cy)  # a far seed is pulled part of the way
        assert found.ok
        assert np.hypot(found.cx - cx, found.cy - cy) < 1.0, (dx, dy, found)
        assert found.residual_px < 0.02 * r
        assert found.polarity == 1.0  # the ball is brighter than the background
    # With the radius free it finds that too, from a radius a few percent off.
    free = RimLook(1.03 * r, max(8.0, 0.05 * r), max(2.0, 0.01 * r))(
        frame, cx, cy, free_radius=True
    )
    assert abs(free.r / r - 1.0) < 0.01, free.r / r


def test_rim_look_says_so_when_the_rim_is_not_where_it_was_told():
    """A look with no rim in its band must be refused, not answered with the seed.

    The residual is what catches it: a fit that has latched onto the wrong edge still
    covers most of the rim, and only its residual gives it away.
    """
    from spintrack.detect import RimLook

    size, center, half = CASES[1]
    frame = sequence(size, center, half, n=1)[0]
    cx, cy, r = ball_circle(PinholeCamera(size[0], size[1], 40.0), center, half)
    good = RimLook(r, max(8.0, 0.05 * r), max(2.0, 0.01 * r))(frame, cx, cy)
    try:
        bad = RimLook(0.75 * r, max(8.0, 0.05 * r), max(2.0, 0.01 * r))(frame, cx, cy)
    except DetectionError:
        return
    assert not bad.ok or bad.residual_px > 5.0 * good.residual_px, (good, bad)
