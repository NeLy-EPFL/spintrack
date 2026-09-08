"""The reported rotation is right in absolute terms, on a ball rendered from scratch.

Every other synthetic check in this repository - the benchmark included - renders its
ball with `spintrack.camera` and `spintrack.sphere` and then asks the tracker to invert
what those modules produced. That verifies the solver but not the geometry: an error in
the camera model or in the depth the assumed radius implies would cancel exactly between
the renderer and the tracker, and the measured error would stay at hundredths of a
degree while every reported speed was wrong.

So this module imports nothing from `spintrack.camera`, `spintrack.sphere` or
`spintrack.geometry`. The projection is the pinhole definition written out, the ball is
an analytic sphere intersected with the rays, and the rotations are a Rodrigues formula
written here. What it checks is scale rather than precision: whether one radian of ball
rotation is reported as one radian, and what a wrong assumed radius does to that.
"""

import numpy as np
import pytest

from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.path import integrate_path
from spintrack.tracker import Tracker

W, H, VFOV = 168, 140, 30.0
# Pinhole, with vfov the full vertical angle.
FOCAL = (H / 2) / np.tan(np.radians(VFOV) / 2)
CENTER = np.array([0.05, -0.02, 1.0]) / np.linalg.norm([0.05, -0.02, 1.0])
# Angular radius, which is 48 source pixels here. Do not shrink it: the renderer below
# point-samples one ray per pixel, and a ball only a few tens of pixels across has
# texture at the pixel scale, which aliases into a pattern that does not turn with the
# ball. At 48 px that is worth nothing (3x and 6x supersampling agree to 1e-4); at 13 px
# it was worth 1.6 percentage points of the rotation about the optical axis, which is
# the renderer's error and not the tracker's.
HALF = np.radians(10.3)
RATE = 0.014  # rad/frame
N_FRAMES = 36
SKIP = 4  # the map is still filling over the first few frames


def rodrigues(w) -> np.ndarray:
    """`exp` of the skew-symmetric matrix of `w`, written out rather than imported."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-15:
        return np.eye(3)
    k = np.asarray(w, dtype=np.float64) / theta
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def blob_texture(rng, n=140):
    """Albedo on the unit sphere: dark caps at random directions."""
    centers = rng.normal(size=(n, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    radii = rng.uniform(0.05, 0.13, n)

    def sample(dirs):
        ang = np.arccos(np.clip(dirs @ centers.T, -1, 1))
        dark = np.clip((radii + 0.02 - ang) / 0.04, 0, 1)
        return 0.9 * np.prod(1 - 0.9 * dark, axis=1)

    return sample


def render(w_true, n=N_FRAMES, seed=0):
    """`n` frames of a sphere of angular radius `HALF` turning by `w_true` per frame.

    The sphere has radius `sin(HALF)` and sits at unit distance along `CENTER`, so it
    subtends `HALF`; shading is fixed in the camera frame and only the albedo lookup
    turns with the ball.
    """
    rng = np.random.default_rng(seed)
    texture = blob_texture(rng)
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(xs - W / 2) / FOCAL, (ys - H / 2) / FOCAL, np.ones_like(xs)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    radius = np.sin(HALF)
    b = rays @ CENTER
    disc = b * b - (1.0 - radius * radius)
    hit = disc >= 0.0
    depth = b - np.sqrt(np.where(hit, disc, 0.0))
    normals = depth[..., None] * rays - CENTER
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    light = np.array([-0.3, -0.5, -0.8]) / np.linalg.norm([-0.3, -0.5, -0.8])
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0.0, 1.0)
    flat = normals.reshape(-1, 3)
    R = np.eye(3)
    frames = []
    for i in range(n):
        if i:
            R = rodrigues(w_true) @ R
        albedo = texture(flat @ R).reshape(H, W)  # R^T applied to each normal
        img = np.where(hit, 25 + 215 * albedo * shade, 45.0) + rng.normal(
            0, 1.5, (H, W)
        )
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def track(frames, eps=0.0, c2a=(0.0, 0.0, 0.0), q_factor=8):
    """The `.dat` rows for `frames`, with the assumed radius off by a fraction `eps`."""
    cfg = Config(
        src_fn="none",
        vfov=VFOV,
        q_factor=q_factor,
        roi_c=list(CENTER),
        roi_r=HALF * (1.0 + eps),
        src_fps=100.0,
    )
    cfg.c2a_r = list(c2a)
    tracker = Tracker(cfg, W, H, TrackParams(center_watch=False))
    rows = []
    for i, img in enumerate(frames):
        res = tracker.process_frame(img, i * 10.0)
        if res is not None:
            rows.append(res.values)
    return np.asarray(rows)


def scale_and_tilt(rows, w_true):
    """Reported rotation as a multiple of `w_true`, and the angle between their axes."""
    w = np.asarray(w_true, dtype=np.float64)
    est = rows[SKIP:, 1:4]
    along = est @ w / (w @ w)
    perp = est - np.outer(along, w)
    mean_perp = np.linalg.norm(perp.mean(axis=0)) / np.linalg.norm(w)
    return float(along.mean()), float(np.degrees(np.arctan(mean_perp))), len(est)


@pytest.fixture(scope="module")
def general_turn():
    """A constant rotation about a general axis, and the frames it produces.

    Constant rate on purpose: any temporal effect (an exposure that averages the motion,
    a solver that lags) shows up as a scale error on a varying rotation but cannot touch
    a constant one, so what is left is the geometry.
    """
    w_true = np.array([0.008, -0.006, 0.009])
    return w_true, render(w_true)


def test_reported_rotation_has_the_right_absolute_scale(general_turn):
    w_true, frames = general_turn
    rows = track(frames)
    assert len(rows) == len(frames)
    scale, tilt, n = scale_and_tilt(rows, w_true)
    # Measured 1.0000 within 5e-4 over `q_factor` 4 to 16 and three ball positions in
    # the frame; half a percent here leaves room for the texture seed while still
    # catching any real regression in the camera model or the depth mapping.
    assert abs(scale - 1.0) < 0.005, (scale, n)
    assert tilt < 0.3, tilt


def test_a_wrong_radius_rescales_the_in_plane_rotation_and_not_the_image_rotation():
    """The assumed radius sets depth, and an image rotation carries no depth.

    So a relative radius error `eps` costs about `1.9 eps` of a rotation about an axis
    in the image plane and nothing at all about the optical axis. Neither
    `autofit.DISAGREEMENT_WARN` nor the ball detector's warning should ever go back to
    calling this 1:1: the in-plane cost is twice that, and the split between the two is
    why `ScaleCheck` can see a radius error in the first place.
    """
    eps = 0.05
    in_plane = RATE * np.array([1.0, 0.0, 0.0])
    optical = RATE * np.array([0.0, 0.0, 1.0])
    responses = {}
    for name, w_true in (("in-plane", in_plane), ("optical", optical)):
        frames = render(w_true)
        at_zero, _, _ = scale_and_tilt(track(frames), w_true)
        wrong, _, _ = scale_and_tilt(track(frames, eps=eps), w_true)
        responses[name] = wrong / at_zero
    slope = (1.0 - responses["in-plane"]) / eps
    assert 1.4 < slope < 2.4, responses
    assert abs(responses["optical"] - 1.0) < 0.01, responses
    assert responses["in-plane"] < responses["optical"] - 5 * eps * 0.1, responses


def test_lab_columns_apply_the_camera_to_animal_rotation_in_the_right_direction(
    general_turn,
):
    """A transposed `c2a_r` would mix forward with side and flip turning.

    Nothing else in the suite would notice: the benchmark builds its ground truth with
    the same `camera_to_lab_from_angles` the tracker inverts, and the other end-to-end
    test uses the identity. Here the transform is a quarter turn about the optical axis,
    written out, and the expected animal-frame rotation follows from it by hand.
    """
    w_true, frames = general_turn
    rows = track(frames, c2a=(0.0, 0.0, np.pi / 2))
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rows[:, 5:8], rows[:, 1:4] @ quarter_turn.T, atol=1e-12)
    # `v_lab = R v_cam` with that R: (a, b, c) in the camera becomes (-b, a, c).
    a, b, c = w_true
    expected = np.array([-b, a, c])
    assert np.allclose(rows[SKIP:, 5:8].mean(axis=0), expected, atol=2e-4), (
        rows[SKIP:, 5:8].mean(axis=0),
        expected,
    )
    # And the fictive path is where that reaches a reader: forward is `dr_lab` y, so a
    # positive one runs along +x while the negative z turns the heading (`spintrack.path`,
    # pinned against a circle in `test_path.py`). The endpoint has to be the one those
    # increments integrate to, sign and all.
    assert expected[1] > 0  # forward
    want = integrate_path(np.tile(expected, (len(rows) - 1, 1)))[-1]
    assert np.allclose(rows[-1, 14:16], want[:2], rtol=0.02)  # x, y
    assert abs((rows[-1, 16] - want[2] + np.pi) % (2 * np.pi) - np.pi) < 0.02
