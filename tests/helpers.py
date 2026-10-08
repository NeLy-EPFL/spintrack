"""Synthetic balls for the tests: a spotted texture, rendered as a frame or a window."""

import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.geometry import normalize
from spintrack.sphere import ball_outline, pixel_circle

LIGHT = normalize(np.array([-0.3, -0.5, -0.8]))


def ball_config(size, center, half, vfov=40.0, **tables) -> Config:
    """A config whose rim lies on the outline of the ball at `center`, `half` (rad).

    The camera frame is the animal frame (`rotation = [0, 0, 0]`).
    """
    rim = ball_outline(PinholeCamera(*size, vfov), center, half, n_points=16)
    camera = {"vfov_deg": vfov, "rotation": (0.0, 0.0, 0.0)}
    return Config(camera=camera, ball={"rim": rim.tolist()}, **tables)


def make_texture(rng, n_blobs=120):
    """`sample(dirs)`: the albedo at (N, 3) unit directions of a ball with dark spots.

    A spot darkens only within `radius + 0.02` rad of its center, so `sample` evaluates
    just those direction/spot pairs: the dense version, an `arccos` per pixel and spot,
    was most of the suite's run time.
    """
    centers = rng.normal(size=(n_blobs, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    radii = rng.uniform(0.06, 0.16, n_blobs)
    # A hair beyond the reach, so that no pair with any darkening is skipped.
    cos_reach = np.cos(radii + 0.02 + 1e-6)

    def sample(dirs):
        cos = dirs @ centers.T
        rows, cols = np.nonzero(cos > cos_reach)
        ang = np.arccos(np.clip(cos[rows, cols], -1, 1))
        dark = np.clip((radii[cols] + 0.02 - ang) / 0.04, 0, 1)
        shade = np.ones(len(dirs))
        np.multiply.at(shade, rows, 1 - 0.9 * dark)
        return 0.92 * shade

    return sample


def render_window(geom, texture, R, rng):
    """The tracking window of `geom` on a ball turned by `R`, with sensor noise."""
    dirs = geom.surface.astype(np.float64) @ R  # R^T applied row-wise
    vals = np.zeros(geom.size * geom.size)
    vals[geom.index] = texture(dirs) * 230 + 10
    img = vals.reshape(geom.size, geom.size) + rng.normal(
        0, 2.0, (geom.size, geom.size)
    )
    return np.clip(img, 0, 255).astype(np.uint8)


def render(texture, R, rng, size, center, half, occluders=True, vfov=40.0):
    """One frame of a shaded, textured ball; optionally with things over its rim.

    The occluders are the two shapes a real rig puts there: a dark blob straddling the
    silhouette (surface texture, which bites into any thresholded outline) and a bright
    bar touching the disc (the holder, which sticks out of it).
    """
    w, h = size
    cam = PinholeCamera(w, h, vfov)
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = cam.rays(xs, ys)
    radius = np.sin(half)
    b = rays @ center
    disc = b * b - (1 - radius * radius)
    hit = disc >= 0
    t = b[hit] - np.sqrt(disc[hit])
    normals = normalize(t[:, None] * rays[hit] - center)
    img = np.full((h, w), 40.0)
    shade = 0.4 + 0.6 * np.clip(normals @ LIGHT, 0, 1)
    img[hit] = 30 + 220 * texture(normals @ R) * shade
    img += rng.normal(0, 2, (h, w))
    if occluders:
        cx, cy, r = pixel_circle(cam, center, half)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = ((xx - cx) / (0.20 * r)) ** 2 + ((yy - (cy - r)) / (0.08 * r)) ** 2 < 1
        img[blob] = 20.0
        bar = (np.abs(xx - cx) < 0.25 * r) & (np.abs(yy - (cy + r)) < 0.04 * r)
        img[bar] = 250.0
    return np.clip(img, 0, 255).astype(np.uint8)
