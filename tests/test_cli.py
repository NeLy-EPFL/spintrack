"""End-to-end smoke test: a tiny synthetic ball video through the CLI to a .dat file."""

import sys

import cv2
import numpy as np

from spintrack.camera import PinholeCamera
from spintrack.cli import main
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.io.dat import N_COLUMNS, read_dat

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_engine import make_texture

W, H = 160, 120
CAM = PinholeCamera(W, H, 40.0)
CENTRE = normalize(np.array([0.0, 0.0, 1.0]))
HALF = 0.28


def render_frame(texture, R, rng):
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = CAM.rays(xs, ys)
    radius = np.sin(HALF)
    b = rays @ CENTRE
    disc = b * b - (1 - radius * radius)
    hit = disc >= 0
    t = b - np.sqrt(np.where(hit, disc, 0))
    normals = normalize(t[..., None] * rays - CENTRE)
    albedo = texture((normals @ R).reshape(-1, 3)).reshape(H, W)
    shade = 0.4 + 0.6 * np.clip(normals @ normalize(np.array([-0.3, -0.5, -0.8])), 0, 1)
    img = np.where(hit, 30 + 220 * albedo * shade, 40.0) + rng.normal(0, 2, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_cli_run_writes_fictrac_compatible_dat(tmp_path):
    rng = np.random.default_rng(0)
    texture = make_texture(rng, n_blobs=80)
    writer = cv2.VideoWriter(
        str(tmp_path / "ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (W, H)
    )
    R = np.eye(3)
    w_true = np.array([0.0, 0.03, 0.01])
    n = 40
    for i in range(n):
        if i > 0:
            R = rotvec_to_matrix(w_true) @ R
        frame = render_frame(texture, R, rng)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    cfg = Config(
        src_fn="ball.mp4", vfov=40.0, q_factor=6, roi_c=list(CENTRE), roi_r=HALF
    )
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.save(tmp_path / "config.txt")
    out = tmp_path / "out.dat"
    assert main(["run", str(tmp_path / "config.txt"), "--out", str(out)]) == 0
    dat = read_dat(out)
    assert dat.shape == (n, N_COLUMNS)
    assert list(dat[:, 0].astype(int)) == list(range(n))
    # Camera-frame increments match the simulated rotation (frame 0 is the reference).
    err = [
        np.degrees(
            np.linalg.norm(
                matrix_to_rotvec(
                    rotvec_to_matrix(row[1:4]) @ rotvec_to_matrix(w_true).T
                )
            )
        )
        for row in dat[2:]
    ]
    assert np.median(err) < 0.3, err
    # Identity camera-to-lab: lab columns equal camera columns; heading integrates -dr_z.
    assert np.allclose(dat[:, 5:8], dat[:, 1:4])
    assert np.isclose(dat[-1, 16], (-dat[1:, 7].sum()) % (2 * np.pi), atol=1e-6)
