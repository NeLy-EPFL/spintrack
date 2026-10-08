"""The command line: a synthetic ball with known rotation, and the committed clip."""

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

import spintrack
from helpers import make_texture, render
from spintrack.calibrate.sliders import c2a_from_angles
from spintrack.cli import main
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.io.dat import COLUMNS, N_COLUMNS, read_dat

W, H = 160, 120
CENTER = normalize(np.array([0.0, 0.0, 1.0]))
HALF = 0.28
SAMPLE = Path(__file__).parents[1] / "examples" / "sample" / "config.txt"


def render_frame(texture, R, rng):
    return render(texture, R, rng, (W, H), CENTER, HALF, occluders=False)


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
        src_fn="ball.mp4", vfov=40.0, q_factor=6, roi_c=list(CENTER), roi_r=HALF
    )
    cfg.c2a_r = [0.0, 0.0, 0.0]
    cfg.save(tmp_path / "config.txt")
    out = tmp_path / "out.dat"
    assert main(["run", str(tmp_path / "config.txt"), "--out", str(out)]) == 0
    dat = read_dat(out)
    assert dat.shape == (n, N_COLUMNS)
    assert list(dat[:, 0].astype(int)) == list(range(n))
    # The Parquet copy holds the same records at full precision.
    table = pq.read_table(out.with_suffix(".parquet"))
    assert np.allclose(np.column_stack([c.to_numpy() for c in table.columns]), dat)
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
    # Identity camera-to-lab: lab columns equal camera columns; heading integrates
    # -dr_z.
    assert np.allclose(dat[:, 5:8], dat[:, 1:4])
    assert np.isclose(dat[-1, 16], (-dat[1:, 7].sum()) % (2 * np.pi), atol=1e-6)


def test_cli_run_refuses_a_config_without_c2a(tmp_path):
    """Without c2a_r the lab-frame columns would silently be camera-frame values."""
    cfg = Config(
        src_fn="ball.mp4", vfov=40.0, q_factor=6, roi_c=list(CENTER), roi_r=HALF
    )
    cfg.save(tmp_path / "config.txt")
    out = tmp_path / "out.dat"
    assert main(["run", str(tmp_path / "config.txt"), "--out", str(out)]) == 2
    assert not out.exists()


def test_cli_calibrate_c2a_angles_writes_the_transform(tmp_path):
    cfg = Config(src_fn="ball.mp4", vfov=40.0)
    cfg.save(tmp_path / "config.txt")
    argv = ["calibrate", str(tmp_path / "config.txt"), "--c2a-angles", "0", "180", "0"]
    assert main(argv) == 0
    written = Config.load(tmp_path / "config.txt")
    assert np.allclose(written.c2a_r, c2a_from_angles(0, 180, 0))
    assert written.c2a_src == "sliders"
    assert written.extra["c2a_angles"] == [0.0, 180.0, 0.0]


def run_sample(out, *extra):
    return main(["run", str(SAMPLE), "--max-frames", "40", "--out", str(out), *extra])


def test_run_names_its_outputs_and_keeps_them(tmp_path):
    assert run_sample(tmp_path, "--debug-video") == 0
    names = {"sample.dat", "sample.parquet", "sample-summary.json", "sample-debug.mp4"}
    assert {p.name for p in tmp_path.iterdir()} == names
    dat = read_dat(tmp_path / "sample.dat")
    assert dat.shape == (40, N_COLUMNS)
    table = pq.read_table(tmp_path / "sample.parquet")
    assert table.column_names == list(COLUMNS)
    cap = cv2.VideoCapture(str(tmp_path / "sample-debug.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()
    # A second run refuses to replace them, unless asked to.
    before = (tmp_path / "sample.dat").stat().st_mtime_ns
    assert run_sample(tmp_path) == 2
    assert (tmp_path / "sample.dat").stat().st_mtime_ns == before
    assert run_sample(tmp_path, "--overwrite") == 0


def test_run_writes_next_to_the_config_or_where_out_says(tmp_path):
    video = SAMPLE.parent / "sample.mp4"
    text = SAMPLE.read_text().replace("sample.mp4", str(video))
    (tmp_path / "config.txt").write_text(text + "output_fn : trial\n")
    argv = ["run", str(tmp_path / "config.txt"), "--max-frames", "20"]
    assert main(argv) == 0
    assert read_dat(tmp_path / "trial.dat").shape == (20, N_COLUMNS)
    assert main([*argv, "--out", str(tmp_path / "sub" / "x.parquet")]) == 0
    written = {p.name for p in (tmp_path / "sub").iterdir()}
    assert written == {"x.dat", "x.parquet", "x-summary.json"}


def test_track_returns_what_run_writes(tmp_path):
    assert run_sample(tmp_path) == 0
    table = pq.read_table(tmp_path / "sample.parquet")
    written = np.column_stack([c.to_numpy() for c in table.columns])
    result = spintrack.track(SAMPLE, max_frames=40)
    assert result.columns == tuple(table.column_names)
    # All but the wall clock, to the last bit.
    assert np.array_equal(result.records[:, :-1], written[:, :-1])
    assert result.quality.n_frames == 40


def test_errors_are_one_line(tmp_path, caplog):
    argv = ["run", str(SAMPLE), "--src", str(tmp_path / "missing.mp4")]
    with caplog.at_level(logging.ERROR, logger="spintrack"):
        assert main([*argv, "--out", str(tmp_path)]) == 2
    (record,) = caplog.records
    assert record.getMessage().startswith("error: ") and not record.exc_info
    assert not any(tmp_path.iterdir())


def test_map_never_overwrites_its_input(tmp_path):
    """A FicTrac template is a `.png` too, so the default output must not be it."""
    rng = np.random.default_rng(0)
    template = tmp_path / "template.png"
    cv2.imwrite(str(template), rng.integers(0, 256, (60, 120), dtype=np.uint8))
    digest = hashlib.md5(template.read_bytes()).hexdigest()
    assert main(["map", str(template)]) == 0
    assert (tmp_path / "template-render.png").exists()
    assert main(["map", str(template), "--out", str(template)]) == 2
    assert hashlib.md5(template.read_bytes()).hexdigest() == digest
