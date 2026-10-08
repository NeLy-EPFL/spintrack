"""The command line: a synthetic ball with known rotation, and the committed clip."""

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import polars as pl

import spintrack
from helpers import ball_config, make_texture, render
from spintrack.cli import main
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.io.records import COLUMNS

W, H = 160, 120
CENTER = normalize(np.array([0.0, 0.0, 1.0]))
HALF = 0.28
SAMPLE = Path(__file__).parents[1] / "examples" / "sample" / "config.toml"


def render_frame(texture, R, rng):
    return render(texture, R, rng, (W, H), CENTER, HALF, occluders=False)


def test_cli_run_measures_a_known_rotation(tmp_path):
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

    cfg = ball_config((W, H), CENTER, HALF, video=str(tmp_path / "ball.mp4"))
    cfg.save(tmp_path / "config.toml")
    out = tmp_path / "out"
    assert main(["run", str(tmp_path / "config.toml"), "--out", str(out)]) == 0
    dat = pl.read_parquet(out / "tracks.parquet").to_numpy()
    assert dat.shape == (n, len(COLUMNS))
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
    # Identity camera-to-lab: lab columns equal camera columns; heading integrates
    # -dr_z.
    assert np.allclose(dat[:, 5:8], dat[:, 1:4])
    assert np.isclose(dat[-1, 16], (-dat[1:, 7].sum()) % (2 * np.pi), atol=1e-6)


def test_cli_run_refuses_a_config_or_video_without_camera_position(tmp_path, caplog):
    """Without one, the lab-frame columns would silently be camera-frame values."""
    cfg = ball_config((W, H), CENTER, HALF, video=str(tmp_path / "ball.mp4"))
    cfg.camera.rotation = None
    cfg.save(tmp_path / "config.toml")
    out = tmp_path / "out"
    assert main(["run", str(tmp_path / "config.toml"), "--out", str(out)]) == 2
    # A video alone has no config at all; the refusal says what to pass.
    video = tmp_path / "ball.mp4"
    video.symlink_to(SAMPLE.parent / "sample.mp4")
    with caplog.at_level(logging.ERROR, logger="spintrack"):
        assert main([str(video)]) == 2
    assert "--config CONFIG" in caplog.records[-1].getMessage()
    assert {p.name for p in tmp_path.iterdir()} == {"config.toml", "ball.mp4"}


def test_cli_calibrate_camera_position_replaces_a_rotation(tmp_path):
    """The position goes in place of the square's rotation, under the file's notes."""
    path = tmp_path / "config.toml"
    path.write_text(
        '# rig 2\nvideo = "ball.mp4"\n\n[camera]\nvfov_deg = 40\nrotation = [0, 1, 0]\n'
    )
    argv = ["calibrate", str(path), "--camera-position", "0", "180", "0"]
    assert main(argv) == 0
    written = Config.load(path)
    assert written.camera.position_deg == (0.0, 180.0, 0.0)
    assert written.camera.rotation is None
    assert written.video == str(tmp_path / "ball.mp4")
    assert path.read_text().startswith("# rig 2\n")


def run_sample(out, *extra):
    return main(["run", str(SAMPLE), "--max-frames", "40", "--out", str(out), *extra])


def test_run_names_its_outputs_and_keeps_them(tmp_path):
    assert run_sample(tmp_path, "--debug-video") == 0
    names = {"tracks.parquet", "summary.json", "debug.mp4", "log.txt", "config.toml"}
    assert {p.name for p in tmp_path.iterdir()} == names
    table = pl.read_parquet(tmp_path / "tracks.parquet")
    assert table.columns == list(COLUMNS) and table.height == 40
    cap = cv2.VideoCapture(str(tmp_path / "debug.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()
    # The log holds what the terminal showed, down to the last line.
    lines = (tmp_path / "log.txt").read_text().splitlines()
    assert lines[0].startswith("spintrack ") and "run quality" in "\n".join(lines)
    assert lines[-1].startswith("wrote tracks.parquet")
    # A second run refuses to replace them, unless asked to.
    before = (tmp_path / "tracks.parquet").stat().st_mtime_ns
    assert run_sample(tmp_path) == 2
    assert (tmp_path / "tracks.parquet").stat().st_mtime_ns == before
    assert run_sample(tmp_path, "--overwrite") == 0


def test_run_writes_the_config_it_ran(tmp_path):
    """config.toml holds the command line's changes, and running it repeats the run."""
    assert run_sample(tmp_path / "a", "--camera-position", "30", "180", "0") == 0
    written = Config.load(tmp_path / "a" / "config.toml")
    assert written.camera.position_deg == (30.0, 180.0, 0.0)
    assert written.video == str(SAMPLE.parent / "sample.mp4")
    argv = ["--max-frames", "40", "--out", str(tmp_path / "b")]
    assert main(["run", str(tmp_path / "a" / "config.toml"), *argv]) == 0
    a, b = (pl.read_parquet(tmp_path / x / "tracks.parquet") for x in "ab")
    assert a.drop("wall_ms").equals(b.drop("wall_ms"))


def test_run_writes_a_folder_next_to_the_video(tmp_path):
    (tmp_path / "config.toml").write_text(SAMPLE.read_text())
    for name in ("sample.mp4", "clip.mp4"):
        (tmp_path / name).symlink_to(SAMPLE.parent / "sample.mp4")
    # `spintrack VIDEO` is `spintrack run VIDEO`, here with the rig's config.
    argv = ["--config", str(tmp_path / "config.toml"), "--max-frames", "20"]
    assert main([str(tmp_path / "clip.mp4"), *argv]) == 0
    assert (tmp_path / "clip_spintrack" / "tracks.parquet").exists()
    # The folder's config names the video it ran, relative to the folder.
    written = (tmp_path / "clip_spintrack" / "config.toml").read_text()
    assert 'video = "../clip.mp4"' in written
    # A config names the video in `video`, and the folder after `output.name` if set.
    with (tmp_path / "config.toml").open("a") as f:
        f.write('\n[output]\nname = "trial"\n')
    assert main(["run", str(tmp_path / "config.toml"), "--max-frames", "20"]) == 0
    assert pl.read_parquet(tmp_path / "trial_spintrack" / "tracks.parquet").height == 20


def test_track_returns_what_run_writes(tmp_path):
    assert run_sample(tmp_path) == 0
    table = pl.read_parquet(tmp_path / "tracks.parquet")
    result = spintrack.track(SAMPLE, max_frames=40)
    assert result.columns == tuple(table.columns)
    # All but the wall clock, to the last bit and with the same types.
    assert result.to_polars().drop("wall_ms").equals(table.drop("wall_ms"))
    assert result.quality.n_frames == 40


def test_errors_are_one_line(tmp_path, caplog):
    argv = ["run", str(tmp_path / "missing.mp4"), "--config", str(SAMPLE)]
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
