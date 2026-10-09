"""The command line: a synthetic ball with known rotation, and the committed clip."""

import json
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
from spintrack.io.records import COLUMNS, TABLE_COLUMNS
from spintrack.pipeline import open_config

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
    argv = ["run", str(tmp_path / "config.toml"), "--out", str(out), "--no-live"]
    assert main(argv) == 0
    dat = pl.read_parquet(out / "tracks.parquet", columns=list(COLUMNS)).to_numpy()
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


def test_overrides_change_the_config_and_name_a_wrong_key(tmp_path, caplog):
    """`--set` goes over the config; a typo is an error with a suggestion."""
    path = tmp_path / "config.toml"
    path.write_text(
        'video = "ball.mp4"\n\n[camera]\nvfov_deg = 40\nrotation = [0, 1, 0]\n'
    )
    cfg, _ = open_config(
        path,
        overrides=["camera.position_deg=0,180,0", "tracking.window_px=80"],
    )
    # The position replaces the config's rotation, its alternative.
    assert cfg.camera.position_deg == (0.0, 180.0, 0.0) and cfg.camera.rotation is None
    assert cfg.tracking.window_px == 80 and cfg.camera.vfov_deg == 40
    cfg, _ = open_config(path, overrides=["output.name=2024", "ball.rim=none"])
    assert cfg.output.name == "2024" and cfg.ball.rim == []
    with caplog.at_level(logging.ERROR, logger="spintrack"):
        argv = ["run", str(tmp_path / "ball.mp4"), "--set", "tracking.windw_px=80"]
        assert main([*argv, "--no-live"]) == 1
    assert "did you mean tracking.window_px?" in caplog.records[-1].getMessage()


def run_sample(out, *extra):
    argv = ["run", str(SAMPLE), "--max-frames", "40", "--out", str(out), *extra]
    return main([*argv, "--no-live"])


def test_run_names_its_outputs_and_keeps_them(tmp_path):
    assert run_sample(tmp_path, "--debug-video") == 0
    names = {"tracks.parquet", "summary.json", "debug.mp4", "log.txt", "config.toml"}
    assert {p.name for p in tmp_path.iterdir()} == names
    table = pl.read_parquet(tmp_path / "tracks.parquet")
    assert table.columns == list(TABLE_COLUMNS) and table.height == 40
    cap = cv2.VideoCapture(str(tmp_path / "debug.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()
    # The log holds what the terminal showed, down to the last line.
    lines = (tmp_path / "log.txt").read_text().splitlines()
    assert lines[0].startswith("spintrack ") and "run quality" in "\n".join(lines)
    assert lines[-1].startswith("wrote tracks.parquet")
    # A second run refuses to replace them, unless asked to.
    before = (tmp_path / "tracks.parquet").stat().st_mtime_ns
    assert run_sample(tmp_path) == 1
    assert (tmp_path / "tracks.parquet").stat().st_mtime_ns == before
    assert run_sample(tmp_path, "--force") == 0


def test_run_writes_the_config_it_ran(tmp_path):
    """config.toml holds the command line's changes, and running it repeats the run."""
    assert run_sample(tmp_path / "a", "--set", "camera.position_deg=30,180,0") == 0
    written = Config.load(tmp_path / "a" / "config.toml")
    assert written.camera.position_deg == (30.0, 180.0, 0.0)
    assert written.video == str(SAMPLE.parent / "sample.mp4")
    argv = ["--max-frames", "40", "--out", str(tmp_path / "b"), "--no-live"]
    assert main(["run", str(tmp_path / "a" / "config.toml"), *argv]) == 0
    a, b = (pl.read_parquet(tmp_path / x / "tracks.parquet") for x in "ab")
    assert a.drop("wall_ms").equals(b.drop("wall_ms"))


def test_run_writes_a_folder_next_to_the_video(tmp_path):
    (tmp_path / "config.toml").write_text(SAMPLE.read_text())
    for name in ("sample.mp4", "clip.mp4"):
        (tmp_path / name).symlink_to(SAMPLE.parent / "sample.mp4")
    # A video with the rig's config: the folder goes next to the video.
    argv = ["-c", str(tmp_path / "config.toml"), "--max-frames", "20", "--no-live"]
    assert main(["run", str(tmp_path / "clip.mp4"), *argv]) == 0
    assert (tmp_path / "clip_spintrack" / "tracks.parquet").exists()
    # The folder's config names the video it ran, relative to the folder.
    written = (tmp_path / "clip_spintrack" / "config.toml").read_text()
    assert 'video = "../clip.mp4"' in written
    # A config names the video in `video`, and the folder after `output.name` if set.
    with (tmp_path / "config.toml").open("a") as f:
        f.write('\n[output]\nname = "trial"\n')
    argv = ["run", str(tmp_path / "config.toml"), "--max-frames", "20", "--no-live"]
    assert main(argv) == 0
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
    argv = [
        "run",
        str(tmp_path / "missing.mp4"),
        "--config",
        str(SAMPLE),
        "--no-live",
    ]
    with caplog.at_level(logging.ERROR, logger="spintrack"):
        assert main([*argv, "--out", str(tmp_path)]) == 1
    (record,) = caplog.records
    assert record.getMessage().startswith("error: ") and not record.exc_info
    assert not any(tmp_path.iterdir())


def test_a_video_needs_the_run_command(capsys):
    """`spintrack VIDEO` is not a command: the error says to use `spintrack run`."""
    assert main(["clip.mp4"]) == 2
    assert "spintrack run clip.mp4" in capsys.readouterr().err


def test_config_keys_go_through_set(capsys):
    """A positional KEY=VALUE is refused, with the `--set` that means it."""
    assert main(["run", "clip.mp4", "tracking.window_px=80"]) == 2
    assert "--set tracking.window_px=80" in capsys.readouterr().err


def test_doctor_reports_as_json(capsys, monkeypatch):
    """The report has octacam doctor's shape; this installation has no error."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # leave the GPU to other work
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["spintrack_version"] == spintrack.__version__
    statuses = {f["status"] for s in report["sections"] for f in s["findings"]}
    assert statuses <= {"ok", "info", "warn"}
