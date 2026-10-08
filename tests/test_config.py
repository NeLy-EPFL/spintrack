import tomllib
from pathlib import Path

import numpy as np
import pytest

from spintrack.config import Config, leading_comments

SAMPLE = Path(__file__).parents[1] / "examples" / "sample" / "config.toml"


def test_the_example_loads_with_its_paths_relative_to_it():
    cfg = Config.load(SAMPLE)
    assert cfg.video == str(SAMPLE.parent / "sample.mp4")
    assert cfg.camera.vfov_deg == 2.3893 and len(cfg.ball.rim) == 10
    assert cfg.ball.rim[0] == (656.5, 411.0)  # half pixels, not rounded
    assert cfg.tracking.window_px == 120 and cfg.tracking.max_bad_frames == 100
    # The two ways of giving the camera-to-animal transform agree.
    rotation = [-1.2091996, 1.2091996, -1.2091996]  # behind, level: [0, 180, 0]
    rotation = Config(camera={"rotation": rotation}).camera.to_animal()
    assert np.allclose(cfg.camera.to_animal(), rotation, atol=1e-6)


def test_mistakes_are_errors_that_say_what_was_meant(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "vfov = 3\n"
        "[camera]\nposition_deg = [0, 180, 0]\nrotation = [0, 0, 0]\n"
        "[ball]\nrim = [[1, 2], [3, 4]]\n"
        "[tracking]\nq_factor = 6\nwindow_px = 0\n"
    )
    with pytest.raises(ValueError) as error:
        Config.load(path)
    lines = str(error.value).splitlines()[1:]
    assert lines[0].startswith("  camera: position_deg and rotation are alternatives")
    assert lines[1].startswith("  ball.rim: at least three rim points")
    assert lines[2] == "  tracking.window_px: Input should be greater than 0"
    assert lines[3].startswith("  tracking.q_factor: unknown key; [tracking] takes ")
    assert lines[4] == "  vfov: unknown key; did you mean camera.vfov_deg?"
    with pytest.raises(ValueError, match="TOML"):
        Config.load(tmp_path / "config.txt")


def test_save_writes_what_load_reads_relative_to_the_new_file(tmp_path):
    cfg = Config.load(SAMPLE)
    cfg.mask.ignore = [[(1, 2), (3, 4), (5, 7)]]
    cfg.tracking.initial_map = str(tmp_path / "maps" / "rig.npz")
    cfg.output.name = 'trial "3" \\ \u00e9\x7f'
    (tmp_path / "run").mkdir()
    path = cfg.save(tmp_path / "run" / "config.toml", ["# a note"], full=True)
    text = path.read_text()
    assert text.startswith("# a note\n\nvideo = ")
    assert 'initial_map = "../maps/rig.npz"' in text
    assert "    [656.5, 411.0],\n" in text  # one rim point per line
    assert Config.load(path) == cfg
    assert leading_comments(path) == ["# a note"]
    # Without `full`, only what differs from the defaults.
    sparse = tomllib.loads(Config(camera={"fisheye": True}).save(path).read_text())
    assert sparse == {"camera": {"fisheye": True}}
