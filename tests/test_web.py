"""The page of `spintrack run` and `spintrack gui`: what it serves, and when."""

import logging
import threading
import time
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spintrack.cli import main
from spintrack.config import Config
from spintrack.io.sources import VideoSource
from spintrack.pipeline import RunStats
from spintrack.tracker import Tracker
from spintrack.web.gui import GuiSession
from spintrack.web.live import LiveView
from spintrack.web.server import build_app

SAMPLE = Path(__file__).parents[1] / "examples" / "sample" / "config.toml"


def feed(view, n):
    """Track the sample's first `n` frames into `view`."""
    cfg = Config.load(SAMPLE)
    source = VideoSource(cfg.video)
    tracker = Tracker(cfg, source.width, source.height)
    stats = RunStats(total=n)
    view.attach(tracker, stats)
    for _ in range(n):
        frame = source.read()
        result = tracker.process_frame(frame.image, frame.ts_ms)
        stats.frames += 1
        view.frame(frame, result)
    source.close()


def test_the_page_captures_images_only_while_watched():
    view = LiveView("run")
    client = TestClient(build_app(view, token="secret"))
    assert client.get("/api/state").status_code == 403
    feed(view, 20)  # nobody watching: the numbers and the path, no pictures
    assert client.get("/api/image/frame?token=secret").status_code == 404
    state = client.get("/api/state?token=secret").json()
    assert state["progress"]["frames"] == 20 and len(state["series"]["turn"]) == 20
    assert state["path"]["n"] == 20 and state["capture"]["images"] == {}
    # The request above marks the page watched: the next frames bring pictures.
    feed(view, 5)
    images = client.get("/api/state?token=secret").json()["capture"]["images"]
    assert {"frame", "window", "map"} <= images.keys()
    response = client.get("/api/image/frame?token=secret")
    assert response.headers["content-type"] == "image/jpeg"
    assert client.get("/").status_code == 200
    client.post("/api/stop?token=secret")
    assert view.stop_requested
    view.close()


def test_run_prints_the_preview_link_and_stops_serving(tmp_path, caplog):
    argv = [str(SAMPLE), "--max-frames", "5", "--out", str(tmp_path / "out")]
    with caplog.at_level(logging.INFO, logger="spintrack"):
        assert main(argv) == 0
    (line,) = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("preview")
    ]
    assert "token=" in line and "ssh -L" in line
    # The log.txt is the run's, without the link.
    lines = (tmp_path / "out" / "log.txt").read_text().splitlines()
    assert not any(line.startswith("preview:") for line in lines)


@pytest.fixture
def gui(tmp_path):
    config = tmp_path / "rig.toml"
    text = SAMPLE.read_text().replace(
        '"sample.mp4"', f'"{SAMPLE.parent / "sample.mp4"}"'
    )
    config.write_text(text)
    session = GuiSession(str(config), None, [], config)
    view = LiveView("gui")
    client = TestClient(build_app(view, session, token="t"))
    yield session, view, client, config
    session.close()
    view.close()


def test_gui_changes_the_config_and_saves_it(gui):
    _, _, client, config = gui
    response = client.post(
        "/api/config?token=t", json={"changes": {"tracking.windw_px": 80}}
    )
    assert (
        response.status_code == 400
        and "did you mean tracking.window_px" in response.text
    )
    client.post("/api/config?token=t", json={"changes": {"tracking.window_px": 80}})
    corners = [[300, 150], [500, 150], [520, 300], [280, 300]]
    response = client.post(
        "/api/square?token=t", json={"corners": corners, "plane": "xy"}
    )
    assert response.status_code == 200
    state = client.get("/api/state?token=t").json()["gui"]
    assert state["version"] == 2 and not state["saved"]
    assert state["config"]["camera"]["position_deg"] is None  # the square replaced it
    assert client.post("/api/save?token=t", json={}).status_code == 200
    saved = tomllib.loads(config.read_text())
    assert saved["tracking"]["window_px"] == 80
    assert saved["camera"]["rotation"] == state["config"]["camera"]["rotation"]
    assert config.read_text().startswith("# Hind camera")  # its comments are kept
    assert client.get("/api/state?token=t").json()["gui"]["saved"]


def test_gui_tracks_the_clip_until_it_quits(gui):
    session, view, client, _ = gui
    client.post("/api/play?token=t", json={"speed": 0, "clip": [100, 140]})
    loop = threading.Thread(target=session.serve, args=(view,))
    loop.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get("/api/state?token=t").json()
        if state.get("progress", {}).get("frames", 0) >= 10:
            break
        time.sleep(0.05)
    assert state["last"]["frame"] >= 100  # the clip starts where asked
    client.post("/api/quit?token=t")
    loop.join(timeout=10)
    assert not loop.is_alive()
