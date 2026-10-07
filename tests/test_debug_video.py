"""Regression tests for the debug video writer.

The `opencv-python` wheels bundle an FFmpeg built without libx264, so
`cv2.VideoWriter` cannot open an H.264 stream and quietly writes MPEG-4 Part 2 instead -
a file browsers and most editors refuse to play. These check that PyAV's bundled encoder
is used instead, and that the fallback at least says what it did.
"""

import logging

import numpy as np
import pytest

from spintrack import debug_video
from spintrack.calibrate.sliders import c2a_from_angles
from spintrack.config import Config
from spintrack.debug_video import DebugCanvas, DebugVideoWriter
from spintrack.engine import StepResult, TrackParams
from spintrack.geometry import rotvec_to_matrix
from spintrack.tracker import FrameResult, Tracker

av = pytest.importorskip("av")


def probe(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
    return (
        stream.codec_context.name,
        stream.pix_fmt,
        stream.width,
        stream.height,
        frames,
    )


def write_clip(path, size=(64, 48), n=10, codec="h264"):
    writer = DebugVideoWriter(path, size, 25.0, codec)
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.close()


def test_h264_is_actually_h264(tmp_path):
    path = tmp_path / "debug.mp4"
    write_clip(path)
    codec, pix_fmt, _, _, frames = probe(path)
    assert codec == "h264"  # not mpeg4, whatever OpenCV would have done
    assert pix_fmt == "yuv420p"  # the only H.264 flavour browsers decode
    assert frames == 10  # the encoder's buffered frames were flushed on close()


def test_odd_sized_canvas_is_padded_not_rejected(tmp_path):
    path = tmp_path / "odd.mp4"
    write_clip(path, size=(65, 47))
    _, _, w, h, _ = probe(path)
    assert (w, h) == (66, 48)  # yuv420p needs even dimensions


def test_opencv_fallback_warns_that_the_file_is_unplayable(
    tmp_path, monkeypatch, caplog
):
    def no_av():
        raise ImportError("no module named 'av'")

    monkeypatch.setattr(debug_video, "_import_av", no_av)
    path = tmp_path / "fallback.mp4"
    with caplog.at_level(logging.WARNING, logger="spintrack.debug_video"):
        write_clip(path)
    assert path.exists()
    assert "cannot" in caplog.text.lower()


def test_writer_rejects_a_canvas_of_the_wrong_size(tmp_path):
    writer = DebugVideoWriter(tmp_path / "bad.mp4", (64, 48), 25.0, "h264")
    writer.write(np.zeros((48, 64, 3), np.uint8))  # so there is something to encode
    try:
        with pytest.raises(ValueError, match="writer opened for"):
            writer.write(np.zeros((48, 32, 3), np.uint8))
    finally:
        writer.close()


# ----- the animal's trail over the ball -----
#
# The trail is drawn where the ball's surface carried the animal's past contact points,
# so a transposed orientation or a mistaken "up" puts it on the far side of the ball, or
# on the wrong side of the image, and still looks like a plausible curve. This pins the
# direction end to end: a fly walking straight forward under a camera behind it leaves a
# trail running from the top of the ball down its face towards the camera.

BALL_HALF_ANGLE = np.radians(5.0)


def frame_result(i, R_cam, tracker):
    step = StepResult(True, "prev", np.zeros(3), np.eye(3), cost=0.0)
    return FrameResult(
        frame=i,
        seq=i,
        ts_ms=10.0 * i,
        w_cam=np.zeros(3),
        w_lab=np.zeros(3),
        R_cam=R_cam,
        R_lab=tracker.cam_to_lab @ R_cam @ tracker.cam_to_lab.T,
        step=step,
        values=np.zeros(25),
    )


def walking_canvas(n=90, step_deg=1.0):
    """A rendered canvas after `n` frames of a fly walking straight forward."""
    cfg = Config(
        src_fn="none",
        vfov=60.0,
        roi_c=[0.0, 0.0, 1.0],  # ball dead ahead, so the image is the ball's own frame
        roi_r=BALL_HALF_ANGLE,
        src_fps=100.0,
    )
    cfg.c2a_r = c2a_from_angles(0.0, 180.0)  # camera behind the animal, level with it
    tracker = Tracker(cfg, 640, 480, TrackParams(center_watch=False))
    canvas = DebugCanvas(tracker)  # 480 rows in and out, so panel px are source px
    gray = np.zeros((480, 640), np.uint8)
    # Walking forward turns the ball about the lab's y axis (`path.PathIntegrator`).
    w_cam = tracker.cam_to_lab.T @ (np.radians(step_deg) * np.array([0.0, 1.0, 0.0]))
    for i in range(n):
        canvas.render(gray, frame_result(i, rotvec_to_matrix(i * w_cam), tracker))
    return tracker, canvas


def test_trail_runs_from_the_animal_down_the_face_of_the_ball():
    tracker, canvas = walking_canvas()
    pts, seen = canvas.trail_points()
    cx, cy = tracker.camera.center
    radius = tracker.ball_radius_px
    assert seen[0] and not seen[-1]  # the animal itself is at the limb, not on the face
    trail = pts[seen]
    assert np.allclose(trail[:, 0], cx)  # walking forward moves it nowhere sideways
    assert np.all(np.diff(trail[:, 1]) < 0)  # newer points are nearer the top
    assert trail[-1, 1] < cy - 0.9 * radius  # the newest is up at the animal
    assert abs(trail[0, 1] - cy) < 0.05 * radius  # the oldest, 89 deg back, faces us


def test_trail_is_bounded_and_survives_a_dropped_frame():
    _, canvas = walking_canvas(n=debug_video.TRAIL_FRAMES + 50, step_deg=0.05)
    canvas.render(np.zeros((480, 640), np.uint8), None)  # a dropped frame still draws
    pts, _ = canvas.trail_points()
    assert len(pts) == debug_video.TRAIL_FRAMES
