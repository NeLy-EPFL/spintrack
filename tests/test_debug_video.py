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
from spintrack.debug_video import DebugVideoWriter

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
