import threading

import cv2
import numpy as np

from spintrack.io.sources import CallableSource, QueueSource, VideoSource, open_source


def _write_video(path, n=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 20 * i, np.uint8)
        writer.write(frame)
    writer.release()


def test_video_source_yields_gray_frames_with_timestamps(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path)
    src = open_source(path)
    assert isinstance(src, VideoSource) and (src.width, src.height) == (64, 48)
    frames = list(src)
    src.close()
    assert len(frames) == 10
    assert frames[0].image.dtype == np.uint8 and frames[0].image.shape == (48, 64)
    assert frames[-1].ts_ms > frames[0].ts_ms and frames[3].index == 3
    assert abs(int(frames[5].image.mean()) - 100) < 6


def test_queue_and_callable_sources():
    q = QueueSource(4, 3)
    threading.Thread(
        target=lambda: [q.put(np.zeros((3, 4), np.uint8), 5.0), q.put(None)]
    ).start()
    frame = q.read()
    assert frame is not None and frame.ts_ms == 5.0 and frame.image.shape == (3, 4)
    assert q.read() is None
    items = iter([(np.zeros((3, 4, 3), np.uint8), 1.0), None])
    c = CallableSource(lambda: next(items), 4, 3)
    assert c.read().image.shape == (3, 4)
    assert c.read() is None
