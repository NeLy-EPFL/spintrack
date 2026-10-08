"""Regression test: stopping early must not strand the prefetch thread.

`_prefetch` only checks `stop` between frames, so a producer blocked on a full queue
never sees it. That is not merely untidy: the caller then closes the source underneath a
thread still inside `read()`, and `cv2.VideoCapture` deadlocks on a concurrent read and
release. `run(max_frames=...)` used to hang forever on exit because of it.
"""

import threading
import time

import numpy as np

from helpers import ball_config
from spintrack.io.sources import Frame
from spintrack.pipeline import run

CONFIG = ball_config((160, 120), [0.0, 0.0, 1.0], 0.25, tracking={"window_px": 40})


class SlowSource:
    """Yields frames forever, and records whether anyone is still inside `read()`."""

    width, height, fps = 160, 120, 100.0

    def __init__(self):
        self.reading = threading.Event()
        self.closed = False
        self._rng = np.random.default_rng(0)
        self._i = 0

    def read(self):
        self.reading.set()
        try:
            time.sleep(0.001)
            if self.closed:
                raise AssertionError("read() after close(): the source was torn down "
                                     "under a live prefetch thread")  # fmt: skip
            img = self._rng.integers(0, 255, (self.height, self.width), dtype=np.uint8)
            self._i += 1
            return Frame(img, self._i * 10.0, 0.0, self._i)
        finally:
            self.reading.clear()

    def close(self):
        self.closed = True


def test_max_frames_returns_and_leaves_no_thread_in_the_source():
    source = SlowSource()
    before = threading.active_count()

    stats = run(CONFIG, source, max_frames=25)

    assert stats.frames == 25
    # run() must have joined the reader, so closing the source now is safe
    assert not source.reading.is_set()
    source.close()
    deadline = time.monotonic() + 2.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == before, "prefetch thread outlived run()"
