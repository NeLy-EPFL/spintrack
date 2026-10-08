import os

import pytest

# Before the test modules load numpy: OpenBLAS threads only spin on the tests' small
# matrix products (one follower test: 31 s wall and 394 s CPU with them, 22 s and 18 s
# without).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@pytest.fixture(autouse=True)
def no_sam_download(monkeypatch):
    """Keep tests from loading SAM 3, a 3.4 GB download.

    Detection then falls back to the classical detector, unless a test patches
    `segment.ball_masks` with masks of its own.
    """
    from spintrack import segment

    def unavailable():
        raise segment.SegmenterUnavailable("SAM 3 is not loaded in tests")

    monkeypatch.setattr(segment, "_model", unavailable)
