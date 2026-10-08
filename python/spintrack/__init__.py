"""spintrack: track the rotation of a spherical treadmill from video.

The compiled core (`spintrack._core`) holds the per-pixel kernels; this package holds
configuration, geometry, the tracking state machine, I/O and the command line.
`track(config)` runs a whole recording; `Tracker` processes one frame at a time.
"""

from spintrack._core import core_version
from spintrack.config import Config
from spintrack.engine import TrackParams
from spintrack.pipeline import Track, track
from spintrack.tracker import FrameResult, Tracker

__version__ = core_version()

__all__ = [
    "Config",
    "FrameResult",
    "Track",
    "TrackParams",
    "Tracker",
    "__version__",
    "track",
]
