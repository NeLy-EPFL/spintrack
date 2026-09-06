"""Notice when the ball itself has moved, and follow it without losing the surface map.

A ball that sinks in its holder mid-recording leaves the tracking window looking at the
wrong part of the image, and every rotation reported from then on is meaningless. It shows
up as a sustained rise in the photometric cost, so `CentreWatch` keeps a long baseline of
accepted costs and calls for a fresh look when the recent ones sit well above it.

That fresh look re-detects the ball in the recent frames rather than searching the cost for
an image shift. The cost cannot find the shift: by the time the ball has moved far enough
to notice, the solver has absorbed the translation into the orientation and built the map
through the displaced window, so the two agree with each other. Measured on the `ball_drop`
scene at frame 500, where the ball sits 60 px low: the cost is 0.512 at no shift and rises
monotonically to 1.16 at the true 60 px, i.e. the minimum is at exactly the wrong place.
Detection, which never looks at the map, puts the centre within 3-5 px while the ball is
moving and within 1 px once it has settled.

The map is not thrown away when the window moves. It is stored in the window frame at
`R = I`, which is the ball's own body frame, so changing the window only re-expresses the
current orientation: with `Q = R_wc_new^T R_wc_old`, `R_win` and the velocity become
`Q R_win` and `Q velocity`, and the camera-frame orientation is unchanged. See
`Tracker.refit_centre`.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("spintrack")

HISTORY = 1000  # accepted costs kept for the baseline
BASELINE_SKIP = 50  # most recent costs left out, so an episode cannot raise its own bar
ARM_AFTER = 200  # frames tracked before the watch has a baseline worth trusting
RECENT = 20  # accepted costs whose median decides "elevated"
ELEVATED_FACTOR = 2.0
CALM_FACTOR = 1.3
# The watch stays alert for this long after the last accepted re-fit as well as until the
# cost has settled: a ball that has moved once usually goes on moving, and waiting for the
# cost to rise again each time leaves the window 10-20 px behind it.
CALM_FRAMES = 50
LOST_FRAMES = 10  # consecutive dropped frames that count as elevated on their own
LOOK_EVERY = 10  # frames between re-detections while elevated
# Frames kept for the re-detection. The span is a compromise: long enough for the surface
# texture to have rotated away (24 frames is 60-70 degrees of ball rotation on the
# benchmark scenes) and short enough that a moving ball has not smeared its own silhouette,
# since the detection reports where the ball was in the middle of the span, not at its end.
BUFFER = 24
SAMPLE_EVERY = 1
# How far the ball must have moved before the window follows it. A ball whose centre
# creeps a few percent of its radius over a whole recording is not what this is for -
# trial 003 of the shakedown drifts 15 px on a 518 px ball and following it changes the
# total turned angle by 4.6%, with nothing to say which answer is better. The failure this
# is for takes the ball out of the window: `ball_drop` moves 71 px on a 175 px ball and
# real trial 004 moves 284 px on a 518 px one.
MIN_SHIFT_PX = 2.0
MIN_SHIFT_RADII = 0.05
# Looks in a row that must agree the ball has moved before the window starts
# following it. One is not enough: on `offaxis`, where the ball is half out of frame,
# a single 24-frame detection scatters far enough to claim a move that never happened.
CONFIRM_LOOKS = 2
MIN_CONFIDENCE = 0.5
# A detection over a span of frames reports where the ball was in the middle of that span,
# so while the ball is moving it lags. `locate` extrapolates forward by this many frames at
# the speed the last two detections imply, which is measured to take the lag on the
# `ball_drop` scene from 10 px to about 2.
LAG_FRAMES = BUFFER * SAMPLE_EVERY / 2
MAX_EXTRAPOLATION_PX = 30.0
# Alpha-beta gains for the carried centre. The window moves every frame, so what matters
# is not the accuracy of any one detection but the smoothness of the track: a velocity
# that jumps by 0.2 px/frame writes 0.07 deg of rotation that never happened. Taking the
# velocity straight from the last two detections does exactly that; these gains take
# `ball_drop`'s episode error from 0.100 to under 0.09 deg.
ALPHA = 0.5
BETA = 0.2


@dataclass
class RefitEvent:
    """One stretch of frames over which the window followed a moving ball."""

    start: int
    end: int
    origin_px: tuple[float, float]  # where the config put the ball
    last_px: tuple[float, float]
    farthest_px: tuple[float, float]
    confidence: float

    def extend(self, frame: int, centre, confidence: float) -> None:
        self.end = frame
        self.last_px = (float(centre[0]), float(centre[1]))
        if self._distance(self.last_px) > self._distance(self.farthest_px):
            self.farthest_px = self.last_px
        self.confidence = confidence

    def _distance(self, point) -> float:
        return float(
            np.hypot(point[0] - self.origin_px[0], point[1] - self.origin_px[1])
        )

    @property
    def max_shift_px(self) -> float:
        return self._distance(self.farthest_px)

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "origin_px": list(self.origin_px),
            "farthest_px": list(self.farthest_px),
            "max_shift_px": self.max_shift_px,
            "confidence": self.confidence,
        }


class CentreWatch:
    """Follow a ball that moves in its holder, and say where the window should be.

    Two time scales. Detection is expensive, so it runs every `LOOK_EVERY` frames; but the
    window has to follow the ball on *every* frame, because a window left where the ball
    used to be turns the ball's own movement into a rotation of about `d / r` radians per
    frame. Measured on `ball_drop`, whose ball moves 0.26 px per frame: re-fitting to the
    exact centre every frame holds the error at the run's baseline (0.045 deg), while doing
    it every second frame with the same exact centre already costs 0.137 deg. So between
    detections the centre is carried forward at the speed the detections imply.
    """

    def __init__(self, origin_px, radius_px: float, factor: float = ELEVATED_FACTOR):
        self.origin_px = np.asarray(origin_px, dtype=np.float64)
        self.min_shift = max(MIN_SHIFT_PX, MIN_SHIFT_RADII * float(radius_px))
        self.costs: deque[float] = deque(maxlen=HISTORY)
        self.frames: deque[np.ndarray] = deque(maxlen=BUFFER)
        self.factor = factor
        self.n_lost = 0
        self.elevated = False
        self.centre_px: np.ndarray | None = None
        self.velocity_px = np.zeros(2)
        self.confidence = 0.0
        # Where the detector says the ball is before anything has happened. The trigger
        # compares against this rather than against the config's circle, so a detector
        # bias cancels: on `offaxis`, where less than half the rim is in frame, the
        # detection sits a few pixels off the true centre on every frame and comparing
        # with the config would claim a move that never happened.
        self.reference_px: np.ndarray | None = None
        self._displaced = 0
        self._last_look = -LOOK_EVERY
        self._last_refit: int | None = None
        self._last_seen: tuple[int, np.ndarray] | None = None

    def baseline(self) -> float | None:
        if len(self.costs) < ARM_AFTER:
            return None
        history = list(self.costs)[:-BASELINE_SKIP]
        return float(np.median(history)) if history else None

    def update(self, frame: int, gray, cost: float | None) -> np.ndarray | None:
        """Feed one frame; returns where the window should be centred, or None."""
        if frame % SAMPLE_EVERY == 0:
            self.frames.append(gray)
        if cost is None or not np.isfinite(cost):
            self.n_lost += 1
        else:
            self.n_lost = 0
            self.costs.append(float(cost))
        if self.reference_px is None:
            if len(self.frames) >= BUFFER and len(self.costs) >= ARM_AFTER:
                self._take_reference()
            return None
        if not self._elevated(frame):
            self.centre_px = None
            self.velocity_px = np.zeros(2)
            self._last_seen = None
            self._displaced = 0
            return None
        if frame - self._last_look >= LOOK_EVERY and len(self.frames) >= BUFFER // 2:
            self._last_look = frame
            self._observe(frame)
        if self.centre_px is None:
            return None
        self.centre_px = self.centre_px + self.velocity_px
        return self.centre_px

    def note_refit(self, frame: int) -> None:
        """A re-fit was accepted: keep looking, since a ball that moved may still be."""
        self._last_refit = frame

    def _elevated(self, frame: int) -> bool:
        baseline = self.baseline()
        if baseline is None or baseline <= 0:
            return False
        if self.elevated:
            calm = list(self.costs)[-CALM_FRAMES:]
            settled = (
                self._last_refit is None or frame - self._last_refit >= CALM_FRAMES
            )
            if (
                settled
                and len(calm) == CALM_FRAMES
                and np.median(calm) < CALM_FACTOR * baseline
            ):
                self.elevated = False
            return self.elevated
        recent = list(self.costs)[-RECENT:]
        median = float(np.median(recent)) if recent else 0.0
        if median > self.factor * baseline or self.n_lost >= LOST_FRAMES:
            self.elevated = True
            log.info(
                "frame %d: cost %.4g is %.1fx the run baseline %.4g, looking for a "
                "moved ball",
                frame, median, median / baseline, baseline,
            )  # fmt: skip
        return self.elevated

    def _take_reference(self) -> None:
        """One detection early in the run, to measure later ones against."""
        from spintrack.detect import DetectionError, detect_ball

        try:
            detection = detect_ball(list(self.frames), max_frames=BUFFER)
        except DetectionError as exc:
            # No reference, no following: without one there is no way to tell a ball that
            # has moved from a detector that reads a few pixels off on this scene.
            log.debug("reference detection refused: %s", exc)
            return
        self.reference_px = np.array([detection.cx, detection.cy])
        log.debug(
            "ball reference at (%.1f, %.1f), %.1f px from the config's circle",
            *self.reference_px, float(np.hypot(*(self.reference_px - self.origin_px))),
        )  # fmt: skip

    def _observe(self, frame: int) -> None:
        """Re-detect the ball in the buffered frames and correct the carried estimate."""
        from spintrack.detect import DetectionError, detect_ball

        try:
            detection = detect_ball(list(self.frames), max_frames=BUFFER)
        except DetectionError as exc:
            log.debug("re-detection refused: %s", exc)
            return
        if detection.confidence < MIN_CONFIDENCE:
            return
        seen = np.array([detection.cx, detection.cy])
        self.confidence = detection.confidence
        previous, self._last_seen = self._last_seen, (frame, seen)
        if self.centre_px is None:
            # An elevated cost has many causes; only start following when the ball really
            # is somewhere else, so that a scene where it never moves is untouched.
            moved = float(np.hypot(*(seen - self.reference_px)))
            if moved < self.min_shift:
                self._displaced = 0
                return
            self._displaced += 1
            if self._displaced < CONFIRM_LOOKS:
                return
            log.info(
                "frame %d: the ball has moved %.1f px from where it started; following it",
                frame, moved,
            )  # fmt: skip
            self.centre_px = seen
            self.velocity_px = np.zeros(2)
            return
        if previous is None or not 0 < frame - previous[0] <= 3 * LOOK_EVERY:
            self.centre_px = seen
            self.velocity_px = np.zeros(2)
            return
        # The detection describes the middle of the buffered span, so it is compared
        # against where the track says the ball was then, and the correction is shared
        # between position and velocity rather than snapping the position onto it.
        span = frame - previous[0]
        predicted = self.centre_px - self.velocity_px * LAG_FRAMES
        residual = np.clip(
            seen - predicted, -MAX_EXTRAPOLATION_PX, MAX_EXTRAPOLATION_PX
        )
        self.centre_px = self.centre_px + ALPHA * residual
        self.velocity_px = self.velocity_px + BETA * residual / span
