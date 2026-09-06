"""Notice when the ball itself has moved, and follow it without losing the surface map.

A ball that sinks in its holder mid-recording leaves the tracking window looking at the
wrong part of the image, and every rotation reported from then on is meaningless. It shows
up as a sustained rise in the photometric cost, so `CentreWatch` keeps a long baseline of
accepted costs and calls for a fresh look when the recent ones sit well above it.

That fresh look re-measures the ball in the image rather than searching the cost for an
image shift. The cost cannot find the shift: by the time the ball has moved far enough to
notice, the solver has absorbed the translation into the orientation and built the map
through the displaced window, so the two agree with each other. Measured on the `ball_drop`
scene at frame 500, where the ball sits 60 px low: the cost is 0.512 at no shift and rises
monotonically to 1.16 at the true 60 px, i.e. the minimum is at exactly the wrong place.
Measuring the ball, which never looks at the map, puts the centre within 1-3 px throughout.

Two different looks do that, because the two jobs need different things.

- **Has the ball moved?** `detect.detect_ball`, which finds the ball wherever it is. It has
  to: by the time the cost has risen the ball is already tens of pixels away, and a search
  seeded at the old centre does not recover it. On trial 004 a seeded search started 20 px
  high walks *away* from the ball, onto a circle through the fly's back - the animal sticks
  out past the rim, and only the hull RANSAC inside `detect_ball` throws that out. It runs
  on a downscaled, strided buffer, which is 12-15 ms rather than 250 and no less reliable,
  because "has it moved by 5% of the radius" needs no sub-pixel accuracy.
- **Where is it now?** `detect.relocate_ball`, seeded on the carried estimate with the
  radius already known and a rim band a few percent of it wide. 3-4 ms, so it can run on
  every frame - which is what the accuracy needs, since a window left one frame behind
  turns the ball's own movement into a rotation of about `d / r` radians. Its capture range
  is only about 10 px, hence the first look above; a full detection re-anchors it every
  `ANCHOR_EVERY` frames so it cannot drift away unnoticed.

Every measurement is a displacement from one taken early in the same run, never an absolute
position, so a systematic difference between an estimator and whatever fitted the config's
circle cancels: while the ball is still, the window does not move at all.

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

import cv2
import numpy as np

log = logging.getLogger("spintrack")

HISTORY = 1000  # accepted costs kept for the baseline
BASELINE_SKIP = 50  # most recent costs left out, so an episode cannot raise its own bar
BASELINE_EVERY = 25  # frames between recomputations of it, which is a median of ~950
ARM_AFTER = 200  # frames tracked before the watch has a baseline worth trusting
RECENT = 20  # accepted costs whose median decides "elevated"
ELEVATED_FACTOR = 2.0
CALM_FACTOR = 1.3
# The watch stays alert until the cost has settled for this long *and* the ball it is
# following has stopped moving - a ball that has moved once usually goes on moving, and
# waiting for the cost to rise again each time leaves the window 10-20 px behind it.
# "Stopped" is having stayed inside half of `MIN_SHIFT` over the last `STILL_FRAMES`,
# read off the displacements the window was actually given. Two details matter. The
# filter's own velocity state is too smooth to answer this - it reads a ball still moving
# at 0.26 px per frame as stopped - and the measure has to be the spread of the window,
# not its two ends: `ball_drop`'s ball sinks and comes back, so its start and end agree
# to 3 px while it has been 22 px away in between, and reading that as stopped leaves the
# window behind for the hundred frames it takes the cost to rise again.
CALM_FRAMES = 50
STILL_FRAMES = 200
LOST_FRAMES = 10  # consecutive dropped frames that count as elevated on their own
# Frames a follow look is taken over. It reports where the ball was in the middle of them,
# so the span is also the measurement's lag - and on the real trials a longer span is no
# quieter (2.2 px of scatter over four frames against 2.1 over eight), so it is kept short.
BUFFER = 4
LAG_FRAMES = (BUFFER - 1) / 2.0
# Half-width of the follow look's rim band, as a fraction of the radius. Narrow on purpose:
# a wider band reaches past the rim, where the animal and the holder are, and 004's episode
# diverges outright at 10% or 20%. The radius itself is the config's, not the one the
# detection fits: the freely fitted radius places the window 0.5 px closer to the ball but
# lets more looks through the residual gate, and the extra jitter costs more rotation than
# the better placement saves (0.082 deg over `ball_drop`'s episode against 0.076, and 12%
# on trial 004's total turned angle).
FOLLOW_BAND = 0.05
# The seed-independent look's buffer: this many frames, every `COARSE_STRIDE`-th, halved
# until the ball's radius would fall below `COARSE_MIN_RADIUS` (at most `COARSE_MAX_SCALE`
# times). The stride is what erases the rotating surface texture from the temporal
# quantile: over the three real trials, 12 consecutive frames are refused one attempt in
# six while 12 every second frame are never refused. The span is also this look's lag -
# it describes the middle of it - which is why the stride is not longer still: 11 frames
# is 50 px of a ball falling at its fastest.
COARSE_BUFFER = 12
COARSE_STRIDE = 2
COARSE_MIN_RADIUS = 120.0
COARSE_MAX_SCALE = 4
COARSE_LAG = (COARSE_BUFFER - 1) * COARSE_STRIDE / 2.0
# Frames between follow looks. Every frame: measured on `ball_drop`, whose ball moves
# 0.26 px per frame, re-fitting to the exact centre every frame holds the error at the
# run's baseline (0.045 deg) while doing it every second frame already costs 0.137 deg.
FOLLOW_EVERY = 1
# Frames between the seed-independent look while following, and the disagreement with
# the carried displacement that makes the watch believe it over the follow looks. Both
# guard against the follow look losing the rim quietly. The tolerance has an absolute
# floor because the two looks differ by a few pixels wherever the ball is, and a snap on
# that difference is a jump the solver reads as rotation: it cost `ball_drop` 0.005 deg.
# A follow look that is not believed brings the next detection forward to `ANCHOR_SOON`,
# since a window that has lost the rim has nothing else to go on. Feeding a *small*
# disagreement back as a slow correction was tried and dropped: the follow look pulls the
# window back to its own answer within a few frames, and holding the correction outside
# that loop left trial 004's window 7 px from the detector against 5.5 px without it.
ANCHOR_EVERY = 40
ANCHOR_SOON = 4  # ... and how soon after a follow look that could not be believed
ANCHOR_TOL_PX = 12.0
ANCHOR_TOL_RADII = 0.03
# Frames between looks while merely elevated, doubling after every look that says the ball
# has not moved, up to `MAX_LOOK_EVERY`. An elevated cost has many causes and most of them
# are not a moved ball: on trial 003, three false alarms at a fixed ten-frame cadence paid
# 73 full detections for nothing. Backing off rather than giving up after a fixed budget
# keeps a ball that moves late in a long hard stretch from being missed.
LOOK_EVERY = 10
MAX_LOOK_EVERY = 320
MAX_REFERENCE_WAIT = 320  # the same backoff for a refused reference take
# How far the ball must have moved before the window follows it. A ball whose centre
# creeps a few percent of its radius over a whole recording is not what this is for -
# trial 003 of the shakedown drifts 15 px on a 518 px ball and following it changes the
# total turned angle by 4.6%, with nothing to say which answer is better. The failure this
# is for takes the ball out of the window: `ball_drop` moves 71 px on a 175 px ball and
# real trial 004 moves 284 px on a 518 px one.
MIN_SHIFT_PX = 2.0
MIN_SHIFT_RADII = 0.05
# Looks in a row that must agree the ball has moved before the window starts following it.
# One is not enough: on `offaxis`, where the ball is half out of frame, a single look
# scatters far enough to claim a move that never happened.
CONFIRM_LOOKS = 2
# What a follow look must show before it is believed: rays that found a rim point, the
# directions those points cover, and how well they fit a circle of the known radius. The
# residual is the one that matters. A look that has latched onto the wrong edge still
# covers most of the rim - it fits a circle through the fly's back and the ball's sides -
# but its residual is 8-30 px where a good look on the same frame sits under 2.5.
MIN_RIM_FRACTION = 0.3
MIN_ARC_FRACTION = 0.25
MAX_RESIDUAL_PX = 3.0
MAX_RESIDUAL_RADII = 0.01
# Alpha-beta gains for the carried displacement, and the largest correction one look may
# make, as a fraction of the radius. What matters is not the accuracy of any one look but
# the smoothness of the track: the median per-frame error of `ball_drop` is its baseline
# plus about `j / r` radians, where `j` is the median frame-to-frame *change* in the
# window's placement error, so gains that chase every look are worse than gains that
# arrive a frame later. Swept on `ball_drop`, where the truth is exact, and against 004's
# measured trajectory: 0.085 deg over the episode (0.088 for the gains a look every tenth
# frame needed) with a median lag on 004 of 1.6 px and a worst of 14, against 8.5 and 90.
ALPHA = 0.2
BETA = 0.01
MAX_CORRECTION_RADII = 0.1


def coarse_scale(radius_px: float) -> int:
    """Downscale factor for the seed-independent look: as coarse as the ball allows."""
    scale = 1
    while scale < COARSE_MAX_SCALE and radius_px / (2 * scale) >= COARSE_MIN_RADIUS:
        scale *= 2
    return scale


def downscale(image, scale: int) -> tuple[np.ndarray, int]:
    """Halve `image` until `scale` is reached; also returns the factor actually applied.

    Halving stops early on an odd side, so that a coarse pixel always covers an exact
    block of source pixels and `scale * coarse_xy` is the source coordinate.
    """
    out = np.asarray(image)
    done = 1
    while done < scale and out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0:
        out = cv2.resize(
            out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA
        )
        done *= 2
    return out, done


@dataclass
class RefitEvent:
    """One stretch of frames over which the window followed a moving ball."""

    start: int
    end: int
    origin_px: tuple[float, float]  # where the config put the ball
    last_px: tuple[float, float]
    farthest_px: tuple[float, float]
    rim_fraction: float  # of the last look that placed the window

    def extend(self, frame: int, centre, rim_fraction: float) -> None:
        self.end = frame
        self.last_px = (float(centre[0]), float(centre[1]))
        if self._distance(self.last_px) > self._distance(self.farthest_px):
            self.farthest_px = self.last_px
        self.rim_fraction = rim_fraction

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
            "rim_fraction": self.rim_fraction,
        }


class CentreWatch:
    """Follow a ball that moves in its holder, and say where the window should be.

    Three states. Until the cost is elevated nothing is measured at all. Elevated, a
    detection every `LOOK_EVERY` frames (backing off) asks whether the ball is still where
    it started. Once two of those running agree that it is not, the window follows it on
    every frame: a rim look per frame, smoothed by an alpha-beta filter, with a detection
    every `ANCHOR_EVERY` frames to catch the rim look losing the rim.
    """

    def __init__(self, origin_px, radius_px: float, factor: float = ELEVATED_FACTOR):
        self.origin_px = np.asarray(origin_px, dtype=np.float64)
        self.radius_px = float(radius_px)
        self.min_shift = max(MIN_SHIFT_PX, MIN_SHIFT_RADII * self.radius_px)
        self.max_correction = max(MIN_SHIFT_PX, MAX_CORRECTION_RADII * self.radius_px)
        self.anchor_tol = max(ANCHOR_TOL_PX, ANCHOR_TOL_RADII * self.radius_px)
        self.max_residual = max(MAX_RESIDUAL_PX, MAX_RESIDUAL_RADII * self.radius_px)
        self.costs: deque[float] = deque(maxlen=HISTORY)
        self.frames: deque[np.ndarray] = deque(maxlen=BUFFER)
        self.coarse: deque[np.ndarray] = deque(maxlen=COARSE_BUFFER)
        self.scale = coarse_scale(self.radius_px)
        self.factor = factor
        self.n_lost = 0
        self.elevated = False
        # Displacement from where the ball started, carried forward every frame, or None
        # while the window is not following anything.
        self.offset_px: np.ndarray | None = None
        self.velocity_px = np.zeros(2)
        self._recent_offsets: deque[np.ndarray] = deque(maxlen=STILL_FRAMES)
        self.rim_fraction = 0.0
        # Where the ball is before anything has happened, measured by each of the two
        # looks. Later looks of each kind are compared against their own reference rather
        # than against the config's circle, so a systematic difference cancels: on
        # `offaxis`, where less than half the rim is in frame, comparing with the config
        # claimed a move that never happened.
        self.reference_px: np.ndarray | None = None
        self.rim_reference_px: np.ndarray | None = None
        self.polarity: float | None = None
        self._displaced = 0
        self._refused = 0
        self._previous_seen: tuple[int, np.ndarray] | None = None
        self._look_every = LOOK_EVERY
        self._next_look = 0
        self._next_anchor = 0
        self._next_reference = 0
        self._reference_wait = 1
        self._baseline: float | None = None
        self._baseline_at = -BASELINE_EVERY

    def baseline(self, frame: int = 0) -> float | None:
        """Median cost of the run so far, recomputed every `BASELINE_EVERY` frames."""
        if len(self.costs) < ARM_AFTER:
            return None
        if self._baseline is None or frame - self._baseline_at >= BASELINE_EVERY:
            history = list(self.costs)[:-BASELINE_SKIP]
            self._baseline = float(np.median(history)) if history else None
            self._baseline_at = frame
        return self._baseline

    def update(self, frame: int, gray, cost: float | None) -> np.ndarray | None:
        """Feed one frame; returns where the window should be centred, or None."""
        self.frames.append(gray)
        if frame % COARSE_STRIDE == 0:
            small, self.scale = downscale(gray, self.scale)
            self.coarse.append(small)
        if cost is None or not np.isfinite(cost):
            self.n_lost += 1
        else:
            self.n_lost = 0
            self.costs.append(float(cost))
        if self.reference_px is None:
            if self._armed() and frame >= self._next_reference:
                self._take_reference(frame)
            return None
        if not self._elevated(frame):
            self.offset_px = None
            self.velocity_px = np.zeros(2)
            self._recent_offsets.clear()
            self._displaced = 0
            self._refused = 0
            self._previous_seen = None
            self._look_every = LOOK_EVERY
            self._next_look = frame
            return None
        if self.offset_px is None:
            if frame >= self._next_look:
                self._trigger(frame)
        else:
            if frame >= self._next_anchor:
                self._anchor(frame)
            if frame >= self._next_look:
                self._follow(frame)
        if self.offset_px is None:
            return None
        self.offset_px = self.offset_px + self.velocity_px
        self._recent_offsets.append(self.offset_px)
        return self.origin_px + self.offset_px

    def _still(self) -> bool:
        """Has the ball the window is following stopped moving? (True if not following.)"""
        if self.offset_px is None:
            return True
        if len(self._recent_offsets) < STILL_FRAMES:
            return False
        spread = np.ptp(np.stack(self._recent_offsets), axis=0)
        return float(np.hypot(*spread)) < 0.5 * self.min_shift

    def _armed(self) -> bool:
        return (
            len(self.frames) >= BUFFER
            and len(self.coarse) >= COARSE_BUFFER
            and len(self.costs) >= ARM_AFTER
        )

    def _elevated(self, frame: int) -> bool:
        baseline = self.baseline(frame)
        if baseline is None or baseline <= 0:
            return False
        recent = list(self.costs)[-RECENT:]
        median = float(np.median(recent)) if recent else 0.0
        if self.elevated:
            # The exit test spans the last `CALM_FRAMES` and the entry test the last
            # `RECENT`, so the recent window has to be calm too - otherwise a run whose
            # cost has just risen after a quiet stretch flips state on every frame.
            calm = list(self.costs)[-CALM_FRAMES:]
            if (
                self._still()
                and len(calm) == CALM_FRAMES
                and np.median(calm) < CALM_FACTOR * baseline
                and median < CALM_FACTOR * baseline
            ):
                if self.offset_px is not None:
                    log.info(
                        "frame %d: the ball has settled %.1f px from where it started; "
                        "the window stays there",
                        frame, float(np.hypot(*self.offset_px)),
                    )  # fmt: skip
                self.elevated = False
            return self.elevated
        if median > self.factor * baseline or self.n_lost >= LOST_FRAMES:
            self.elevated = True
            log.info(
                "frame %d: cost %.4g is %.1fx the run baseline %.4g, looking for a "
                "moved ball",
                frame, median, median / baseline, baseline,
            )  # fmt: skip
        return self.elevated

    def _detect(self, frame: int):
        """Find the ball wherever it is, from the downscaled buffer; source pixels.

        Returns `(position, velocity)`, the velocity being what the previous such look
        implies. The position describes `COARSE_LAG` frames ago, so both are needed to
        say where the ball is now: on trial 004 it falls 50 px in that time.
        """
        from spintrack.detect import DetectionError, detect_ball

        try:
            found = detect_ball(list(self.coarse), max_frames=COARSE_BUFFER)
        except DetectionError as exc:
            log.debug("detection refused: %s", exc)
            return None
        seen = self.scale * np.array([found.cx, found.cy])
        previous, self._previous_seen = self._previous_seen, (frame, seen)
        velocity = np.zeros(2)
        if previous is not None and 0 < frame - previous[0] <= MAX_LOOK_EVERY:
            velocity = (seen - previous[1]) / (frame - previous[0])
        return seen, velocity

    def _rim_look(self):
        """Re-measure the rim about where the ball is believed to be."""
        from spintrack.detect import DetectionError, relocate_ball

        seed = self.rim_reference_px
        if self.offset_px is not None:
            seed = seed + self.offset_px
        try:
            found = relocate_ball(
                list(self.frames),
                float(seed[0]),
                float(seed[1]),
                self.radius_px,
                self.polarity,
                band=FOLLOW_BAND * self.radius_px,
                min_rim_fraction=MIN_RIM_FRACTION,
                min_arc_fraction=MIN_ARC_FRACTION,
            )
        except DetectionError as exc:
            log.debug("rim look refused: %s", exc)
            return None
        if not found.ok or found.residual_px > self.max_residual:
            log.debug(
                "rim look rejected: rim %.2f arc %.2f residual %.1f px",
                found.rim_fraction, found.arc_fraction, found.residual_px,
            )  # fmt: skip
            return None
        return found

    def _take_reference(self, frame: int) -> None:
        """One look of each kind early in the run, to measure later ones against."""
        found = self._detect(frame)
        if found is not None:
            seen = found[0]
            self.reference_px = self.rim_reference_px = seen
            found = self._rim_look()
            if found is not None:
                self.rim_reference_px = np.array([found.cx, found.cy])
                self.polarity = found.polarity
                log.debug(
                    "ball reference at (%.1f, %.1f), %.1f px from the config's circle",
                    *seen, float(np.hypot(*(seen - self.origin_px))),
                )  # fmt: skip
                return
            self.reference_px = self.rim_reference_px = None
        # No reference, no following: without one there is no way to tell a ball that has
        # moved from an estimator that reads a few pixels off on this scene.
        self._next_reference = frame + self._reference_wait
        self._reference_wait = min(2 * self._reference_wait, MAX_REFERENCE_WAIT)

    def _back_off(self, frame: int) -> None:
        self._next_look = frame + self._look_every
        self._look_every = min(2 * self._look_every, MAX_LOOK_EVERY)

    def _retry(self, frame: int) -> None:
        """A look that could not fit the ball at all is no evidence either way.

        So the interval it is looking on does not grow - only a run of refusals backs off,
        and one refusal in the wrong place is expensive: on trial 004 it cost 40 frames of
        an unfollowed ball falling at 4 px per frame.
        """
        self._refused += 1
        wait = min(LOOK_EVERY * 2 ** (self._refused - 1), MAX_LOOK_EVERY)
        self._next_look = frame + wait

    def _trigger(self, frame: int) -> None:
        """Elevated cost: ask whether the ball is still where it started."""
        found = self._detect(frame)
        if found is None:
            self._retry(frame)
            return
        self._refused = 0
        seen, velocity = found
        moved = float(np.hypot(*(seen - self.reference_px)))
        if moved < self.min_shift:
            # An elevated cost has many causes; only start following when the ball really
            # is somewhere else, so that a scene where it never moves is untouched. A ball
            # already half-way to the threshold is not that case, so the interval stops
            # growing there: at a 40-frame cadence, confirming a real move takes 80.
            self._displaced = 0
            if moved < 0.5 * self.min_shift:
                self._back_off(frame)
            else:
                self._next_look = frame + LOOK_EVERY
            return
        self._displaced += 1
        self._next_look = frame + LOOK_EVERY
        if self._displaced < CONFIRM_LOOKS:
            return
        # This look describes `COARSE_LAG` frames ago, so the speed the last two imply is
        # what carries it forward to now. Without that the window starts 50 px behind a
        # fast-falling ball, outside the follow look's capture range.
        log.info(
            "frame %d: the ball has moved %.1f px from where it started at %.2f px per "
            "frame; following it",
            frame, moved, float(np.hypot(*velocity)),
        )  # fmt: skip
        self.offset_px = seen - self.reference_px + velocity * COARSE_LAG
        self.velocity_px = velocity
        self._recent_offsets.clear()
        self._next_look = frame + FOLLOW_EVERY
        self._next_anchor = frame + ANCHOR_EVERY

    def _anchor(self, frame: int) -> None:
        """Check the carried displacement against a look that needs no seed."""
        self._next_anchor = frame + ANCHOR_EVERY
        found = self._detect(frame)
        if found is None:
            return
        # The detection describes `COARSE_LAG` frames ago, so it is compared against
        # where the track says the ball was then, not against where it is now, and the
        # speed the last two detections imply carries it forward again.
        seen, velocity = found
        offset = seen - self.reference_px
        carried = self.offset_px - self.velocity_px * COARSE_LAG
        gap = float(np.hypot(*(offset - carried)))
        if gap <= self.anchor_tol:
            return
        log.info(
            "frame %d: the followed centre is %.1f px from a fresh detection; taking it",
            frame, gap,
        )  # fmt: skip
        self.offset_px = offset + velocity * COARSE_LAG
        self.velocity_px = velocity

    def _follow(self, frame: int) -> None:
        """Correct the carried displacement with a fresh rim look."""
        self._next_look = frame + FOLLOW_EVERY
        found = self._rim_look()
        if found is None:
            # Coasting on the last speed drifts away from a ball that has stopped, so the
            # window freezes instead and a detection is brought forward to place it.
            self.velocity_px = np.zeros(2)
            self._next_anchor = min(self._next_anchor, frame + ANCHOR_SOON)
            return
        seen = np.array([found.cx, found.cy]) - self.rim_reference_px
        # The look describes the middle of the buffered span, so it is compared against
        # where the track says the ball was then, and the correction is shared between
        # displacement and velocity rather than snapping the position onto it.
        predicted = self.offset_px - self.velocity_px * LAG_FRAMES
        residual = np.clip(seen - predicted, -self.max_correction, self.max_correction)
        self.offset_px = self.offset_px + ALPHA * residual
        self.velocity_px = self.velocity_px + BETA * residual / FOLLOW_EVERY
        self.rim_fraction = found.rim_fraction
