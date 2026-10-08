"""Follow a ball that moves in its holder, without losing the surface map.

A ball that sinks in its holder mid-recording leaves the tracking window looking at the
wrong part of the image, and the solver quietly absorbs the translation into the
rotation: a window left `d` pixels behind the ball reads the ball's own movement as a
rotation of about `d / r` radians. The photometric cost cannot say where the ball is. By
the time it has risen the map has been built through the displaced window and the two
agree with each other, and on the lab ball it is flat over +-6 px of window shift
anyway. So the window is placed by measuring the ball's silhouette, on every frame.

The measurement is `detect.relocate_ball`: the rim of a circle of the *known* radius,
fitted in a band about where the ball is predicted to be, from the current frame alone
(0.7 ms on a 1600 x 1008 frame with a 518 px ball, of the 1.2 ms per frame the whole
follower costs there). Two details make it usable on every frame.

- The fit's outlier cut is fixed at 1% of the radius, annealed down from the band,
  rather than estimated from the residuals. On trial 004 the animal's body stands past
  the rim at the top of the ball, and a scale taken from the residuals grows to
  accommodate those rays: a look seeded on the previous answer then climbs the animal's
  back a few pixels per frame and walks off the ball (250 px in 400 frames). With the
  fixed cut the same iteration stays within a couple of pixels of one place.
- The seed is the last accepted look carried by a velocity, never the smoothed
  position. The look tolerates a seed up to about half its band off and pulls larger
  errors part of the way in, so a look whose answer lands far from its seed is repeated
  once from that answer, and a look that fails is tried again from where the last look
  left the ball (it may have stopped) and, while the ball rests, from where its last few
  looks were heading (it may have started to fall). Seeded on the smoothed position,
  004's fall put the seed 12-27 px behind, the rim fraction collapsed and the ball was
  lost for 25 frames.

While the ball rests, its looks feed a slow alpha-beta filter and the window does not
move at all: the look scatters about a pixel per axis and excursions of up to 5 px last
tens of frames (the animal at the rim), and a window that chased them would turn each
one into rotation. A look farther than `INNOVATION_GATE` of the band from the filter is
not believed, and the filter's gains are slow so that a few frames of such an excursion
cannot carry it past `T_MOVE`, which on the synthetic `lab_small_ball` put 0.26 deg
spikes into a run whose ball never moved. A move is followed once `CONFIRM_FRAMES` looks
in a row lie beyond `T_MOVE_FAST` of the window; a creep, the filtered position beyond
`T_MOVE` for `CONFIRM_SLOW_FRAMES`, only nudges the window a step towards it. The
animal's body pulls the look 3-8 px toward itself for up to 80 frames at a time, and
each such excursion taken for a move cost `lab_small_ball` a tenth of its p95 error and
2 deg/min of drift. The fast
test is made on the looks themselves, counting those the filter's gate or the residual
gate refused. Made on the filter, 004's first jerk let the ball fall 45 px before the
window moved; and a ball that has dropped can show less of its rim (the frame's edge,
the legs), which on the synthetic drops raises its residual 20-50% over the resting
level.

While following, the window sits on `causal_estimate` of the looks taken since the move
began: it leans on the longest of a set of straight-line fits over the last 2 to 64
frames whose value now stays within `FOLLOW_BOUND` times `T_MOVE` of every shorter
one's and of the latest look. Where the ball moves steadily the long fits average the
looks' jitter away; through a jerk they fall behind, and the span shrinks to the last
few looks, so the window keeps up rather than lagging and then overshooting as a filter
of fixed gains does. The cut is soft (each fit fades out as it nears the bound): with a
hard one the window jumped by up to the bound whenever a fit flipped in or out, which
right after a jerk happened from one frame to the next. The fits never reach back past
the start of the move, where a line through the resting ball and the moving one ran
several pixels past a ball that had stopped. A moving ball's look is judged against its
own seed rather than the filter, its residual is learned afresh at the new place, and
following ends once the window has stayed within half of `T_MOVE` for `STILL_FRAMES`,
leaving it at the median of those frames rather than on the last of them.
`detect.detect_ball` on a downscaled buffer, the look that needs no seed, recovers a
rim look that has failed for `COAST_FRAMES` in a row; its buffer is kept while
following and while a resting ball's look is failing, so a ball that jumps farther in a
frame than any seed reaches is found again. The window is moved before the frame it was
placed for is tracked (`Tracker.process_frame`), not after.

Measured on trial 004, whose ball falls 205 px in three jerks of 50-65 px, each over
about ten frames (up to 12 px a frame), and climbs back over 400 frames, against a
lag-free reference (each frame's rim look iterated to convergence): over the jerks, the
window each frame is tracked in is 5.2 px from the ball at p95 and 18 at worst, against
42 and 49 for the filter-driven follower this replaces, which trailed the ball by eight
frames and overshot each jerk by 10 px. The optical-flow cross-check, run on a patch
carried with the ball, goes from 5.6 to 3.1 px rms along the fall over the jerks. (A
patch fixed in the image reads the fall itself, which a window that lags the ball also
reads as rotation, so that version of the check rewards lag.) With exact truth, on a
synthetic drop of the same shape at the rig's scale (`lab_jerky_drop`), the rotation
error over the jerks goes from 0.53 to 0.12 deg (median) and from 1.46 to 0.67 (p95),
and on `ball_drop`'s slow excursion from 0.060 to 0.053 (median) and 0.20 to 0.16
(p95).

Every position is a displacement from a reference taken over the first accepted looks,
never an absolute position, so a systematic difference between the look and whatever
fitted the config's circle (3.5 px on 004) does not move a still ball's window. The map
is not thrown away when the window moves: it is stored in the window frame at `R = I`,
which is the ball's own body frame, so changing the window only re-expresses the current
orientation. See `Tracker.refit_center`.

The delay that remains online is structural: the confirmation that keeps a still ball's
window still. With two passes over a recording (`Tracker.prime_from`) it goes. The
first pass records every look it believed, `plan_window_trajectory` turns them into a
window position per frame - interpolated, median-cleaned, smoothed by
`_adaptive_filter`, held at the resting level by the rule above and following each move
from the frame the ball left that level - and `ScriptedWatch` replays it in the second
pass, which therefore measures nothing and runs faster. The smoothing is zero-phase, up
to 12 frames wide where the ball rests or drifts and narrower through a jerk. The fixed
six-frame Gaussian it replaces put the window up to 24 px behind 004's ball mid-jerk and
started it moving before the ball did; over the jerks the planned window is now 2.7 px
from the ball at p95 and 5.3 at worst, against 15 and 24, and the cross-check above goes
from 3.7 to 2.0 px rms. With exact truth on `lab_jerky_drop` the rotation error over the
jerks goes from 0.34 to 0.068 deg (median) and from 0.98 to 0.14 (p95). On `ball_drop`,
a slow 0.4-radius excursion, it is 0.041 deg over the episode (median, 0.044 before)
and 0.12 (p95, 0.11 before): the legs crossing the rim step the looks by 2-3 px, and a
nine-frame median or a looser bound that ignored those steps there cost 004's own
cross-check 6-11%.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("spintrack")

# Half-width of the rim band, as a fraction of the radius with a floor for small balls,
# and the wider band used once a look has failed. The band is the look's capture range:
# on 004, at rest and mid-fall, a seed half a band off still lands within 2 px of the
# answer from a good seed, and one a full band off finds nothing.
BAND_RADII = 0.05
BAND_MIN_PX = 4.0
BAND_WIDE_RADII = 0.10
# Tukey cut of the rim fit's radial residual (see the module docstring), and how far
# from its seed a look may land before it is repeated from where it landed.
CUT_RADII = 0.01
CUT_MIN_PX = 2.0
RELOOK_BAND_FRACTION = 0.25
# What a look must show to be believed: rays that found a rim point, the directions
# those points cover, and a residual under this multiple of the running median residual
# of the run's looks. The gate is relative because the residual is a property of the
# recording (blur, compression, ball size): 1.6 px on 004, where a look pulled 5 px off
# by the animal reads 3-6. Until enough looks are in, 1% of the radius stands in for the
# median.
MIN_RIM_FRACTION = 0.3
MIN_ARC_FRACTION = 0.25
RESIDUAL_FACTOR = 1.2
RESIDUAL_HISTORY = 200
RESIDUAL_MIN_LOOKS = 20
REFERENCE_LOOKS = 60
# Gains of the alpha-beta filter that watches a resting ball. They are slow on purpose:
# a few frames of the animal pulling the look must not carry the filtered position past
# `T_MOVE`, which on the synthetic `lab_small_ball` put 0.26 deg spikes into a run whose
# ball never moved.
ALPHA = 0.1
BETA = 0.005
SCATTER_MIN_PX = 0.1
# A look farther than this fraction of the band from where it was expected is not
# believed, unless three in a row are; and how many failed looks the follower coasts
# through before the window freezes and the seed-independent detection is asked instead.
# A resting ball's look is expected where the slow filter puts it, a moving ball's at
# its own seed.
INNOVATION_GATE = 0.75
COAST_FRAMES = 8
# The bound a resting ball's displacement must pass to count as a move: the larger of an
# absolute floor, a fraction of the radius and a multiple of the median innovation seen
# while still, so it sits above the look's own scatter on any recording. (A speed
# trigger was tried and dropped: it brought the follow on 004's drop forward by two
# frames and fired on the occluder excursions of the still synthetic scenes.)
T_MOVE_PX = 4.0
T_MOVE_RADII = 0.01
T_MOVE_SCATTER = 4.0
# A move is followed once `CONFIRM_FRAMES` looks in a row (at most two frames apart)
# lie beyond `T_MOVE_FAST` (an absolute distance with a floor as a fraction of the
# radius); once the filtered position has stayed beyond `T_MOVE` for
# `CONFIRM_SLOW_FRAMES`, the window is only nudged `CREEP_STEP` of the way to it. The
# fast bound is absolute because the look's excursions toward the animal's body are
# 3-8 px on any ball measured (an 80 px synthetic one and the 518 px lab one alike),
# while a drop worth following is tens of pixels. It is judged on the looks themselves,
# not on the filter: the filter is slow so that the animal cannot drag it, and on 004's
# first jerk it let the ball fall 45 px before the window moved.
CONFIRM_FRAMES = 3
CONFIRM_SLOW_FRAMES = 100
CREEP_STEP = 0.3
T_MOVE_FAST_PX = 10.0
T_MOVE_FAST_RADII = 0.02
# While following, `causal_estimate` places the window from the believed looks of up to
# the last `FOLLOW_SPANS[-1]` frames, fading out a span as it comes within `FOLLOW_SOFT`
# of the bound (`FOLLOW_BOUND` times `T_MOVE`) of a shorter one. Against exact truth the
# soft cut at this bound took the p95 rotation error over the moves of the three jerky
# synthetic scenes from 0.70, 1.01 and 0.32 deg (a hard cut at `T_MOVE`) to 0.36, 0.42
# and 0.20, and kept `ball_drop`'s at 0.16; a tighter bound follows the 2-3 px steps the
# legs put into the looks. The next look is seeded on the looks' own velocity, an
# exponential mean of their frame-to-frame steps with weight `LOOK_VEL_EMA` on the
# newest.
FOLLOW_SPANS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
FOLLOW_BOUND = 1.25
FOLLOW_SOFT = 0.2
LOOK_VEL_EMA = 0.5
# Following ends when the window has spread less than this fraction of `T_MOVE` over
# `STILL_FRAMES` frames and moved less than `V_STILL` px per frame across them. Ending
# too early is the costlier mistake: a ball that pauses and moves on is not followed
# again until it is `T_MOVE_FAST` away, while a window that follows a stopped ball only
# takes on its looks' slow wander. At 30 frames `ball_drop` was let go at the top of its
# excursion, where it moves half a pixel in 30 frames, and its window then lagged 10 px.
STILL_FRAMES = 60
STILL_SPREAD = 0.5
V_STILL = 0.05
# Planning the window from a whole recording's looks (`plan_window_trajectory`): the
# looks are cleaned with a median over `PLAN_MEDIAN` frames, which removes isolated
# wrong looks, then smoothed by `_adaptive_filter` with Gaussians of up to `PLAN_SIGMA`
# frames, held within `PLAN_BOUND` times `T_MOVE` of the narrower ones. `PLAN_LOOKBACK`
# is how far back from the frame where the smoothed looks leave the resting level to
# look for the frame where they last sat within the look's scatter of it, so that the
# window leaves along the ball rather than stepping onto it, and `CATCHUP` is the share
# of the remaining gap to a new target (the looks where a move starts, the level where
# it ends) left open each frame.
PLAN_SIGMA = 12.0
PLAN_SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
PLAN_BOUND = 0.5
PLAN_WIDTH_EASE = 2.0
PLAN_MEDIAN = 5
PLAN_LOOKBACK = 60
CATCHUP = 0.7
# The seed-independent look's buffer: this many frames, every `COARSE_STRIDE`-th, halved
# until the ball's radius would fall below `COARSE_MIN_RADIUS` (at most
# `COARSE_MAX_SCALE` times). The stride is what erases the rotating surface texture from
# the temporal quantile. The span is also this look's lag: it describes the middle of
# the buffer.
COARSE_BUFFER = 12
COARSE_STRIDE = 2
COARSE_MIN_RADIUS = 120.0
COARSE_MAX_SCALE = 4
COARSE_LAG = (COARSE_BUFFER - 1) * COARSE_STRIDE / 2.0
RECOVER_EVERY = 4  # frames between detections while the rim look keeps failing
# A recovery detection must fit a circle of the known radius: on 004 the detections the
# animal fools are also 5-25% off in radius.
RECOVER_RADIUS_TOL = 0.02


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

    def extend(self, frame: int, center, rim_fraction: float) -> None:
        self.end = frame
        self.last_px = (float(center[0]), float(center[1]))
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


class CenterWatch:
    """Follow a ball that moves in its holder, and say where the window should be.

    A rim look per frame. While the ball rests, the looks feed a slow alpha-beta filter
    and the window stays put; it starts following when the last `CONFIRM_FRAMES` looks
    all lie beyond `T_MOVE_FAST` of it, or the filtered position has stayed beyond
    `T_MOVE` for `CONFIRM_SLOW_FRAMES`. While following, the window sits every frame on
    `causal_estimate` of the recent looks, until it has been still for `STILL_FRAMES`.
    See the module docstring.
    """

    def __init__(self, origin_px, radius_px: float):
        self.origin_px = np.asarray(origin_px, dtype=np.float64)
        self.radius_px = float(radius_px)
        self.band = max(BAND_MIN_PX, BAND_RADII * self.radius_px)
        self.band_wide = max(2.0 * BAND_MIN_PX, BAND_WIDE_RADII * self.radius_px)
        self.cut = max(CUT_MIN_PX, CUT_RADII * self.radius_px)
        self.polarity: float | None = None
        # Where the ball was before anything happened, as the rim look sees it.
        # Positions below are in the look's own coordinates; `update` returns them as
        # offsets from `origin_px`, so the config's circle and the look need not agree.
        self.reference_px: np.ndarray | None = None
        self._reference_looks: list[np.ndarray] = []
        # The resting ball's filtered position and velocity; while following, where the
        # window is, and the looks' velocity.
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.window_px: np.ndarray | None = None  # where the window is
        self.following = False
        self.rim_fraction = 0.0
        # Every look the filter believed, as `(frame, position, rim fraction)`, for a
        # second pass over the same frames to plan the window from (see `replay`).
        self.looks: list[tuple[int, np.ndarray, float]] = []
        self._last_look: tuple[int, np.ndarray] | None = None
        self._seed: np.ndarray | None = None
        self._doubtful: tuple[np.ndarray, float] | None = None
        # The believed looks the window is placed from while following, and the last
        # few looks of any kind, which are what shows that the ball has started to move.
        self._track: deque[tuple[int, np.ndarray]] = deque(maxlen=FOLLOW_SPANS[-1])
        self._recent: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=CONFIRM_FRAMES
        )
        self._failed = 0  # looks that failed in a row
        self._unbelieved = 0  # looks that landed outside the innovation gate in a row
        self._residuals: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._innovations: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._confirm = 0
        self._history: deque[np.ndarray] = deque(maxlen=STILL_FRAMES)
        self.coarse: deque[np.ndarray] = deque(maxlen=COARSE_BUFFER)
        self.scale = coarse_scale(self.radius_px)
        self._next_recovery = 0
        self._previous_seen: tuple[int, np.ndarray] | None = None

    # ----- measurements -----
    def _look(self, gray, seed, band: float) -> np.ndarray | None:
        """The rim look about `seed`, or None when it is not to be believed."""
        from spintrack.detect import DetectionError, relocate_ball

        try:
            found = relocate_ball(
                [gray],
                float(seed[0]),
                float(seed[1]),
                self.radius_px,
                self.polarity,
                band=band,
                cut=self.cut,
                min_rim_fraction=MIN_RIM_FRACTION,
                min_arc_fraction=MIN_ARC_FRACTION,
            )
        except DetectionError as exc:
            log.debug("rim look refused: %s", exc)
            return None
        if not found.ok:
            log.debug(
                "rim look rejected: rim %.2f arc %.2f",
                found.rim_fraction, found.arc_fraction,
            )  # fmt: skip
            return None
        self._residuals.append(found.residual_px)
        if len(self._residuals) >= RESIDUAL_MIN_LOOKS:
            gate = RESIDUAL_FACTOR * float(np.median(self._residuals))
        else:
            gate = self.cut
        if found.residual_px > gate:
            log.debug(
                "rim look rejected: residual %.2f px over %.2f",
                found.residual_px, gate,
            )  # fmt: skip
            self._doubtful = (np.array([found.cx, found.cy]), found.rim_fraction)
            return None
        if self.polarity is None:
            self.polarity = found.polarity
        self.rim_fraction = found.rim_fraction
        return np.array([found.cx, found.cy])

    def _measure(self, frame: int, gray) -> np.ndarray | None:
        """One measurement of the ball, seeded on the last look plus the velocity.

        When that look fails, two other seeds are tried in turn: a ball that has just
        stopped is where the last look left it, not where its velocity would carry it,
        and a resting ball that has just started to fall is where its last few looks
        were heading, which the slow filter does not know yet. Without the second, a
        ball whose band is small (4 px on a 76 px ball) was lost at the onset of a fall
        of 3-4 px a frame and never found again.
        """
        self._doubtful = None
        others = []
        if self._last_look is None:
            seed = self.pos
        else:
            last_frame, last = self._last_look
            seed = last + self.vel * (frame - last_frame)
            others.append(last)
            if not self.following and len(self._recent) > 1:
                (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
                if f1 == last_frame:
                    others.append(last + (p1 - p0) / (f1 - f0) * (frame - last_frame))
        band = self.band if self._failed == 0 else self.band_wide
        seen = self._look(gray, seed, band)
        for other in others:
            if seen is not None:
                break
            if np.hypot(*(other - seed)) > RELOOK_BAND_FRACTION * band:
                seed = other
                seen = self._look(gray, seed, band)
        if seen is not None and np.hypot(*(seen - seed)) > RELOOK_BAND_FRACTION * band:
            again = self._look(gray, seen, band)
            if again is not None:
                seen = again
        self._seed = seed
        if seen is None and not self.following and self._doubtful is None:
            self._probe(frame, gray, band)
        if seen is None:
            self._failed += 1
            return None
        self._failed = 0
        if self.following and self._last_look is not None:
            gap = frame - self._last_look[0]
            step = (seen - self._last_look[1]) / gap if gap <= COAST_FRAMES else 0.0
            self.vel = self.vel + LOOK_VEL_EMA * (step - self.vel)
        self._last_look = (frame, seen)
        return seen

    def _probe(self, frame: int, gray, band: float) -> None:
        """Look where the evidence of a move is heading, if it has run ahead.

        The looks that count towards a move (`_recent`) include ones refused for their
        residual, which seed nothing; once they lead, the believed looks are left behind
        on a ball that has dropped out of their band. This look follows the evidence
        instead, and what it finds counts towards the move but seeds nothing either.
        """
        if len(self._recent) < 2:
            return
        (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
        if self._last_look is not None and f1 == self._last_look[0]:
            return
        seen = self._look(gray, p1 + (p1 - p0) / (f1 - f0) * (frame - f1), band)
        if seen is not None:
            self._doubtful = (seen, self.rim_fraction)

    def _detect(self, frame: int):
        """Find the ball wherever it is, from the downscaled buffer; source pixels.

        Returns `(position, velocity)`, the velocity being what the previous such look
        implies. The position describes `COARSE_LAG` frames ago, so both are needed to
        say where the ball is now: on trial 004 it falls 50 px in that time.
        """
        from spintrack.detect import DetectionError, detect_ball

        if len(self.coarse) < COARSE_BUFFER:
            return None
        try:
            found = detect_ball(list(self.coarse), max_frames=COARSE_BUFFER)
        except DetectionError as exc:
            log.debug("detection refused: %s", exc)
            return None
        if abs(found.r * self.scale / self.radius_px - 1.0) > RECOVER_RADIUS_TOL:
            log.debug("detection rejected: radius %.1f px", found.r * self.scale)
            return None
        seen = self.scale * np.array([found.cx, found.cy])
        previous, self._previous_seen = self._previous_seen, (frame, seen)
        velocity = np.zeros(2)
        if previous is not None and 0 < frame - previous[0] <= 10 * COARSE_LAG:
            velocity = (seen - previous[1]) / (frame - previous[0])
        return seen, velocity

    # ----- the filter -----
    def scatter(self) -> float | None:
        """Median innovation while still: what a look does about a good prediction."""
        if len(self._innovations) < RESIDUAL_MIN_LOOKS:
            return None
        return max(SCATTER_MIN_PX, float(np.median(self._innovations)))

    def t_move(self) -> float:
        bound = max(T_MOVE_PX, T_MOVE_RADII * self.radius_px)
        scatter = self.scatter()
        if scatter is not None:
            bound = max(bound, T_MOVE_SCATTER * scatter)
        return bound

    def _take_reference(self, frame: int, gray) -> None:
        """Median of the first `REFERENCE_LOOKS` accepted looks; nothing moves yet."""
        looks = self._reference_looks
        seed = self.origin_px if not looks else np.median(looks, axis=0)
        seen = self._look(gray, seed, self.band)
        if seen is None:
            return
        looks.append(seen)
        self.looks.append((frame, seen, self.rim_fraction))
        if len(looks) < REFERENCE_LOOKS:
            return
        self.reference_px = np.median(looks, axis=0)
        self.pos = self.reference_px.copy()
        self.window_px = self.reference_px.copy()
        self._last_look = (frame, self.reference_px.copy())
        log.debug(
            "ball reference at (%.1f, %.1f), %.1f px from the config's circle",
            *self.reference_px,
            float(np.hypot(*(self.reference_px - self.origin_px))),
        )

    def _filter(self, frame: int, gray) -> None:
        """Predict, measure, correct."""
        if not self.following:
            self.pos = self.pos + self.vel
        seen = self._measure(frame, gray)
        if seen is None and self._doubtful is not None and not self.following:
            # A look refused only for its residual still says where the rim is. It
            # seeds nothing, so a look that climbs the animal cannot build on it, but it
            # counts towards a move: a ball that has dropped can show less of its rim
            # than it did at rest (the frame's edge, the legs), and its residual rises
            # with it - to 1.2-1.5 times the resting median on the synthetic drops.
            xy, rim = self._doubtful
            self._recent.append((frame, xy, rim))
        if seen is not None:
            self._recent.append((frame, seen, self.rim_fraction))
            # A resting ball's look is judged against the filter, which an animal at
            # the rim cannot drag along; a moving ball's against its own seed, since
            # the filter is not following it.
            expected = self._seed if self.following else self.pos
            off = float(np.hypot(*(seen - expected)))
            if off > INNOVATION_GATE * self.band and self._unbelieved < 3:
                self._unbelieved += 1
                return
            self._unbelieved = 0
            self.looks.append((frame, seen, self.rim_fraction))
            self._track.append((frame, seen))
            if not self.following:
                innovation = seen - self.pos
                self._innovations.append(float(np.hypot(*innovation)))
                self.pos = self.pos + ALPHA * innovation
                self.vel = self.vel + BETA * innovation
            return
        if self._failed <= COAST_FRAMES:
            return
        # The rim look has lost the ball. Coasting further would drift away from a ball
        # that has stopped, so the window freezes and the look that needs no seed is
        # asked.
        self.vel = np.zeros(2)
        if frame >= self._next_recovery:
            self._next_recovery = frame + RECOVER_EVERY
            found = self._detect(frame)
            if found is not None:
                seen, velocity = found
                self.pos = seen + velocity * COARSE_LAG
                self.vel = velocity
                self._last_look = (frame, self.pos.copy())
                self._track.clear()
                self._failed = 0
                log.info(
                    "frame %d: the rim look lost the ball; a detection puts it %.1f px "
                    "from where it started",
                    frame, float(np.hypot(*(self.pos - self.reference_px))),
                )  # fmt: skip

    def _moved_away(self, frame: int, bound: float) -> bool:
        """Whether the last `CONFIRM_FRAMES` looks, all recent, lie beyond `bound`.

        Those looks are the start of the move, and the resting ball's gate will have
        refused some of them, so they are taken into the track that the window follows.
        """
        recent = list(self._recent)
        if len(recent) < CONFIRM_FRAMES or recent[-1][0] != frame:
            return False
        if frame - recent[0][0] >= 2 * CONFIRM_FRAMES:
            return False
        if any(np.hypot(*(xy - self.window_px)) <= bound for _, xy, _ in recent):
            return False
        believed = {f for f, _, _ in self.looks[-2 * CONFIRM_FRAMES :]}
        self.looks.extend((f, xy, rim) for f, xy, rim in recent if f not in believed)
        self.looks.sort(key=lambda e: e[0])
        # The window follows from these looks alone: a line through the resting ball's
        # looks and the moving one's describes neither, and after a jump it ran several
        # pixels past a ball that had stopped.
        self._track = deque(((f, xy) for f, xy, _ in recent), FOLLOW_SPANS[-1])
        return True

    def _estimate(self, frame: int) -> np.ndarray:
        """Where the ball is on `frame`; where the window is when the looks ran out."""
        if not self._track or frame - self._track[-1][0] > COAST_FRAMES:
            return self.pos.copy()
        t = np.array([f - frame for f, _ in self._track], dtype=np.float64)
        xy = np.stack([p for _, p in self._track])
        bound = FOLLOW_BOUND * self.t_move()
        return causal_estimate(t, xy, bound, FOLLOW_SOFT * bound)

    # ----- per frame -----
    def update(self, frame: int, gray) -> np.ndarray | None:
        """Feed one frame; returns where the window should be centered, or None."""
        if self.reference_px is None:
            self._take_reference(frame, gray)
            return None
        # The seed-independent look's buffer is kept while following and while a
        # resting ball's look is failing, which is when it may be needed; a resting ball
        # seen again empties it, since it would describe a ball that has moved since.
        if self.following or self._failed > 0:
            if frame % COARSE_STRIDE == 0:
                small, self.scale = downscale(gray, self.scale)
                self.coarse.append(small)
        elif self.coarse:
            self.coarse.clear()
        self._filter(frame, gray)
        if not self.following:
            bound = self.t_move()
            fast = max(T_MOVE_FAST_PX, T_MOVE_FAST_RADII * self.radius_px)
            far = float(np.hypot(*(self.pos - self.window_px))) > bound
            self._confirm = self._confirm + 1 if far else 0
            if self._confirm >= CONFIRM_SLOW_FRAMES:
                # A slow creep nudges the window a step towards where the filter puts
                # the ball. Following it frame by frame would follow the looks' wander
                # too, which on some recordings has a 3 px standard deviation at rest,
                # and moving all the way at once put steps of up to 8 px - most of a
                # degree of rotation each - into recordings the old follower nudged.
                self._confirm = 0
                self.window_px = self.window_px + CREEP_STEP * (
                    self.pos - self.window_px
                )
                log.info(
                    "frame %d: the ball has crept %.1f px from where it started; the "
                    "window moves towards it",
                    frame, float(np.hypot(*(self.pos - self.reference_px))),
                )  # fmt: skip
                return self.origin_px + (self.window_px - self.reference_px)
            if not self._moved_away(frame, fast):
                return None
            self.following = True
            # The looks that showed the move seed the next look, carried by their own
            # velocity.
            (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
            self.vel = (p1 - p0) / (f1 - f0)
            self._last_look = (f1, p1)
            # The rim the band sees at the new place is not the one it saw at rest, so
            # the residual a look is judged by is learned again.
            self._residuals.clear()
            self._history.clear()
            self.coarse.clear()
            log.info(
                "frame %d: the ball is %.1f px from where the window looks; following "
                "it",
                frame, float(np.hypot(*(self._recent[-1][1] - self.window_px))),
            )  # fmt: skip
        self.pos = self._estimate(frame)
        self._history.append(self.pos.copy())
        if len(self._history) == STILL_FRAMES:
            history = np.stack(self._history)
            spread = float(np.hypot(*np.ptp(history, axis=0)))
            speed = float(np.hypot(*(history[-1] - history[0]))) / (STILL_FRAMES - 1)
            if spread < STILL_SPREAD * self.t_move() and speed < V_STILL:
                # The window stays where the ball has been over those frames, not on
                # the last of them, which is as far off as the looks wander.
                self.pos = np.median(history, axis=0)
                self.following = False
                self._confirm = 0
                self.vel = np.zeros(2)
                log.info(
                    "frame %d: the ball has settled %.1f px from where it started; the "
                    "window stays there",
                    frame, float(np.hypot(*(self.pos - self.reference_px))),
                )  # fmt: skip
        self.window_px = self.pos.copy()
        return self.origin_px + (self.window_px - self.reference_px)

    def replay(self, n_frames: int) -> ScriptedWatch | None:
        """The window trajectory a second pass over the same frames should follow.

        None when no reference was ever taken (the rim look never worked here), in which
        case the second pass is better off watching for itself.
        """
        if self.reference_px is None:
            return None
        scatter = self.scatter()
        if scatter is None:
            scatter = T_MOVE_PX / T_MOVE_SCATTER
        trajectory, rim, episodes = plan_window_trajectory(
            self.looks, n_frames, self.reference_px, self.radius_px, scatter
        )
        return ScriptedWatch(
            self.origin_px, self.reference_px, trajectory, rim, episodes
        )


def span_estimates(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The latest look, then each `FOLLOW_SPANS` line fit's value at `t = 0`.

    Shape `(1 + len(FOLLOW_SPANS), 2)`, NaN where a span holds fewer than two looks.
    """
    out = np.full((1 + len(FOLLOW_SPANS), 2), np.nan)
    if len(t) == 0:
        return out
    out[0] = xy[int(np.argmax(t))]
    for k, span in enumerate(FOLLOW_SPANS, start=1):
        sel = t > -span
        if int(sel.sum()) < 2:
            continue
        ts, ps = t[sel], xy[sel]
        tm = ts.mean()
        sxx = float(np.sum((ts - tm) ** 2))
        slope = ((ts - tm)[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / sxx
        out[k] = ps.mean(axis=0) - slope * tm
    return out


def causal_estimate(
    t: np.ndarray, xy: np.ndarray, bound: float, softness: float
) -> np.ndarray:
    """The ball's position at `t = 0` from looks at times `t <= 0` (at least one).

    A straight line is fitted to the looks of the last `span` frames for each span in
    `FOLLOW_SPANS`, and the estimate leans on the longest span whose value at `t = 0`
    lies within `bound` of every shorter span's, and of the latest look itself
    (Lepski's rule, as `_adaptive_filter` applies it to a whole recording). A line is
    unbiased while the ball moves steadily, so a long span averages the looks' jitter
    away; through a jerk it is not, and the span shrinks to the last few looks.
    Starting from the latest look rather than from the shortest line keeps a run of
    failed looks from being bridged by extrapolating the last two looks' jitter.

    The cut is soft: each span survives with a logistic weight of how far inside the
    bound it stays (`softness` wide), and the estimate is the spans' values weighted by
    the chance that each is the longest to survive. A hard cut moved the window by up to
    `bound` whenever a span near it flipped in or out, which after a jerk happened from
    one frame to the next and was rotation of its own.
    """
    ests = span_estimates(t, xy)
    ests = ests[np.isfinite(ests[:, 0])]
    survive = np.ones(len(ests))
    for j in range(1, len(ests)):
        margin = bound - float(np.hypot(*(ests[j] - ests[:j]).T).max())
        survive[j] = survive[j - 1] / (1.0 + np.exp(-margin / softness))
    longest = survive - np.r_[survive[1:], 0.0]
    return (longest[:, None] * ests).sum(axis=0) / longest.sum()


def _median_filter(x: np.ndarray, width: int) -> np.ndarray:
    """Running median of `width` frames along the first axis, edges held."""
    from numpy.lib.stride_tricks import sliding_window_view

    if width <= 1:
        return x
    half = width // 2
    pad = np.pad(x, ((half, half), (0, 0)), mode="edge")
    return np.median(sliding_window_view(pad, width, axis=0), axis=-1)


def _gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian of `sigma` frames along the first axis, edges held."""
    if sigma <= 0.0:
        return x
    half = int(np.ceil(4.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    pad = np.pad(x, ((half, half), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(pad[:, k], kernel, mode="valid") for k in range(x.shape[1])], 1
    )


def _adaptive_filter(x: np.ndarray, sigma_max: float, bound: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing whose width adapts to the motion, per frame.

    Each frame gets the widest of the Gaussians in `PLAN_SIGMAS` (up to `sigma_max`)
    whose estimate lies within `bound` of every narrower one's and of `x` itself, or
    within the typical size of that difference over the recording if it is larger
    (Lepski's rule). Where the ball rests or drifts, every width agrees and the widest
    averages the looks' jitter away; where it jerks, the wide ones are pulled off by
    the curvature and the width shrinks, so the window keeps up with the ball instead
    of being smeared over the jerk. The bound is in pixels rather than in units of the
    looks' noise because that noise is not what tells a jerk apart: the looks wander
    slowly and step by a few pixels as legs cross the rim (2-3 px in a frame on
    `ball_drop`), and a rule scaled to their frame-to-frame jitter took every such step
    for a jerk. The chosen width is eased over a few frames, since stepping between two
    estimates would be a jitter of its own.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ests = [x] + [_gaussian_filter(x, s) for s in PLAN_SIGMAS if s <= sigma_max]
    width = np.zeros(len(x))
    alive = np.ones(len(x), dtype=bool)
    for j in range(1, len(ests)):
        for i in range(j):
            d = np.hypot(*(ests[j] - ests[i]).T)
            alive &= d <= max(float(np.median(d)), bound)
        width[alive] = j
    # The narrowest width within a couple of easing lengths, then eased: the width
    # starts shrinking before a jerk rather than at it, and recovers after it.
    reach = int(np.ceil(2.0 * PLAN_WIDTH_EASE))
    pad = np.pad(width, reach, mode="edge")
    width = sliding_window_view(pad, 2 * reach + 1).min(axis=-1)
    width = _gaussian_filter(width[:, None], PLAN_WIDTH_EASE)[:, 0]
    lo = np.floor(width).astype(int)
    hi = np.minimum(lo + 1, len(ests) - 1)
    f = (width - lo)[:, None]
    stack = np.stack(ests)
    idx = np.arange(len(x))
    return (1.0 - f) * stack[lo, idx] + f * stack[hi, idx]


def plan_window_trajectory(
    looks,
    n_frames: int,
    reference_px,
    radius_px: float,
    scatter: float,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """Where the window should have been on every frame, from all the looks at once.

    `looks` are `(frame, position, rim fraction)` as `CenterWatch` records them. They
    are interpolated over the frames without one, cleaned with a `PLAN_MEDIAN`-frame
    median and smoothed with a zero-phase Gaussian of `sigma` frames. The window then
    holds the ball's resting level - `reference_px` to begin with - until the smoothed
    looks leave it by more than `T_MOVE`. An excursion that never reaches `T_MOVE_FAST`
    and lasts less than `CONFIRM_SLOW_FRAMES` is the animal at the rim, not a move, and
    is ignored as the online rule ignores it. A move is followed from the last frame the
    looks sat within the scatter of the level (at most `PLAN_LOOKBACK` frames before the
    departure) until they have stayed within `STILL_SPREAD * T_MOVE` of each other for
    `STILL_FRAMES`, where their median becomes the new level. The window reaches a new
    target - the looks where a move starts, the level where it ends - by closing
    `1 - CATCHUP` of the remaining gap per frame rather than stepping onto it, as it
    does online.
    Returns the per-frame window position, the rim fraction of the look on each frame
    (NaN where there was none) and the moves as `(start, stop, peak px)`.
    """
    sigma = PLAN_SIGMA if sigma is None else sigma
    n = int(n_frames)
    reference = np.asarray(reference_px, dtype=np.float64)
    pos = np.full((n, 2), np.nan)
    rim = np.full(n, np.nan)
    for frame, xy, fraction in looks:
        if 0 <= frame < n:
            pos[frame] = xy
            rim[frame] = fraction
    valid = np.flatnonzero(np.isfinite(pos[:, 0]))
    if valid.size < 2:
        return np.tile(reference, (n, 1)), rim, []
    idx = np.arange(n)
    s = np.stack([np.interp(idx, valid, pos[valid, k]) for k in range(2)], 1)
    t_move = max(T_MOVE_PX, T_MOVE_RADII * radius_px, T_MOVE_SCATTER * scatter)
    s = _adaptive_filter(_median_filter(s, PLAN_MEDIAN), sigma, PLAN_BOUND * t_move)
    t_fast = max(T_MOVE_FAST_PX, T_MOVE_FAST_RADII * radius_px)
    near = max(1.0, scatter)
    # Where the window is meant to be: the level while the ball rests, the looks while
    # it moves. The window itself lags these targets only where they jump.
    target = np.empty_like(s)
    follow = np.zeros(n, dtype=bool)
    level = reference.copy()
    episodes: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        d = np.hypot(*(s[i:] - level).T)
        away = np.flatnonzero(d > t_move)
        if away.size == 0:
            target[i:] = level
            break
        j = i + int(away[0])  # the departure from the level
        back = np.flatnonzero(d[away[0] :] <= t_move)
        end = j + int(back[0]) if back.size else n
        peak = float(np.hypot(*(s[j:end] - level).T).max())
        if peak < t_fast and end - j < CONFIRM_SLOW_FRAMES:
            target[i:end] = level
            i = end
            continue
        first = max(i, j - PLAN_LOOKBACK)
        close = np.flatnonzero(np.hypot(*(s[first:j] - level).T) <= near)
        start = first + int(close[-1]) if close.size else j
        target[i:start] = level
        k = j + 1
        while k + STILL_FRAMES <= n:
            spread = float(np.hypot(*np.ptp(s[k : k + STILL_FRAMES], axis=0)))
            if spread < STILL_SPREAD * t_move:
                break
            k += 1
        stop = n if k + STILL_FRAMES > n else k  # the former: moving as the frames end
        target[start:stop] = s[start:stop]
        follow[start:stop] = True
        episodes.append((start, stop, peak))
        if stop == n:
            break
        level = np.median(s[stop : stop + STILL_FRAMES], axis=0)
        i = stop
    w = np.empty_like(s)
    w[0] = target[0]
    for t in range(1, n):
        # Carry the ball's own motion exactly; only a gap to the target is closed
        # gradually, so the window leaves a level along the ball and settles onto the
        # next one without a step.
        predicted = w[t - 1] + (s[t] - s[t - 1] if follow[t] and follow[t - 1] else 0.0)
        w[t] = predicted + (1.0 - CATCHUP) * (target[t] - predicted)
    return w, rim, episodes


class ScriptedWatch:
    """`CenterWatch`'s interface for a window trajectory planned in advance.

    Built by `CenterWatch.replay` for the second of two passes over a recording: the
    first pass measured the ball on every frame, so the second can put the window where
    the ball was on each frame, from the frame it left its resting place, without the
    confirmation delay and the filter lag an online follower pays to keep a still ball's
    window still.
    """

    def __init__(self, origin_px, reference_px, trajectory, rim_fraction, episodes):
        self.origin_px = np.asarray(origin_px, dtype=np.float64)
        self.reference_px = np.asarray(reference_px, dtype=np.float64)
        self.trajectory = np.asarray(trajectory, dtype=np.float64)
        self._rim = np.asarray(rim_fraction, dtype=np.float64)
        self.episodes = list(episodes)
        self.rim_fraction = 0.0

    def update(self, frame: int, gray) -> np.ndarray | None:
        """Where to center the window on `frame`, as `CenterWatch.update` says it."""
        if frame >= len(self.trajectory):
            return None
        if np.isfinite(self._rim[frame]):
            self.rim_fraction = float(self._rim[frame])
        return self.origin_px + (self.trajectory[frame] - self.reference_px)

    def replay(self, n_frames: int) -> ScriptedWatch:
        """A plan replays as itself."""
        return self
