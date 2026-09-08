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
- The seed is the last accepted look carried by the estimated velocity, never the
  smoothed position. The look tolerates a seed up to about half its band off and pulls
  larger errors part of the way in, so a look whose answer lands far from its seed is
  repeated once from that answer. Seeded on the smoothed position, 004's fall (up to 8
  px between frames) put the seed 12-27 px behind, the rim fraction collapsed and the
  ball was lost for 25 frames.

An alpha-beta filter carries the position and velocity from the first frame on; what
changes is whether the window follows it. While the filtered position stays within
`T_MOVE` of the window, the window does not move at all: the look scatters about a pixel
per axis and excursions of up to 5 px last tens of frames (the animal at the rim), and a
window that chased them would turn each one into rotation. The filter runs at its slow
gains until then - scheduled gains would let a few frames of such an excursion carry it
past the bound, which on the synthetic `lab_small_ball` put 0.26 deg spikes into a run
whose ball never moved. Once the bound is exceeded for `CONFIRM_FRAMES` in a row, the
window follows the filtered position on every frame, bleeding off the gap it started
with over a few frames rather than jumping it, until the position has stayed within half
of `T_MOVE` for `STILL_FRAMES`. A displacement that stays under `T_MOVE_FAST` needs a
long confirmation and a larger one a short one: on the synthetic lab-like scenes the
animal's body pulls the look 3-8 px toward itself for up to 80 frames at a time, and
each such excursion taken for a move cost `lab_small_ball` a tenth of its p95 error and
2 deg/min of drift, while a real drop passes the fast bound within a few frames of
passing `T_MOVE`.

Measured on trial 004, where the ball falls 240 px and comes back: the window starts
following while the ball is still within a few pixels of where it rested, against 50 px
for the cost-triggered follower this replaces; its distance from an independent detector
trajectory over the episode is 5-6 px (median) and 25 px (worst) against 6 and 110, and
the episode's median photometric cost falls from 0.204 to 0.171 (an oracle-placed window
gives 0.158). `detect.detect_ball` on a downscaled buffer, the seed-independent look
that follower relied on, is kept only to recover a rim look that has failed for
`COAST_FRAMES` in a row.

Every position is a displacement from a reference taken over the first accepted looks,
never an absolute position, so a systematic difference between the look and whatever
fitted the config's circle (3.5 px on 004) does not move a still ball's window. The map
is not thrown away when the window moves: it is stored in the window frame at `R = I`,
which is the ball's own body frame, so changing the window only re-expresses the current
orientation. See `Tracker.refit_center`.

The delay that remains online is structural: the confirmation that keeps a still ball's
window still, and the filter's lag on a fast drop. With two passes over a recording
(`Tracker.prime_from`) both go. The first pass records every look it believed,
`plan_window_trajectory` turns them into a window position per frame - interpolated,
median-cleaned, zero-phase smoothed, held at the resting level by the rule above and
following each move from the frame the ball left that level - and `ScriptedWatch`
replays it in the second pass, which therefore measures nothing and runs faster. On
`ball_drop` (exact truth) the window's p95 distance from the ball over the episode goes
from 7.8 px to 2.2, and the episode's per-frame error from 0.056 to 0.044 deg (median)
and from 0.205 to 0.113 (p95), against a second pass that follows the ball for itself;
on trial 004's 240 px fall the optical-flow cross-check's residual along the fall over
the episode goes from 4.31 px rms online to 4.17 for a second pass that follows for
itself and 3.82 for the planned window.
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
# Alpha-beta gains, scheduled on the size of the innovation relative to the look's own
# scatter (the median innovation while still): slow while the looks scatter like noise
# about the prediction, fast while they run away from it. Neither pair does on its own.
# On `ball_drop` (a smooth 70 px excursion on a 175 px ball) the fast pair puts the
# look's noise into the window and costs 0.082 deg per frame over the episode against
# 0.064 for the slow pair; on 004's drop (up to 8 px between frames) the slow pair falls
# 50 px behind.
ALPHA_SLOW = 0.1
BETA_SLOW = 0.005
ALPHA_FAST = 0.3
BETA_FAST = 0.05
ADAPT_LO = (
    3.0  # innovation, in units of the still scatter, where the gains start to rise
)
ADAPT_HI = 8.0  # ... and where they are fully fast
ADAPT_EMA = 0.3  # weight of the newest innovation in the running size
SCATTER_MIN_PX = 0.1
# A look farther than this fraction of the band from the prediction is not believed,
# unless three in a row are; and how many failed looks the filter coasts through on its
# last velocity before the window freezes and the seed-independent detection is asked
# instead.
INNOVATION_GATE = 0.75
COAST_FRAMES = 8
# When the window starts following: the filtered position this far from the window for
# `CONFIRM_FRAMES` frames running. The distance is the larger of an absolute floor, a
# fraction of the radius and a multiple of the median innovation seen while still, so it
# sits above the look's own scatter on any recording. (A speed trigger was tried and
# dropped: it brought the follow on 004's drop forward by two frames and fired on the
# occluder excursions of the still synthetic scenes.)
T_MOVE_PX = 4.0
T_MOVE_RADII = 0.01
T_MOVE_SCATTER = 4.0
# Frames the displacement must persist: `CONFIRM_FRAMES` once it exceeds `T_MOVE_FAST`
# (an absolute distance with a floor as a fraction of the radius), `CONFIRM_SLOW_FRAMES`
# while it only exceeds `T_MOVE`. The fast bound is absolute because the look's
# excursions toward the animal's body are 3-8 px on any ball measured (an 80 px
# synthetic one and the 518 px lab one alike), while a drop worth following is tens of
# pixels.
CONFIRM_FRAMES = 3
CONFIRM_SLOW_FRAMES = 100
T_MOVE_FAST_PX = 10.0
T_MOVE_FAST_RADII = 0.02
# The gap between window and filtered position when following starts is multiplied by
# this every frame instead of being jumped: a 5 px step is 0.55 degrees of rotation on
# 004's ball.
CATCHUP = 0.7
# Following ends when the filtered position has spread less than this fraction of
# `T_MOVE` over `STILL_FRAMES` frames and the speed is below `V_STILL`.
STILL_FRAMES = 30
STILL_SPREAD = 0.5
V_STILL = 0.05
# Planning the window from a whole recording's looks (`plan_window_trajectory`): the
# width in frames of the zero-phase Gaussian the looks are smoothed with, after a median
# over `PLAN_MEDIAN` frames that removes isolated wrong looks; and how far back from the
# frame where the smoothed looks leave the resting level to look for the frame where
# they last sat within the look's scatter of it, so that the window leaves along the
# ball rather than stepping onto it. A window offset that is constant costs no rotation,
# only its change does, so smoothing bias on an accelerating ball matters less than the
# look's noise, and wider is better on both scenes measured: on `ball_drop` (a smooth
# 0.26 px/frame excursion) the episode error is 0.053, 0.047 and 0.044 deg at sigma 1.5,
# 3 and 6 (0.056 following online, 0.039 at 12), and on trial 004's 8 px/frame fall the
# optical-flow cross-check residual over the episode is 3.96, 3.90 and 3.82 px at 1.5, 3
# and 6 (4.17 following online). Six is the widest tried on the fast ball, so it is the
# default.
PLAN_SIGMA = 6.0
PLAN_MEDIAN = 5
PLAN_LOOKBACK = 60
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

    A rim look per frame feeds an alpha-beta filter on the ball's displacement from
    where it started. The window stays put while the filtered displacement is within
    `T_MOVE` of it; once it is not, the window follows the filter every frame until the
    ball has been still for `STILL_FRAMES`. See the module docstring.
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
        self.pos: np.ndarray | None = None  # filtered ball position
        self.vel = np.zeros(2)
        self.window_px: np.ndarray | None = None  # where the window is
        self.following = False
        self.rim_fraction = 0.0
        # Every look the filter believed, as `(frame, position, rim fraction)`, for a
        # second pass over the same frames to plan the window from (see `replay`).
        self.looks: list[tuple[int, np.ndarray, float]] = []
        self._last_look: tuple[int, np.ndarray] | None = None
        self._failed = 0  # looks that failed in a row
        self._unbelieved = 0  # looks that landed outside the innovation gate in a row
        self._residuals: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._innovations: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._confirm = 0
        self._innovation_size = 0.0
        self._catchup = np.zeros(2)
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
            return None
        if self.polarity is None:
            self.polarity = found.polarity
        self.rim_fraction = found.rim_fraction
        return np.array([found.cx, found.cy])

    def _measure(self, frame: int, gray) -> np.ndarray | None:
        """One measurement of the ball, seeded on the last look plus the velocity."""
        if self._last_look is None:
            seed = self.pos
        else:
            last_frame, last = self._last_look
            seed = last + self.vel * (frame - last_frame)
        band = self.band if self._failed == 0 else self.band_wide
        seen = self._look(gray, seed, band)
        if seen is not None and np.hypot(*(seen - seed)) > RELOOK_BAND_FRACTION * band:
            again = self._look(gray, seen, band)
            if again is not None:
                seen = again
        if seen is None:
            self._failed += 1
        else:
            self._failed = 0
            self._last_look = (frame, seen)
        return seen

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

    def _gains(self, innovation: float) -> tuple[float, float]:
        scatter = self.scatter()
        if scatter is None or not self.following:
            return ALPHA_SLOW, BETA_SLOW
        size = innovation / scatter
        self._innovation_size += ADAPT_EMA * (size - self._innovation_size)
        s = min(
            max((self._innovation_size - ADAPT_LO) / (ADAPT_HI - ADAPT_LO), 0.0), 1.0
        )
        return ALPHA_SLOW + (ALPHA_FAST - ALPHA_SLOW) * s, BETA_SLOW + (
            BETA_FAST - BETA_SLOW
        ) * s

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
        self.pos = self.pos + self.vel
        seen = self._measure(frame, gray)
        if seen is not None:
            innovation = seen - self.pos
            size = float(np.hypot(*innovation))
            if size <= INNOVATION_GATE * self.band or self._unbelieved >= 3:
                self._unbelieved = 0
                self.looks.append((frame, seen, self.rim_fraction))
                if not self.following:
                    self._innovations.append(size)
                alpha, beta = self._gains(size)
                self.pos = self.pos + alpha * innovation
                self.vel = self.vel + beta * innovation
            else:
                self._unbelieved += 1
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
                self._failed = 0
                log.info(
                    "frame %d: the rim look lost the ball; a detection puts it %.1f px "
                    "from where it started",
                    frame, float(np.hypot(*(self.pos - self.reference_px))),
                )  # fmt: skip

    # ----- per frame -----
    def update(self, frame: int, gray) -> np.ndarray | None:
        """Feed one frame; returns where the window should be centered, or None."""
        if self.reference_px is None:
            self._take_reference(frame, gray)
            return None
        if self.following and frame % COARSE_STRIDE == 0:
            small, self.scale = downscale(gray, self.scale)
            self.coarse.append(small)
        self._filter(frame, gray)
        self._history.append(self.pos.copy())
        gap = self.pos - self.window_px
        if not self.following:
            size = float(np.hypot(*gap))
            bound = self.t_move()
            self._confirm = self._confirm + 1 if size > bound else 0
            fast = max(T_MOVE_FAST_PX, T_MOVE_FAST_RADII * self.radius_px)
            needed = CONFIRM_FRAMES if size > fast else CONFIRM_SLOW_FRAMES
            if self._confirm < needed:
                return None
            self.following = True
            self._catchup = -gap
            self._innovation_size = 0.0
            self.coarse.clear()
            log.info(
                "frame %d: the ball is %.1f px from where the window looks, moving at "
                "%.2f px per frame; following it",
                frame, float(np.hypot(*gap)), float(np.hypot(*self.vel)),
            )  # fmt: skip
        self._catchup = self._catchup * CATCHUP
        target = self.pos + self._catchup
        if len(self._history) == STILL_FRAMES:
            spread = float(np.hypot(*np.ptp(np.stack(self._history), axis=0)))
            if spread < STILL_SPREAD * self.t_move() and np.hypot(*self.vel) < V_STILL:
                self.following = False
                self._confirm = 0
                self.vel = np.zeros(2)
                log.info(
                    "frame %d: the ball has settled %.1f px from where it started; the "
                    "window stays there",
                    frame, float(np.hypot(*(self.window_px - self.reference_px))),
                )  # fmt: skip
        self.window_px = target
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


def plan_window_trajectory(
    looks,
    n_frames: int,
    reference_px,
    radius_px: float,
    scatter: float,
    sigma: float = PLAN_SIGMA,
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
    s = _gaussian_filter(_median_filter(s, PLAN_MEDIAN), sigma)
    t_move = max(T_MOVE_PX, T_MOVE_RADII * radius_px, T_MOVE_SCATTER * scatter)
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
