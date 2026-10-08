"""Follow a ball that moves in its holder, so its movement is not read as rotation.

A window left `d` pixels behind the ball reads the ball's own movement as a rotation of
about `d / r`, and the photometric cost cannot tell, so `CenterWatch` measures the
silhouette on every frame (`detect.RimLook`) and says where the window should be. The
look's radius is measured on the first frames rather than taken from the config: a look
held even half a percent too large climbs whatever stands past the rim. While the ball
rests its looks feed a slow filter and the window stays put, because the animal at the
rim pulls the look a few pixels for tens of frames; a move is followed once several
looks in a row lie well away from the window, and then the window sits on a causal
estimate that averages the looks while the ball moves steadily and shrinks to the last
few through a jerk. Positions are displacements from a reference taken over the first
looks, so the config's circle and the look need not agree.

A second pass over the same recording replays a trajectory planned from all of the first
pass's looks at once (`plan_window_trajectory`, `ScriptedWatch`), without the online
confirmation delay.
"""

from __future__ import annotations

import logging
from array import array
from collections import deque

import cv2
import numpy as np

from spintrack.detect import DetectionError, RimLook, detect_ball

log = logging.getLogger("spintrack")

# Half-width of the rim band (the look's capture range) and of the wider band used once
# a look has failed, as fractions of the radius with floors for small balls.
BAND_RADII = 0.05
BAND_MIN_PX = 4.0
BAND_WIDE_RADII = 0.10
# The rim fit's fixed outlier cut; a look landing farther than this share of the band
# from its seed is repeated from where it landed.
CUT_RADII = 0.01
CUT_MIN_PX = 2.0
RELOOK_BAND_FRACTION = 0.25
# What a look must show to be believed: rim rays found, directions covered, and a
# residual under a multiple of the running median of the run's own looks, since blur,
# compression and ball size set the residual.
MIN_RIM_FRACTION = 0.3
MIN_ARC_FRACTION = 0.25
# A ball that has moved can show less of its rim (the frame's edge, occluders fixed in
# the image), so a followed ball's look needs less of it.
MIN_RIM_FRACTION_FOLLOWING = 0.2
RESIDUAL_FACTOR = 1.2
RESIDUAL_HISTORY = 200
RESIDUAL_MIN_LOOKS = 20
# Frames of free-radius looks whose median radius becomes the look's, then the looks
# that set the reference. A silhouette radius this far from the config's is a warning,
# unless too little of the rim is in view to tell (an ellipse's arc off the axis).
RADIUS_FRAMES = 30
REFERENCE_LOOKS = 60
RADIUS_WARN = 0.03
RADIUS_MIN_ARC = 0.4
# Gain of the resting ball's filter: slow, so the animal pulling the look for a few
# frames cannot carry it past `T_MOVE`.
ALPHA = 0.1
SCATTER_MIN_PX = 0.1
# A look farther than this share of the band from where it was expected is not
# believed unless three in a row are. After `COAST_FRAMES` failed looks the window
# freezes and the seed-free detection is asked.
INNOVATION_GATE = 0.75
COAST_FRAMES = 8
# A resting ball counts as displaced beyond the largest of a floor, a share of the
# radius and a multiple of its looks' scatter.
T_MOVE_PX = 4.0
T_MOVE_RADII = 0.01
T_MOVE_SCATTER = 4.0
# A move is followed once `CONFIRM_FRAMES` looks lie beyond `T_MOVE_FAST`, which is
# above the animal's pull on the look (3-8 px on any ball). A slow creep past `T_MOVE`
# for `CONFIRM_SLOW_FRAMES` only nudges the window `CREEP_STEP` of the way, since moving
# all the way would also follow the looks' wander.
CONFIRM_FRAMES = 3
CONFIRM_SLOW_FRAMES = 100
CREEP_STEP = 0.3
T_MOVE_FAST_PX = 10.0
T_MOVE_FAST_RADII = 0.02
# While following, line fits over these spans of recent looks, each faded out as it
# nears `FOLLOW_BOUND * t_move` of a shorter one (`causal_estimate`). The next look is
# seeded on an exponential mean of the looks' steps.
FOLLOW_SPANS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
FOLLOW_BOUND = 1.25
FOLLOW_SOFT = 0.2
LOOK_VEL_EMA = 0.5
# Following ends once the window has spread less than `STILL_SPREAD * t_move` and moved
# less than `V_STILL` px a frame over `STILL_FRAMES`. Ending early is the costlier
# mistake: a ball that pauses and moves on is not followed again until it is far off.
STILL_FRAMES = 60
STILL_SPREAD = 0.5
V_STILL = 0.05
# A window this many radii from where it started has left the ball (a ball moves a
# fraction of its radius in its holder; a look climbing the animal runs on): stop.
MAX_SHIFT_RADII = 1.0
# Planning from a whole recording: a median over `PLAN_MEDIAN` frames removes isolated
# wrong looks; then the widest of `PLAN_SIGMAS` that stays within `PLAN_BOUND * t_move`
# of the narrower ones, eased over `PLAN_WIDTH_EASE` frames; a move is traced back up to
# `PLAN_LOOKBACK` frames to where it left the resting level; and `CATCHUP` of the gap to
# a new target is left open each frame, so the window does not step.
PLAN_SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
PLAN_BOUND = 0.5
PLAN_WIDTH_EASE = 2.0
PLAN_MEDIAN = 5
PLAN_LOOKBACK = 60
CATCHUP = 0.7
# The seed-free look: a detection on `COARSE_BUFFER` frames, every `COARSE_STRIDE`-th
# (which erases the rotating texture), downscaled while the radius stays above
# `COARSE_MIN_RADIUS`. It describes the middle of its buffer, `COARSE_LAG` frames ago.
COARSE_BUFFER = 12
COARSE_STRIDE = 2
COARSE_MIN_RADIUS = 120.0
COARSE_MAX_SCALE = 4
COARSE_LAG = (COARSE_BUFFER - 1) * COARSE_STRIDE / 2.0
RECOVER_EVERY = 4  # frames between detections while the rim look keeps failing
RECOVER_RADIUS_TOL = 0.02  # detections the animal fools are off in radius too


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


class CenterWatch:
    """Follow a ball that moves in its holder, and say where the window should be.

    `update` returns the window center for a frame (source pixels), or None to leave the
    window where it is. `episodes` lists the window's moves as `[start, end, peak px]`.
    """

    def __init__(self, origin_px, radius_px: float):
        self.origin_px = np.asarray(origin_px, dtype=np.float64)
        self.radius_config = float(radius_px)
        self.radius_measured: float | None = None
        self.radius_arc = 0.0  # share of directions the radius looks saw rim in
        self._set_radius(radius_px)
        self.polarity: float | None = None
        self._radius_looks: list[np.ndarray] = []
        self._radius_arcs: list[float] = []
        self._reference_looks: list[np.ndarray] = []
        self._reference_seed: np.ndarray | None = None
        # Positions are in the look's own coordinates; `update` returns them as offsets
        # from `origin_px`, so the config's circle and the look need not agree.
        self.reference_px: np.ndarray | None = None
        self.pos: np.ndarray | None = None  # resting: filtered; following: estimated
        self.vel = np.zeros(2)  # the looks' velocity, while following
        self.window_px: np.ndarray | None = None
        self.following = False
        self.stopped_at: int | None = None
        self.rim_fraction = 0.0
        self.episodes: list[list] = []
        # Every believed look as flat `frame, x, y, rim fraction`, for a second pass.
        self._looks = array("d")
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
        self._unbelieved = 0  # looks outside the innovation gate in a row
        self._residuals: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._innovations: deque[float] = deque(maxlen=RESIDUAL_HISTORY)
        self._confirm = 0
        self._history: deque[np.ndarray] = deque(maxlen=STILL_FRAMES)
        self.coarse: deque[np.ndarray] = deque(maxlen=COARSE_BUFFER)
        self._next_recovery = 0
        self._previous_seen: tuple[int, np.ndarray] | None = None

    def _set_radius(self, radius: float) -> None:
        self.radius_px = float(radius)
        self.band = max(BAND_MIN_PX, BAND_RADII * self.radius_px)
        self.band_wide = max(2.0 * BAND_MIN_PX, BAND_WIDE_RADII * self.radius_px)
        self.cut = max(CUT_MIN_PX, CUT_RADII * self.radius_px)
        self.scale = coarse_scale(self.radius_px)
        self._narrow = RimLook(self.radius_px, self.band, self.cut)
        self._wide = RimLook(self.radius_px, self.band_wide, self.cut)

    @property
    def looks(self) -> np.ndarray:
        """Believed looks as rows of `frame, x, y, rim fraction`."""
        return np.frombuffer(self._looks, dtype=np.float64).reshape(-1, 4).copy()

    def _record(self, frame: int, xy, rim: float | None = None) -> None:
        rim = self.rim_fraction if rim is None else rim
        self._looks.extend((frame, xy[0], xy[1], rim))

    # ----- measurements -----
    def _look(self, gray, seed, wide: bool = False) -> np.ndarray | None:
        """The rim look about `seed`, or None when it is not to be believed."""
        try:
            found = (self._wide if wide else self._narrow)(
                gray,
                float(seed[0]),
                float(seed[1]),
                self.polarity,
                min_rim_fraction=(
                    MIN_RIM_FRACTION_FOLLOWING if self.following else MIN_RIM_FRACTION
                ),
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
        """One look at the ball, from the best seed that finds it.

        A resting ball is looked for where the filter puts it, which a look that has
        started to climb the animal cannot drag along. A moving ball is looked for where
        its last look and velocity put it. The fallbacks are the last look itself (a
        ball that has just stopped) and, at rest, where the last few looks were heading
        (a ball that has just started to fall).
        """
        self._doubtful = None
        last_frame, last = self._last_look
        dt = frame - last_frame
        seeds = [last + self.vel * dt, last]
        if not self.following:
            seeds.insert(0, self.pos)
            if len(self._recent) > 1:
                (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
                if f1 == last_frame:
                    seeds.append(last + (p1 - p0) / (f1 - f0) * dt)
        wide = self._failed > 0
        reach = RELOOK_BAND_FRACTION * (self.band_wide if wide else self.band)
        seed = seeds[0]
        seen = self._look(gray, seed, wide)
        for other in seeds[1:]:
            if seen is not None:
                break
            if np.hypot(*(other - seed)) > reach:
                seed = other
                seen = self._look(gray, seed, wide)
        # The look pulls a far seed only part of the way in, so a far answer is checked.
        if seen is not None and np.hypot(*(seen - seed)) > reach:
            again = self._look(gray, seen, wide)
            if again is not None:
                seen = again
        self._seed = seed
        if seen is None and not self.following and self._doubtful is None:
            self._probe(frame, gray, wide)
        if seen is None:
            self._failed += 1
            return None
        self._failed = 0
        if self.following:
            step = (seen - last) / dt if dt <= COAST_FRAMES else 0.0
            self.vel = self.vel + LOOK_VEL_EMA * (step - self.vel)
        self._last_look = (frame, seen)
        return seen

    def _probe(self, frame: int, gray, wide: bool) -> None:
        """Look where the evidence of a move is heading, when it has run ahead.

        The looks that count towards a move include ones refused for their residual,
        which seed nothing; on a small ball falling fast the believed looks are then
        left behind on a ball that has dropped out of their band. What this look finds
        counts towards the move but seeds nothing either.
        """
        if len(self._recent) < 2:
            return
        (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
        if f1 == self._last_look[0]:
            return
        seen = self._look(gray, p1 + (p1 - p0) / (f1 - f0) * (frame - f1), wide)
        if seen is not None:
            self._doubtful = (seen, self.rim_fraction)

    def _detect(self, frame: int):
        """Find the ball wherever it is, from the downscaled buffer; source pixels.

        Returns `(position, velocity)`, the velocity being what the previous such look
        implies: the position describes `COARSE_LAG` frames ago.
        """
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

    # ----- the resting ball -----
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

    def _measure_radius(self, gray) -> None:
        """Free-radius looks on the first frames; their median radius is the look's.

        Each look is repeated from its own answer, so a config circle a few percent off
        converges before it counts.
        """
        looks = self._radius_looks
        if looks:
            seed = np.median(looks, axis=0)
        else:
            seed = np.r_[self.origin_px, self.radius_px]
        for _ in range(3):
            try:
                found = RimLook(seed[2], self.band, self.cut)(
                    gray, seed[0], seed[1], self.polarity, free_radius=True
                )
            except DetectionError as exc:
                log.debug("radius look refused: %s", exc)
                return
            self.polarity = found.polarity
            seed = np.array([found.cx, found.cy, found.r])
        if not found.ok:
            return
        looks.append(seed)
        self._radius_arcs.append(found.arc_fraction)
        if len(looks) < RADIUS_FRAMES:
            return
        cx, cy, radius = np.median(looks, axis=0)
        self.radius_measured = float(radius)
        self.radius_arc = float(np.median(self._radius_arcs))
        self._set_radius(radius)
        self._reference_seed = np.array([cx, cy])
        change = radius / self.radius_config - 1.0
        warn = abs(change) > RADIUS_WARN and self.radius_arc >= RADIUS_MIN_ARC
        log.log(
            logging.WARNING if warn else logging.DEBUG,
            "ball silhouette radius %.1f px, %+.1f%% from the config's %.1f px",
            radius, 100 * change, self.radius_config,
        )  # fmt: skip

    def _take_reference(self, frame: int, gray) -> None:
        """Median of the first `REFERENCE_LOOKS` accepted looks; nothing moves yet."""
        if self.radius_measured is None:
            self._measure_radius(gray)
            return
        looks = self._reference_looks
        seed = self._reference_seed if not looks else np.median(looks, axis=0)
        seen = self._look(gray, seed)
        if seen is None:
            return
        looks.append(seen)
        self._record(frame, seen)
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
        """Measure, then update the resting filter or the track being followed."""
        seen = self._measure(frame, gray)
        if seen is None and self._doubtful is not None and not self.following:
            # A look refused only for its residual seeds nothing, so a look that climbs
            # the animal cannot build on it, but it counts towards a move: a ball that
            # has dropped can show less of its rim, and its residual rises with it.
            xy, rim = self._doubtful
            self._recent.append((frame, xy, rim))
        if seen is not None:
            self._recent.append((frame, seen, self.rim_fraction))
            # A resting ball's look is judged against the filter, which the animal
            # cannot drag; a moving ball's against its own seed.
            expected = self._seed if self.following else self.pos
            off = float(np.hypot(*(seen - expected)))
            if off > INNOVATION_GATE * self.band and self._unbelieved < 3:
                self._unbelieved += 1
                return
            self._unbelieved = 0
            self._record(frame, seen)
            self._track.append((frame, seen))
            if not self.following:
                innovation = seen - self.pos
                self._innovations.append(float(np.hypot(*innovation)))
                self.pos = self.pos + ALPHA * innovation
                self.vel = np.zeros(2)
            return
        if self._failed <= COAST_FRAMES:
            return
        # The rim look has lost the ball. Coasting further would drift away from a ball
        # that has stopped, so the window freezes and the seed-free look is asked.
        self.vel = np.zeros(2)
        if frame < self._next_recovery:
            return
        self._next_recovery = frame + RECOVER_EVERY
        found = self._detect(frame)
        if found is None:
            return
        seen, velocity = found
        self.pos = seen + velocity * COARSE_LAG
        self.vel = velocity
        self._last_look = (frame, self.pos.copy())
        self._track.clear()
        self._failed = 0
        log.info(
            "frame %d: the rim look lost the ball; a detection puts it %.1f px from "
            "where it started",
            frame, float(np.hypot(*(self.pos - self.reference_px))),
        )  # fmt: skip

    def _moved_away(self, frame: int, bound: float) -> bool:
        """Whether the last `CONFIRM_FRAMES` looks, all recent, lie beyond `bound`.

        Those looks are the start of the move, and the resting filter's gate will have
        refused some of them, so they become the track the window follows. A line
        through the resting ball's looks and the moving one's describes neither.
        """
        recent = list(self._recent)
        if len(recent) < CONFIRM_FRAMES or recent[-1][0] != frame:
            return False
        if frame - recent[0][0] >= 2 * CONFIRM_FRAMES:
            return False
        if any(np.hypot(*(xy - self.window_px)) <= bound for _, xy, _ in recent):
            return False
        believed = set(self._looks[-4 * 2 * CONFIRM_FRAMES :: 4])
        for f, xy, rim in recent:
            if f not in believed:
                self._record(f, xy, rim)
        self._track = deque(((f, xy) for f, xy, _ in recent), FOLLOW_SPANS[-1])
        return True

    def _start_following(self, frame: int) -> None:
        self.following = True
        # The looks that showed the move seed the next look, carried by their velocity.
        (f0, p0, _), (f1, p1, _) = self._recent[0], self._recent[-1]
        self.vel = (p1 - p0) / (f1 - f0)
        self._last_look = (f1, p1)
        # The rim the band sees at the new place is not the one it saw at rest.
        self._residuals.clear()
        self._history.clear()
        self.coarse.clear()
        self.episodes.append([frame, frame, 0.0])
        log.info(
            "frame %d: the ball is %.1f px from where the window looks; following it",
            frame, float(np.hypot(*(p1 - self.window_px))),
        )  # fmt: skip

    def _estimate(self, frame: int) -> np.ndarray:
        """Where the ball is on `frame`; where the window is when the looks ran out."""
        if not self._track or frame - self._track[-1][0] > COAST_FRAMES:
            return self.pos.copy()
        t = np.array([f - frame for f, _ in self._track], dtype=np.float64)
        xy = np.stack([p for _, p in self._track])
        bound = FOLLOW_BOUND * self.t_move()
        return causal_estimate(t, xy, bound, FOLLOW_SOFT * bound)

    def _target(self, frame: int) -> np.ndarray:
        """The window center for `frame`, in source pixels; stops a runaway follow."""
        shift = float(np.hypot(*(self.window_px - self.reference_px)))
        episode = self.episodes[-1]
        episode[1] = frame
        if shift > MAX_SHIFT_RADII * self.radius_px:
            log.warning(
                "frame %d: the window ran %.0f px, a whole ball radius, from where "
                "it started; the ball follower stops and the window goes back",
                frame, shift,
            )  # fmt: skip
            self.stopped_at = frame
            self.following = False
            self.window_px = self.reference_px.copy()
            return self.origin_px.copy()
        episode[2] = max(episode[2], shift)
        return self.origin_px + (self.window_px - self.reference_px)

    # ----- per frame -----
    def update(self, frame: int, gray) -> np.ndarray | None:
        """Feed one frame; returns where the window should be centered, or None."""
        if self.stopped_at is not None:
            return None
        if self.reference_px is None:
            self._take_reference(frame, gray)
            return None
        # The seed-free look's buffer is kept while following and while a resting
        # ball's look is failing; a resting ball seen again empties it.
        if self.following or self._failed > 0:
            if frame % COARSE_STRIDE == 0:
                small, self.scale = downscale(gray, self.scale)
                self.coarse.append(small)
        elif self.coarse:
            self.coarse.clear()
        self._filter(frame, gray)
        if not self.following:
            far = float(np.hypot(*(self.pos - self.window_px))) > self.t_move()
            self._confirm = self._confirm + 1 if far else 0
            if self._confirm >= CONFIRM_SLOW_FRAMES:
                self._confirm = 0
                self.window_px = self.window_px + CREEP_STEP * (
                    self.pos - self.window_px
                )
                log.info(
                    "frame %d: the ball has crept %.1f px from where it started; the "
                    "window moves towards it",
                    frame, float(np.hypot(*(self.pos - self.reference_px))),
                )  # fmt: skip
                self.episodes.append([frame, frame, 0.0])
                return self._target(frame)
            fast = max(T_MOVE_FAST_PX, T_MOVE_FAST_RADII * self.radius_px)
            if not self._moved_away(frame, fast):
                return None
            self._start_following(frame)
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
        return self._target(frame)

    def replay(self, n_frames: int) -> ScriptedWatch | None:
        """The window trajectory a second pass over the same frames should follow.

        None when there is nothing to plan from (the rim look never locked on) or the
        follower stopped, in which case the second pass is better off watching itself.
        """
        if self.reference_px is None or self.stopped_at is not None:
            return None
        scatter = self.scatter() or T_MOVE_PX / T_MOVE_SCATTER
        trajectory, rim, episodes = plan_window_trajectory(
            self.looks, n_frames, self.reference_px, self.radius_px, scatter
        )
        return ScriptedWatch(self, trajectory, rim, episodes)


def span_estimates(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The latest look, then each `FOLLOW_SPANS` line fit's value at `t = 0`.

    `t` is ascending and at most 0. Shape `(1 + len(FOLLOW_SPANS), 2)`, NaN where a span
    holds fewer than two looks. Sums from the newest look back give every span at once.
    """
    spans = np.asarray(FOLLOW_SPANS)
    out = np.full((1 + len(spans), 2), np.nan)
    if len(t) == 0:
        return out
    out[0] = xy[-1]
    tr, xr = t[::-1], xy[::-1]
    n = np.arange(1, len(t) + 1)
    st, stt = np.cumsum(tr), np.cumsum(tr * tr)
    sx, stx = np.cumsum(xr, axis=0), np.cumsum(tr[:, None] * xr, axis=0)
    count = np.searchsorted(-tr, spans)  # looks with t > -span
    has = count >= 2
    i = count[has] - 1
    tm = st[i] / n[i]
    slope = (stx[i] - tm[:, None] * sx[i]) / (stt[i] - n[i] * tm**2)[:, None]
    out[1:][has] = sx[i] / n[i][:, None] - slope * tm[:, None]
    return out


def causal_estimate(
    t: np.ndarray, xy: np.ndarray, bound: float, softness: float
) -> np.ndarray:
    """The ball's position at `t = 0` from looks at times `t <= 0` (at least one).

    The estimate leans on the longest line fit whose value lies within `bound` of every
    shorter span's and of the latest look (Lepski's rule): a line is unbiased while the
    ball moves steadily, so a long span averages the looks' jitter away, and through a
    jerk the span shrinks to the last few looks. The cut is soft - each span survives
    with a logistic weight of its margin, `softness` wide - because a hard one moved the
    window by up to `bound` whenever a span flipped in or out.
    """
    ests = span_estimates(t, xy)
    ests = ests[np.isfinite(ests[:, 0])]
    gaps = np.hypot(*(ests[:, None, :] - ests[None, :, :]).transpose(2, 0, 1))
    worst = np.tril(gaps, -1).max(axis=1)[1:]  # each span against every shorter one
    keep = 1.0 / (1.0 + np.exp(-(bound - worst) / softness))
    survive = np.r_[1.0, np.cumprod(keep)]
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


def _adaptive_filter(x: np.ndarray, bound: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing whose width adapts to the motion, per frame.

    Each frame gets the widest of `PLAN_SIGMAS` whose estimate lies within `bound` of
    every narrower one's and of `x` itself, or within the typical size of that
    difference over the recording if larger (Lepski's rule). The bound is in pixels: the
    legs crossing the rim step the looks by a few pixels, and a bound scaled to the
    looks' jitter took every such step for a jerk. The chosen width is eased over a few
    frames, since stepping between two estimates would be a jitter of its own.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ests = [x] + [_gaussian_filter(x, s) for s in PLAN_SIGMAS]
    width = np.zeros(len(x))
    alive = np.ones(len(x), dtype=bool)
    for j in range(1, len(ests)):
        for i in range(j):
            d = np.hypot(*(ests[j] - ests[i]).T)
            alive &= d <= max(float(np.median(d)), bound)
        width[alive] = j
    # The narrowest width within a couple of easing lengths, then eased: the width
    # starts shrinking before a jerk rather than at it.
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
    looks, n_frames: int, reference_px, radius_px: float, scatter: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """Where the window should have been on every frame, from all the looks at once.

    `looks` are rows of `frame, x, y, rim fraction` (`CenterWatch.looks`). They are
    interpolated over the frames without one, median-cleaned and smoothed by
    `_adaptive_filter`. The window holds the ball's resting level - `reference_px` to
    begin with - until the smoothed looks leave it by more than `T_MOVE`; an excursion
    that never reaches `T_MOVE_FAST` and is shorter than `CONFIRM_SLOW_FRAMES` is the
    animal at the rim and is ignored, as online. A move is followed from the last frame
    the looks sat within the scatter of the level until they have stayed within
    `STILL_SPREAD * T_MOVE` for `STILL_FRAMES`, where their median becomes the new
    level. Returns the per-frame window position, the rim fraction of the look on each
    frame (NaN where there was none) and the moves as `(start, stop, peak px)`.
    """
    n = int(n_frames)
    reference = np.asarray(reference_px, dtype=np.float64)
    looks = np.asarray(looks, dtype=np.float64).reshape(-1, 4)
    frames = looks[:, 0].astype(int)
    inside = (frames >= 0) & (frames < n)
    pos = np.full((n, 2), np.nan)
    rim = np.full(n, np.nan)
    pos[frames[inside]] = looks[inside, 1:3]
    rim[frames[inside]] = looks[inside, 3]
    valid = np.flatnonzero(np.isfinite(pos[:, 0]))
    if valid.size < 2:
        return np.tile(reference, (n, 1)), rim, []
    idx = np.arange(n)
    s = np.stack([np.interp(idx, valid, pos[valid, k]) for k in range(2)], 1)
    t_move = max(T_MOVE_PX, T_MOVE_RADII * radius_px, T_MOVE_SCATTER * scatter)
    s = _adaptive_filter(_median_filter(s, PLAN_MEDIAN), PLAN_BOUND * t_move)
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

    Built by `CenterWatch.replay` for the second of two passes over a recording, which
    puts the window where the first pass saw the ball, from the frame it left its
    resting place.
    """

    def __init__(self, watch: CenterWatch, trajectory, rim_fraction, episodes):
        self.origin_px = watch.origin_px
        self.reference_px = watch.reference_px
        self.radius_config = watch.radius_config
        self.radius_measured = watch.radius_measured
        self.radius_arc = watch.radius_arc
        self.stopped_at = None
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


def watch_checks(watch) -> dict[str, str]:
    """The run summary's lines on where the ball sat and how big it looked."""
    if watch.reference_px is None:
        return {"ball moved": "not measured (the rim look never locked on)"}
    if watch.episodes:
        moves = ", ".join(
            f"frames {int(a)}-{int(b)} ({c:.0f} px)" for a, b, c in watch.episodes[:3]
        )
        if len(watch.episodes) > 3:
            moves += f" and {len(watch.episodes) - 3} more"
    else:
        moves = "no"
    if watch.stopped_at is not None:
        moves += (
            f"; the follower stopped at frame {watch.stopped_at}: the window ran "
            f"{MAX_SHIFT_RADII:g} ball radius from where it started"
        )
    out = {"ball moved": moves}
    if watch.radius_measured is not None:
        change = watch.radius_measured / watch.radius_config - 1.0
        out["radius"] = (
            f"silhouette {watch.radius_measured:.1f} px, config "
            f"{watch.radius_config:.1f} px ({100 * change:+.1f}%)"
        )
        if watch.radius_arc < RADIUS_MIN_ARC:
            out["radius"] += ", too little rim in view to compare"
        elif abs(change) > RADIUS_WARN:
            out["radius"] += ": check the config's ball, the rotation scale is off"
    return out


def watch_record(watch) -> dict:
    """The follower's results for the sidecar."""
    return {
        "reference_px": None
        if watch.reference_px is None
        else [float(v) for v in watch.reference_px],
        "radius_config_px": watch.radius_config,
        "radius_measured_px": watch.radius_measured,
        "radius_arc": watch.radius_arc,
        "episodes": [[int(a), int(b), float(c)] for a, b, c in watch.episodes],
        "stopped_at": watch.stopped_at,
    }
