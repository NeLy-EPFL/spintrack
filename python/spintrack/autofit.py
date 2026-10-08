"""Fill in the geometry a config leaves open, from the recording itself.

`complete_config` runs `prepare_config` when the config lacks the ball, the field of
view or the camera position, and refuses when nothing says where the camera sits around
the animal: a video cannot tell the animal's front from its back. `prepare_config`
detects the ball when the config does not describe one (and compares the two when it
does), places the camera at `camera.azimuth_deg` from where the animal stands when
nothing else does, and fits the field of view when nothing gives it. It needs a source
it can read before tracking begins, so a live camera is refused.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _replace

import cv2
import numpy as np

from spintrack import segment
from spintrack.camera import source_camera
from spintrack.config import BallConfig, Config
from spintrack.detect import (
    BallDetection,
    DetectionError,
    ball_from_masks,
    circle_points,
    detect_ball,
    fit_circle,
    sample_frames,
    temporal_stats,
)
from spintrack.sphere import fit_ball

log = logging.getLogger("spintrack")

CAMERA_SOURCE_MESSAGE = (
    "a live camera needs a config with the ball, the field of view and the camera "
    "position: they are found in frames read before tracking starts. Record a short "
    "clip, track it (spintrack run CLIP) or fix it (spintrack gui CLIP), and pass the "
    "config.toml it writes."
)
NO_POSITION = (
    "where the camera sits around the animal is unknown, and a video cannot tell the "
    "animal's front from its back. Say it: camera.azimuth_deg=180 for a camera behind "
    "the animal, 0 in front, 90 at its right, -90 at its left (or "
    "camera.position_deg=ELEV,AZIM,TWIST, or set it in spintrack gui)"
)
MIN_ANIMAL_SCORE = 0.3  # a weaker animal mask places no camera
# The most of the ball's disk an animal covers: a fly 3-4%, a cockroach on a 10 cm ball
# about 20%; a mask over more is the ball.
MAX_ANIMAL_COVER = 0.5
# Where the animal's silhouette sits, from the ball's center in ball radii: beyond
# `ON_RIM` it stands on the outline, where the elevation does not show (its cosine is
# flat there, and the animal's height above the ball outweighs it: on the lab's octacam
# the silhouette sits 1.08-1.12 radii out from a camera 10.5-12 deg above the animal,
# by deeperfly's leg tips, as it would from a level one); within `OVERHEAD` (a camera
# over 75 deg up) its direction is too short to give the twist, as the body's offset
# from where it stands is a sizable part of it; beyond `OFF_BALL` it is not on the
# ball. On the rim the silhouette's direction gives the twist to within 0.5 deg.
ON_RIM = 0.95
OVERHEAD = 0.25
OFF_BALL = 2.0
# The animal is the mask found within `SAME_PLACE` ball radii in at least `MIN_VOTES`
# of `ANIMAL_FRAMES` frames.
ANIMAL_FRAMES = 3
MIN_VOTES = 2
SAME_PLACE = 0.15
# Below this contrast (gray levels) the ball's surface shows too little texture to
# track at the window's scale: the lab's patterned balls show 10-16, plain white
# polystyrene (a cockroach rig) 2.
MIN_TEXTURE = 4.0
# A detected radius this far from the config's is worth saying out loud: a relative
# radius error costs about twice as much of every in-plane rotation reported.
DISAGREEMENT_WARN = 0.03

VFOV_GRID = (1.0, 120.0, 9)  # log-spaced candidates, degrees
VFOV_REFINE = 6  # candidates between the best grid point's neighbors
MAX_HALF_ANGLE_DEG = 60.0
WARMUP_FRAMES = 20  # the map is still filling; those costs say nothing about the vfov
# The cost separates fields of view only once a new view is matched against surface
# seen under a different rotation, so the map has to fill first.
VFOV_FRAMES = 1000
# A minimum is believed when its nearer neighbor costs `MIN_DEPTH` times more, or
# `MIN_VALLEY_DEPTH` times on a curve that falls to it from both ends (within
# `FLAT_TOL`). A wide lens gives one (1.2-2.6: FicTrac's webcam sample, a crab rig's
# consumer camera; 1.06-1.08 in a clean valley: a webcam over a fiddler crab, a 40
# frame synthetic clip). A shallower or ragged one is a model mismatch, such as lens
# distortion, not a wide lens: a lab octacam (2 deg lens) leaned to 26 deg at 1.02
# and read rotations 18% low; a Tuthill-lab pose camera (0.5 deg) dipped twice, to 11
# and 37 deg.
MIN_DEPTH = 1.15
MIN_VALLEY_DEPTH = 1.05
# Without a believable minimum the lens is assumed narrow, unless the narrow end costs
# `NARROW_MAX_COST` times the lowest (the crab rig's narrowest candidates cost 4x,
# the two lenses above 1.3x and 1.6x), or `LEAN_COST` times on less rotation than
# `LEAN_MIN_TURN_DEG`, where a lean says too little either way (a crab clip whose lens
# is wide leaned so after only 180 deg).
NARROW_MAX_COST = 2.0
LEAN_COST = 1.2
LEAN_MIN_TURN_DEG = 500.0
# Candidates within this factor of the minimum count as indistinguishable from it.
FLAT_TOL = 1.05
# Below this much accumulated rotation the recording says nothing about the geometry.
# A still start is tracked further, up to `VFOV_MAX_FRAMES`, for the rotation to come.
MIN_TURN_DEG = 90.0
VFOV_MAX_FRAMES = 6000


@dataclass
class VfovFit:
    """The field of view the photometric cost prefers, and how much it prefers it."""

    vfov: float
    identifiable: bool
    spread: float  # max cost / min cost over the grid
    flat_range: tuple[float, float] | None  # vfovs whose cost is within FLAT_TOL
    turned_deg: float = 0.0  # rotation the ball showed while the curve was measured
    depth: float = 1.0  # cost at the minimum's nearer neighbor, over the minimum
    curve: list[tuple[float, float]] = field(default_factory=list)
    # How much the rotation scale changes over `flat_range`, or, when a narrow lens is
    # assumed against `leaning` (the vfov the cost leans to without a clear minimum),
    # how much smaller the rotations would read there.
    scale_spread: float = 0.0
    leaning: float | None = None

    def report(self) -> dict:
        return {
            "value": self.vfov,
            "source": "auto",
            "identifiable": self.identifiable,
            "spread": self.spread,
            "flat_range": list(self.flat_range) if self.flat_range else None,
            "turned_deg": self.turned_deg,
            "depth": self.depth,
            "scale_spread": self.scale_spread,
            "leaning": self.leaning,
            "curve": [list(point) for point in self.curve],
        }

    def line(self) -> str:
        if self.identifiable:
            return (
                f"{self.vfov:.4g} deg (fitted; the cost minimum is {self.depth:.2g}x "
                f"below its neighbors)"
            )
        pct = (
            f"{100 * self.scale_spread:.0f}%"
            if np.isfinite(self.scale_spread)
            else "an unknown amount"
        )
        if self.leaning is not None:
            return (
                f"{self.vfov:.4g} deg (assumed, a narrow lens: the cost leans to "
                f"{self.leaning:.3g} deg without a clear minimum, which would read "
                f"rotations {pct} smaller; set camera.vfov_deg from the lens)"
            )
        lo, hi = self.flat_range or (float("nan"), float("nan"))
        return (
            f"{self.vfov:.4g} deg (fitted; not identifiable, the cost is flat over "
            f"{lo:.3g}-{hi:.3g} deg, over which the rotation scale changes by {pct})"
        )


@dataclass
class Prepared:
    """What automatic preparation found, for the run summary and the sidecar."""

    detection: BallDetection | None = None
    ball_source: str = "config"  # "config" or "detected"
    config_radius_px: float | None = None
    radius_disagreement: float | None = None  # detected / config - 1
    vfov: VfovFit | None = None
    camera_position: str = "config"  # "config", "estimated", "unknown"
    camera: CameraFit | None = None
    notes: list[str] = field(default_factory=list)
    texture: float | None = None  # `ball_texture` of a frame, gray levels

    def report(self) -> dict:
        """The `ball` entry of the sidecar's provenance."""
        out: dict = {"source": self.ball_source}
        if self.texture is not None:
            out["texture"] = self.texture
        if self.detection is not None:
            out.update(
                center_px=[self.detection.cx, self.detection.cy],
                radius_px=self.detection.r,
                confidence=self.detection.confidence,
                rim_fraction=self.detection.rim_fraction,
                residual_px=self.detection.residual_px,
                n_frames=self.detection.n_frames,
                model_score=self.detection.model_score,
            )
        if self.config_radius_px is not None:
            out["config_radius_px"] = self.config_radius_px
        if self.radius_disagreement is not None:
            out["radius_disagreement"] = self.radius_disagreement
        return out

    def masks(self) -> list[tuple[str, np.ndarray, float]]:
        """SAM 3's masks of the ball and the animal, where they were used, as
        `(name, mask, score)`."""
        out = []
        if self.detection is not None and self.detection.mask is not None:
            out.append(("ball", self.detection.mask, self.detection.model_score))
        camera = self.camera
        if isinstance(camera, CameraFit) and camera.mask is not None:
            out.append(("animal", camera.mask, camera.score))
        return out

    def line(self) -> str:
        """One line for the terminal block."""
        found = self._found()
        if self.texture is not None and self.texture < MIN_TEXTURE:
            found += f"; {self.texture_warning()}"
        return found

    def texture_warning(self) -> str:
        return (
            f"its surface shows little texture to track ({self.texture:.1f} gray "
            f"levels, patterned balls show 10 or more), so the rotation reported may "
            f"be the animal's or noise"
        )

    def _found(self) -> str:
        if self.detection is None:
            return "from config (detection failed, not checked)"
        d = self.detection
        where = f"({d.cx:.1f}, {d.cy:.1f}) r {d.r:.1f} px"
        if self.ball_source == "detected" and d.model_score is not None:
            return (
                f"detected at {where} by SAM 3 (score {d.model_score:.2f}), "
                f"rim confirms {100 * d.confidence:.0f}% of its outline"
            )
        if self.ball_source == "detected":
            return f"detected at {where}, confidence {d.confidence:.2f}"
        if self.radius_disagreement is None:
            return f"from config; detection says {where}"
        return (
            f"from config; detection says {where}, "
            f"{100 * self.radius_disagreement:+.1f}% on the radius"
        )


def complete_config(
    cfg: Config, src_spec, *, require_position: bool = True
) -> Prepared | None:
    """`prepare_config` when the config leaves geometry open; None when it does not.

    Raises `ValueError` (`NO_POSITION`) when nothing says where the camera sits, before
    any slow work, unless not `require_position`, which leaves the position unset.
    """
    camera = cfg.camera
    if cfg.ball.rim and camera.vfov_deg is not None and camera.to_animal() is not None:
        return None
    if require_position and not _position_given(cfg):
        raise ValueError(NO_POSITION)
    prepared = prepare_config(cfg, src_spec)
    if require_position and camera.to_animal() is None:
        raise ValueError(NO_POSITION)
    return prepared


def _position_given(cfg: Config) -> bool:
    """Whether the config says where the camera sits."""
    camera = cfg.camera
    return camera.to_animal() is not None or camera.azimuth_deg is not None


def prepare_config(
    cfg: Config,
    src_spec,
    *,
    n_frames: int = 100,
    span: int = 300,
    vfov_frames: int = VFOV_FRAMES,
    params=None,
) -> Prepared:
    """Detect the ball, place the camera and fit the field of view where unknown.

    The detection is used only when the config has no ball of its own; when it has one,
    the two are compared and the config's is kept, and a failed detection is only
    reported. Without a camera position, `place_camera` places it at
    `camera.azimuth_deg` from the animal's silhouette, and leaves it unset without that
    azimuth (`camera_position` "unknown"). The field of view is fitted with
    the pixel circle held fixed, since the two together set the ball's angular radius
    and the cost cannot separate them. Mutates `cfg`.
    """
    if str(src_spec).isdigit():
        raise ValueError(CAMERA_SOURCE_MESSAGE)
    frames = sample_frames(str(src_spec), n_frames, span)
    height, width = frames[0].shape
    prepared = Prepared()
    try:
        # The comparison with a configured ball is only a check, not worth a model.
        prepared.detection = find_ball(frames, n_frames, use_model=not cfg.ball.rim)
    except DetectionError as exc:
        if not cfg.ball.rim:
            raise
        log.warning("ball detection failed, keeping the config's ball: %s", exc)
    detection = prepared.detection

    if cfg.ball.rim:
        radius = fit_circle(cfg.ball.rim)[2]
        if detection is not None:
            prepared.config_radius_px = radius
            prepared.radius_disagreement = detection.r / radius - 1.0
            if abs(prepared.radius_disagreement) > DISAGREEMENT_WARN:
                message = (
                    f"detected ball radius {detection.r:.1f} px differs from the "
                    f"config's {radius:.1f} px by "
                    f"{100 * prepared.radius_disagreement:+.1f}%; that is about twice "
                    f"as much again on every in-plane rotation reported"
                )
                log.warning("%s", message)
                prepared.notes.append(message)
    else:
        prepared.ball_source = "detected"
        cfg.ball.rim = _circle_points(detection)
        vfov, fisheye = cfg.camera.vfov_deg, cfg.camera.fisheye
        if vfov is not None:
            camera = source_camera(width, height, vfov, fisheye)
            half_angle = fit_ball(cfg.ball.rim, camera)[1]
            prepared.notes.append(
                f"ball fitted from the recording: half-angle "
                f"{np.degrees(half_angle):.4f} deg at confidence "
                f"{detection.confidence:.2f}"
            )

    prepared.texture = ball_texture(frames[len(frames) // 2], fit_circle(cfg.ball.rim))
    if prepared.texture < MIN_TEXTURE:
        log.warning("the ball: %s", prepared.texture_warning())
    if cfg.camera.to_animal() is None:
        picks = np.linspace(0, len(frames), ANIMAL_FRAMES + 2)[1:-1].astype(int)
        prepared.camera = place_camera(
            [frames[i] for i in picks], cfg.ball.rim, cfg.camera.azimuth_deg
        )
        position = prepared.camera.position_deg
        prepared.camera_position = "unknown" if position is None else "estimated"
        cfg.camera.position_deg = position
    if cfg.camera.vfov_deg is None:
        circle = fit_circle(cfg.ball.rim)
        prepared.vfov = fit_vfov(
            src_spec, cfg, circle, n_frames=vfov_frames, params=params
        )
        cfg.camera.vfov_deg = prepared.vfov.vfov
    return prepared


@dataclass
class CameraFit:
    """The camera's elevation and twist from where the animal stands on the ball, and
    the azimuth the config gave (`camera.azimuth_deg`)."""

    azimuth_deg: float | None  # None: unknown, and so the position
    reason: str  # what decided the elevation and the twist
    elevation_deg: float = 0.0  # level, unless the silhouette said otherwise
    twist_deg: float = 0.0
    measured: bool = False  # whether the silhouette gave them
    angle_deg: float | None = None  # the animal about the ball, clockwise from up
    distance: float | None = None  # from the ball's center, in ball radii
    score: float | None = None  # the animal mask's
    mask: np.ndarray | None = field(default=None, repr=False)  # the animal's, bool

    @property
    def position_deg(self) -> tuple[float, float, float] | None:
        if self.azimuth_deg is None:
            return None
        return (self.elevation_deg, float(self.azimuth_deg), self.twist_deg)

    def report(self) -> dict:
        """The sidecar's account of it."""
        out = asdict(_replace(self, mask=None))
        return {k: v for k, v in out.items() if v is not None}

    def line(self) -> str:
        """One line for the terminal block."""
        if self.measured:
            rest = (
                f"elevation {self.elevation_deg:g}, twist {self.twist_deg:g} deg from "
                f"where the animal stands ({self.reason})"
            )
        else:
            rest = f"level assumed ({self.reason})"
        if self.azimuth_deg is None:
            return f"unknown: no azimuth given; {rest}"
        return f"azimuth {self.azimuth_deg:g} from camera.azimuth_deg; {rest}"


def ball_texture(image: np.ndarray, circle) -> float:
    """Contrast of the ball's surface at the tracking window's scale: the standard
    deviation, in gray levels, of a band-pass of `image` inside 0.8 of its radius."""
    cx, cy, r = circle
    sigma = max(1.0, r / 60)  # the window spans about 60 px across the ball
    image = np.asarray(image, np.float32)
    band = cv2.GaussianBlur(image, (0, 0), sigma) - cv2.GaussianBlur(
        image, (0, 0), 4 * sigma
    )
    rows, cols = np.ogrid[: image.shape[0], : image.shape[1]]
    inside = (cols - cx) ** 2 + (rows - cy) ** 2 <= (0.8 * r) ** 2
    return round(float(np.std(band[inside])), 2) if inside.any() else 0.0


def place_camera(images, rim, azimuth: float | None = None) -> CameraFit:
    """The camera at `azimuth`, its elevation and twist from where the animal stands on
    the ball in `images`, a few frames spread over the recording.

    The animal's frame has its z axis along the ball's normal where it stands, so its
    silhouette's direction from the ball's center gives the camera's twist, and its
    distance the elevation: inside the outline, at the elevation's cosine, for a camera
    above. On the outline the elevation does not show and the camera is taken for
    level, as it is without a silhouette; near the middle (a camera nearly overhead)
    the direction is too short to read and no twist is assumed. Which way the animal
    faces does not show in a silhouette, hence `azimuth`. The geometry is orthographic,
    which the narrow fields of view of trackball rigs allow.

    The animal is tethered, so it is the mask that stays put from frame to frame: a
    spot of the ball's texture can be the best "insect" in one frame, but it moves.
    """
    if isinstance(images, np.ndarray) and images.ndim == 2:
        images = [images]
    cx, cy, r = fit_circle(rim)
    rows, cols = np.ogrid[: images[0].shape[0], : images[0].shape[1]]
    disk = (cols - cx) ** 2 + (rows - cy) ** 2 <= r * r
    found = []  # (frame, mask, score, x, y) on the ball, best first per frame
    for k, image in enumerate(images):
        try:
            masks, scores = segment.animal_masks(image)
        except segment.SegmenterUnavailable as exc:
            return CameraFit(azimuth, str(exc))
        for mask, score in zip(masks, scores, strict=True):
            # A mask over most of the ball is the ball (a patterned one reads as an
            # animal); one far off it is something else.
            if score < MIN_ANIMAL_SCORE or (mask & disk).sum() > MAX_ANIMAL_COVER * (
                np.pi * r * r
            ):
                continue
            ys, xs = np.nonzero(mask)
            x, y = xs.mean(), ys.mean()
            if np.hypot(x - cx, y - cy) <= OFF_BALL * r:
                found.append((k, mask, float(score), x, y))
    # The candidate seen at the same place (within `SAME_PLACE` radii) in the most
    # frames, then with the most score there.
    best, key = None, None
    for _, _, _, x0, y0 in found:
        near = [c for c in found if np.hypot(c[3] - x0, c[4] - y0) <= SAME_PLACE * r]
        votes = (len({c[0] for c in near}), sum(c[2] for c in near))
        if key is None or votes > key:
            best, key = near, votes
    if best is None or key[0] < min(len(images), MIN_VOTES):
        return CameraFit(azimuth, "no animal found that stays on the ball")
    _, mask, score, _, _ = max(best, key=lambda c: c[2])
    dx = float(np.median([c[3] for c in best])) - cx
    dy = float(np.median([c[4] for c in best])) - cy
    distance = float(np.hypot(dx, dy) / r)
    angle = float(np.degrees(np.arctan2(dx, -dy)))
    fit = CameraFit(azimuth, "", angle_deg=round(angle, 1), mask=mask)
    fit.distance, fit.score = round(distance, 3), round(score, 2)
    on_rim = distance >= ON_RIM
    elevation = 0.0 if on_rim else float(np.degrees(np.arccos(distance)))
    # A camera twisted clockwise sees the animal turned anticlockwise.
    twist = 0.0 if distance < OVERHEAD else -angle
    fit.elevation_deg, fit.twist_deg = round(elevation, 1), round(twist, 1) + 0.0
    fit.measured = True
    if on_rim:
        where = (
            "on the ball's outline, where the elevation does not show and level is "
            "assumed"
        )
    elif distance < OVERHEAD:
        where = (
            "near the middle of the ball, seen from nearly straight above, where it "
            "does not show the twist: none is assumed, and the azimuth says which way "
            "it faces in the image (180 up, 90 right, 0 down, -90 left)"
        )
    else:
        where = "inside the ball's outline"
    fit.reason = f"it stands {where}; SAM 3 score {score:.2f}"
    return fit


def find_ball(frames, n_frames: int = 100, *, use_model: bool = True) -> BallDetection:
    """The ball in a run of frames; raises `DetectionError` when it cannot be trusted.

    SAM 3 proposes the ball's silhouette in the frames' temporal quantile and
    `ball_from_masks` measures and checks the rim; its refusal is final. Without
    `use_model`, or when the model cannot be loaded (offline before its first
    download), the classical detector runs instead, which finds far fewer balls but
    refuses rather than guess.
    """
    if use_model:
        hi, _, n = temporal_stats(frames, n_frames)
        try:
            masks, scores = segment.ball_masks(np.clip(hi, 0, 255).astype(np.uint8))
        except segment.SegmenterUnavailable as exc:
            log.warning("%s; using the classical detector, which finds fewer", exc)
        else:
            return ball_from_masks(hi, masks, scores, n_frames=n)
    return detect_ball(frames, max_frames=n_frames)


def _circle_points(circle, n: int = 16) -> list[tuple[int, int]]:
    """Rim points, as `ball.rim` holds them, of a `BallDetection` or a `(cx, cy, r)`."""
    if isinstance(circle, BallDetection):
        flat = circle.rim_points(n)
    else:
        flat = circle_points(*circle, n)
    return list(zip(flat[::2], flat[1::2], strict=True))


def _costs_at(
    src_spec, cfg: Config, points, vfovs, n_frames: int, params, max_frames=None
) -> tuple[np.ndarray, np.ndarray]:
    """Median cost and turned angle (deg) of tracking the first frames at each vfov:
    `n_frames`, or more, up to `max_frames`, until the ball turned `MIN_TURN_DEG`.

    All candidates track in lockstep, so the video is decoded once. A candidate whose
    geometry is impossible (the rim points do not describe a ball there, or the ball
    would cover most of the sky) or whose tracker fails scores NaN.
    """
    from spintrack.engine import TrackParams
    from spintrack.io.sources import VideoSource
    from spintrack.tracker import Tracker

    # This search lives on the cost's sensitivity to the geometry, so what explains
    # away part of a wrong field of view's cost is off: the static illumination field
    # absorbs a systematic misregistration as shading, and the pre-filter blurs it.
    params = _replace(
        params or TrackParams(),
        prefilter=0.0,
        illum_bias=False,
        center_watch=False,
    )
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    source = VideoSource(src_spec)
    trackers: list = []
    costs: list[list[float]] = [[] for _ in vfovs]
    turned = np.zeros(len(vfovs))
    fisheye = cfg.camera.fisheye
    try:
        for vfov in vfovs:
            trial = cfg.model_copy(
                update={
                    "camera": cfg.camera.model_copy(
                        update={
                            "vfov_deg": float(vfov),
                            "position_deg": None,
                            "rotation": (0.0, 0.0, 0.0),
                        }
                    ),
                    "ball": BallConfig(rim=[(round(x), round(y)) for x, y in pts]),
                }
            )
            try:
                camera = source_camera(source.width, source.height, vfov, fisheye)
                _, half = fit_ball(pts, camera)
                if not np.isfinite(half) or np.degrees(half) > MAX_HALF_ANGLE_DEG:
                    raise ValueError("impossible geometry")
                trackers.append(Tracker(trial, source.width, source.height, params))
            except ValueError, np.linalg.LinAlgError:
                trackers.append(None)
        for index in range(max(n_frames, max_frames or 0)):
            if index >= n_frames:
                moved = turned[turned > 0]
                if not len(moved) or np.degrees(np.median(moved)) >= MIN_TURN_DEG:
                    break
            frame = source.read()
            if frame is None:
                break
            for k, tracker in enumerate(trackers):
                if tracker is None:
                    continue
                try:
                    result = tracker.process_frame(frame.image, frame.ts_ms)
                except ValueError, np.linalg.LinAlgError:
                    trackers[k], costs[k], turned[k] = None, [], 0.0
                    continue
                if result is not None:
                    turned[k] += float(np.linalg.norm(result.w_cam))
                    if index >= WARMUP_FRAMES:
                        costs[k].append(result.step.cost)
    finally:
        source.close()
    out = []
    for values in costs:
        values = [c for c in values if np.isfinite(c)]
        out.append(float(np.median(values)) if len(values) >= 10 else float("nan"))
    return np.array(out), np.degrees(turned)


def fit_vfov(
    src_spec,
    cfg: Config,
    circle,
    *,
    n_frames: int = VFOV_FRAMES,
    grid=None,
    params=None,
) -> VfovFit:
    """Fit the field of view from the cost, holding the ball's pixel circle fixed.

    Where the ball is small in the frame the curve is flat and any value in the flat
    region tracks alike; the fit says so rather than pretending to a number. Without a
    clear minimum the lens is taken for narrow, as on most trackball rigs, and the fit
    says how much smaller the rotations would read at the vfov the cost leans to.
    """
    lo, hi, count = grid or VFOV_GRID
    log.info(
        "fitting the field of view: tracking %d frames at each candidate", n_frames
    )
    points = _circle_points(circle)
    values = np.geomspace(lo, hi, count)
    costs, turns = _costs_at(
        src_spec, cfg, points, values, n_frames, params, VFOV_MAX_FRAMES
    )
    for value, cost in zip(values, costs, strict=True):
        log.debug("vfov %.3g deg: cost %.5g", value, cost)
    curve = [(float(v), float(c)) for v, c in zip(values, costs, strict=True)]
    usable = np.isfinite(costs)
    if usable.sum() < 3:
        raise ValueError("could not track at any candidate field of view")
    turned_deg = float(np.median(turns[turns > 0])) if (turns > 0).any() else 0.0
    if turned_deg < MIN_TURN_DEG:
        raise ValueError(
            f"the ball turned only {turned_deg:.0f} deg in the frames tracked to fit "
            f"the field of view (up to {VFOV_MAX_FRAMES}), too little to fit it from; "
            f"set camera.vfov_deg from the lens"
        )
    values, costs, turns = values[usable], costs[usable], turns[usable]
    best = int(np.argmin(costs))
    spread = float(costs.max() / costs.min())

    def turned_at(vfov: float) -> float:  # the rotation read at `vfov`
        return float(np.interp(np.log(vfov), np.log(values), turns))

    # An interior minimum clearly below both neighbors on a curve that is not flat end
    # to end. On a flat curve (the near-orthographic regime) which candidate comes out
    # lowest is noise.
    depth = 1.0
    if spread > FLAT_TOL and 0 < best < len(costs) - 1:
        depth = float(min(costs[best - 1], costs[best + 1]) / costs[best])
        if depth >= MIN_DEPTH or (
            depth >= MIN_VALLEY_DEPTH and _one_valley(costs, best)
        ):
            inner = np.geomspace(values[best - 1], values[best + 1], VFOV_REFINE + 2)
            inner = inner[1:-1]
            more, _ = _costs_at(src_spec, cfg, points, inner, n_frames, params)
            for value, cost in zip(inner, more, strict=True):
                log.debug("vfov %.3g deg: cost %.5g", value, cost)
            vfov = _parabola_minimum(
                np.r_[values, inner], np.r_[costs, more], values[[best - 1, best + 1]]
            )
            extra = [(float(v), float(c)) for v, c in zip(inner, more, strict=True)]
            flat_range = _flat_run(values, costs, best)
            return VfovFit(
                vfov, True, spread, flat_range, turned_deg, depth, curve + extra
            )

    # No believable minimum: the narrow end of the curve, as far as it stays level,
    # unless the curve rules a narrow lens out.
    lean = float(costs[0] / costs[best])
    if lean > NARROW_MAX_COST or (lean > LEAN_COST and turned_deg < LEAN_MIN_TURN_DEG):
        why = (
            "rules out a narrow lens"
            if lean > NARROW_MAX_COST
            else f"the ball turned only {turned_deg:.0f} deg, too little to tell"
        )
        raise ValueError(
            f"the photometric cost leans to {values[best]:.3g} deg ({lean:.2g}x the "
            f"cost at {values[0]:g} deg) without a clear minimum, and {why}, so the "
            f"field of view cannot be fitted from this recording; set "
            f"camera.vfov_deg from the lens"
        )
    narrow = _level_run(values, costs)
    vfov = float(np.sqrt(narrow[0] * narrow[1]))
    fit = VfovFit(vfov, False, spread, narrow, turned_deg, depth, curve)
    if values[best] > narrow[1]:  # the cost leans wider, without a clear minimum
        fit.leaning = float(values[best])
        fit.scale_spread = 1.0 - turned_at(fit.leaning) / turned_at(vfov)
        log.warning("field of view: %s", fit.line())
    else:
        fit.scale_spread = turned_at(narrow[0]) / turned_at(narrow[1]) - 1.0
    return fit


def _parabola_minimum(values, costs, bracket) -> float:
    """Minimum of a parabola in `log(vfov)` through the best point and its neighbors."""
    ok = np.isfinite(costs)
    order = np.argsort(values[ok])
    x, c = np.log(values[ok][order]), costs[ok][order]
    best = int(np.argmin(c))
    if not 0 < best < len(c) - 1:
        return float(np.exp(x[best]))
    (x0, x1, x2), (c0, c1, c2) = x[best - 1 : best + 2], c[best - 1 : best + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (c1 - c0) + x1 * (c0 - c2) + x0 * (c2 - c1)) / denom
    b = (x2**2 * (c0 - c1) + x1**2 * (c2 - c0) + x0**2 * (c1 - c2)) / denom
    if a <= 0:
        return float(np.exp(x1))
    lo, hi = np.log(bracket)
    return float(np.exp(np.clip(-b / (2.0 * a), max(x0, lo), min(x2, hi))))


def _one_valley(costs, best) -> bool:
    """Whether the curve falls to `best` from both ends, up to `FLAT_TOL` of noise."""
    left = all(costs[i] * FLAT_TOL >= costs[i + 1] for i in range(best))
    right = all(
        costs[i] * FLAT_TOL >= costs[i - 1] for i in range(best + 1, len(costs))
    )
    return left and right


def _level_run(values, costs) -> tuple[float, float]:
    """Fields of view from the narrowest on whose cost stays within `FLAT_TOL` of the
    narrowest's, either way."""
    hi = 0
    while hi < len(costs) - 1 and abs(np.log(costs[hi + 1] / costs[0])) <= np.log(
        FLAT_TOL
    ):
        hi += 1
    return float(values[0]), float(values[hi])


def _flat_run(values, costs, best) -> tuple[float, float]:
    """Fields of view around `best` whose cost is indistinguishable from the minimum.

    Contiguous on purpose: an isolated candidate elsewhere that comes within `FLAT_TOL`
    is a wobble, not evidence that the range between is flat.
    """
    limit = FLAT_TOL * costs[best]
    lo = hi = best
    while lo > 0 and costs[lo - 1] <= limit:
        lo -= 1
    while hi < len(costs) - 1 and costs[hi + 1] <= limit:
        hi += 1
    return float(values[lo]), float(values[hi])
