"""A camera from a deeperfly calibration: its field of view and where it sits.

deeperfly (multi-camera pose estimation) keeps a project's calibrations under
`calibrations/NNNN-<label>.toml`, each a `[views.<name>]` table per camera: solved
(`rvec`, `tvec` world to camera as OpenCV has them, `intr` `[fx, fy, cx, cy]`,
`image_size` `[height, width]`) or an orbit placement (`azimuth_deg`, `elevation_deg`,
`roll_deg`, `focal`). Its manifest, `deeperfly.toml`, names the active one and each
view's videos. deeperfly's world is the rig's version of the animal frame here: x
forward (its front camera sits at azimuth 0), y left, z up.

The animal is never mounted exactly along the rig's x axis (on the lab's octacam, up to
12 deg off), nor on the ball's top (there, 9-13 deg toward the hind camera), so when
the project holds pose results (`results/*.h5`), the camera is placed relative to the
animal rather than to the rig. Its z axis is the ball's normal where it stands: a ball
is fitted to its leg tips, which rest on it in stance, and the normal runs through its
thorax-coxa points (`on_ball`). Its x axis is the body's long axis, from front to hind
thorax-coxa points, on the tangent plane there. Until the ball's image is known, the
rig's up stands in for the normal and only the heading comes from the body.
"""

from __future__ import annotations

import functools
import json
import logging
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from spintrack.calibrate.sliders import angles_from_camera_to_lab
from spintrack.geometry import rotvec_to_matrix

log = logging.getLogger("spintrack")

MANIFEST = "deeperfly.toml"
ASPECT_TOL = 0.01  # how far a video's aspect ratio may be from the calibration's
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
# The thorax-coxa points whose front and hind midpoints give the body's long axis.
FRONT = ("lf_thorax_coxa", "rf_thorax_coxa")
HIND = ("lh_thorax_coxa", "rh_thorax_coxa")
# Every leg's tip, which rests on the ball in stance, and its thorax-coxa point; the
# centroid of those is where the body stands.
TIPS = tuple(f"{leg}_pretarsus" for leg in LEGS)
COXAE = tuple(f"{leg}_thorax_coxa" for leg in LEGS)
MIN_CONFIDENCE = 0.5  # a detection weaker than this is not triangulated
MAX_POSE_FRAMES = 400  # frames sampled for the heading; the animal is tethered
# The ball is fitted in the opening frames, where its image circle was measured: a
# ball can move in its holder later.
CONTACT_FRAMES = 1000
# A leg tip this near the ball's surface, in radii, rests on it. On the lab's octacam
# the triangulated tips scatter about 2% of the radius about the surface, and 73-84%
# of them rest on it in the opening frames.
ON_SURFACE = 0.02
MIN_ON_SURFACE = 0.3  # fewer tips on the fitted ball: not a ball
MAX_HEIGHT = 0.5  # the thorax-coxa centroid above the surface, in radii (octacam: 0.13)


@dataclass(frozen=True)
class Contact:
    """Where the animal stands on the ball, from a ball fitted to its leg tips."""

    tilt_deg: float  # the ball's normal there, from the rig's up
    toward_deg: float  # where it tilts to, counterclockwise from the rig's x
    radius: float  # the ball's, in the calibration's units
    on_surface: float  # the share of leg tips that rest on it
    height: float  # the thorax-coxa centroid above the surface, in ball radii


@dataclass(frozen=True)
class CalibratedView:
    """One view of a calibration, as spintrack needs it."""

    file: Path
    name: str
    to_animal: np.ndarray  # `v_animal = to_animal @ v_camera`
    focal_px: float  # the vertical focal length, at `image_size`
    image_size: tuple[int, int] | None  # (height, width); None: the video's own
    distorted: bool  # it has lens distortion, which spintrack does not model
    # The animal's heading in the rig (deg, counterclockwise from x seen from above),
    # which `to_animal` accounts for, and the pose results that gave it.
    heading_deg: float | None = None
    results: Path | None = None
    contact: Contact | None = None  # where the animal stands, once `on_ball` found it

    def vfov_deg(self, height: int) -> float:
        """The vertical field of view, for a video `height` pixels tall."""
        calibrated = self.image_size[0] if self.image_size else height
        return float(np.degrees(2.0 * np.arctan(calibrated / (2.0 * self.focal_px))))

    @property
    def position_deg(self) -> tuple[float, float, float]:
        return angles_from_camera_to_lab(self.to_animal)

    def line(self) -> str:
        el, az, tw = self.position_deg
        line = (
            f"elevation {el:.2f}, azimuth {az:.2f}, twist {tw:.2f} deg from the "
            f"deeperfly calibration {self.file}, view {self.name}"
        )
        if self.heading_deg is None:
            return line + ", relative to the rig (no pose results to find the animal)"
        if self.contact is not None:
            c = self.contact
            return line + (
                f", relative to the animal, which stands {c.tilt_deg:.1f} deg from the "
                f"ball's top (toward {c.toward_deg:.0f} deg in the rig) and points "
                f"{self.heading_deg:+.2f} deg from the rig's x axis; ball fitted to "
                f"{100 * c.on_surface:.0f}% of its leg tips in {self.results.name}"
            )
        return line + (
            f", relative to the animal, whose body points {self.heading_deg:+.2f} deg "
            f"from the rig's x axis in {self.results.name}"
        )

    def report(self) -> dict:
        out = {
            "calibration": str(self.file),
            "view": self.name,
            "position_deg": list(self.position_deg),
            "distorted": self.distorted,
        }
        if self.heading_deg is not None:
            out.update(heading_deg=self.heading_deg, results=str(self.results))
        if self.contact is not None:
            out["contact"] = asdict(self.contact)
        return out


def read_view(path, view: str | None = None, video=None) -> CalibratedView:
    """The view `view` of the calibration at `path`, or the one that filmed `video`.

    `path` is a calibration file, a manifest (its active calibration) or a folder
    holding a manifest, directly or in `deeperfly/`. Without `view`, the manifest's
    videos say which view filmed `video`; a calibration of one view needs neither.
    Raises `ValueError` when the view cannot be told or is not supported.
    """
    calibration, manifest = _locate(Path(path))
    views = _views(calibration)
    names = {name.lower(): name for name in views}
    if view is None and video is not None and manifest is not None:
        view = _view_of(manifest, video)
    if view is None and len(views) == 1:
        view = next(iter(views))
    if view is None or view.lower() not in names:
        known = ", ".join(views)
        what = f"no view {view!r}" if view else "which view filmed the video is unknown"
        raise ValueError(f"{calibration}: {what}; set camera.view to one of {known}")
    name = names[view.lower()]
    spec = views[name]
    if spec.get("mirrored"):
        raise ValueError(
            f"{calibration}: view {name} is mirrored, which is unsupported"
        )
    to_animal, heading, results = _world_to_camera(spec).T, None, None
    found = body_heading(calibration.parent.parent, views)
    if found is not None:
        heading, results = found
        c, s = np.cos(np.radians(heading)), np.sin(np.radians(heading))
        world_to_animal = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        to_animal = world_to_animal @ to_animal
    return CalibratedView(
        file=calibration,
        name=name,
        to_animal=to_animal,
        focal_px=_focal(spec, calibration, name),
        image_size=tuple(spec["image_size"]) if "image_size" in spec else None,
        distorted=any(abs(float(k)) > 0 for k in spec.get("dist", ())),
        heading_deg=heading,
        results=results,
    )


def body_heading(project: Path, views: dict[str, dict]) -> tuple[float, Path] | None:
    """The animal's heading in the rig, from the newest single-animal pose results of
    the deeperfly `project` folder: degrees counterclockwise from x, and the file.

    The median of the front and hind thorax-coxa midpoints' difference gives the body's
    long axis; its horizontal part is the heading. None when there are no such results.
    """
    results = _pose_results(project)
    if results is None:
        return None
    X = _triangulate(results, views, FRONT + HIND)
    if X is None:
        return None
    axis = 0.5 * (X[:, 0] + X[:, 1]) - 0.5 * (X[:, 2] + X[:, 3])  # front - hind
    axis = np.nanmedian(axis, axis=0)
    if not np.isfinite(axis[:2]).all() or np.hypot(*axis[:2]) == 0:
        return None
    return float(np.degrees(np.arctan2(axis[1], axis[0]))), results


def on_ball(view: CalibratedView, circle, frame_hw) -> CalibratedView:
    """`view`, placed relative to where the animal stands on the ball.

    The ball's center lies on the ray through the center of its image `circle`
    (`(cx, cy, r)` in the continuous pixels of a video `frame_hw` tall and wide), and
    its radius follows from the circle's angular size and the distance along that ray,
    which is the one that rests the most leg tips on its surface. The animal's z axis
    is the ball's normal through its thorax-coxa centroid, its x axis the body's long
    axis on the tangent plane there. `view` comes back as it was without pose results
    with leg tips, or when the fit is not believable: too few tips on the ball, or the
    body not just above it.
    """
    if view.results is None:
        return view
    views = _views(view.file)
    size = view.image_size or tuple(frame_hw)
    K, dist, P = _camera(views[view.name], size)
    R, t = P[:, :3], P[:, 3]
    k = size[0] / frame_hw[0]  # the video's pixels to the calibration's
    cx, cy, r = (k * float(v) for v in circle)
    # Continuous pixel coordinates to OpenCV's, whose pixel centers are integers.
    xy = cv2.undistortPoints(np.array([[[cx - 0.5, cy - 0.5]]]), K, dist).reshape(2)
    origin, ray = -R.T @ t, R.T @ np.array([xy[0], xy[1], 1.0])
    ray /= np.linalg.norm(ray)
    sin_a = float(np.sin(np.arctan(r / K[1, 1])))
    try:
        X = _triangulate(view.results, views, TIPS + COXAE, stop=CONTACT_FRAMES)
    except ValueError:  # the results have no leg tips
        return view
    if X is None:
        return view
    tips = X[:, : len(TIPS)].reshape(-1, 3)
    tips = tips[np.isfinite(tips).all(axis=1)]
    coxae = np.nanmedian(X[:, len(TIPS) :], axis=0)
    body = coxae.mean(axis=0)
    if len(tips) < 10 or not np.isfinite(body).all():
        return view
    depth, share = _ball_depth(tips, origin, ray, sin_a, float((body - origin) @ ray))
    radius = depth * sin_a
    z = body - (origin + depth * ray)
    height = float(np.linalg.norm(z) / radius - 1.0)
    if share < MIN_ON_SURFACE or not 0.0 < height < MAX_HEIGHT:
        log.warning(
            "not placing the camera where the animal stands on the ball: %.0f%% of its "
            "leg tips rest on the ball fitted to them, and its body is %.2f radii "
            "above it; the rig's up stands in for the ball's normal",
            100 * share, height,
        )  # fmt: skip
        return view
    z /= np.linalg.norm(z)
    i = [COXAE.index(name) for name in FRONT + HIND]
    axis = 0.5 * (coxae[i[0]] + coxae[i[1]]) - 0.5 * (coxae[i[2]] + coxae[i[3]])
    x = axis - (axis @ z) * z
    x /= np.linalg.norm(x)
    world_to_animal = np.array([x, np.cross(z, x), z])
    contact = Contact(
        tilt_deg=float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0)))),
        toward_deg=float(np.degrees(np.arctan2(z[1], z[0]))),
        radius=float(radius),
        on_surface=share,
        height=height,
    )
    return replace(
        view,
        to_animal=world_to_animal @ R.T,
        heading_deg=float(np.degrees(np.arctan2(x[1], x[0]))),
        contact=contact,
    )


def _ball_depth(tips, origin, ray, sin_a: float, start: float) -> tuple[float, float]:
    """The ball center's distance along `ray` that rests the most `tips` on its
    surface (the radius is `sin_a` times the distance), searched within two radii of
    `start` and refined by least squares on those tips; and their share."""
    depths = start + start * sin_a * np.linspace(-2.0, 2.0, 401)
    centers = origin + depths[:, None] * ray
    gap = np.linalg.norm(tips - centers[:, None], axis=-1) - sin_a * depths[:, None]
    on = np.abs(gap) < ON_SURFACE * sin_a * depths[:, None]
    depth = float(depths[np.argmax(on.sum(axis=1))])
    for _ in range(5):  # Gauss-Newton on the distance, over the tips on the surface
        offset = tips - (origin + depth * ray)
        distance = np.linalg.norm(offset, axis=1)
        gap = distance - sin_a * depth
        on = np.abs(gap) < ON_SURFACE * sin_a * depth
        if not on.any():
            break
        slope = -(offset[on] / distance[on, None]) @ ray - sin_a
        depth -= float(slope @ gap[on] / (slope @ slope))
    return depth, float(on.mean())


def _triangulate(
    results: Path, views: dict[str, dict], names, stop: int | None = None
) -> np.ndarray | None:
    """The keypoints `names` of `results`, triangulated with `views` in up to
    `MAX_POSE_FRAMES` frames of the first `stop` (of all, without): `(frames, points,
    3)`, NaN where fewer than two views saw a point. None with fewer than two views."""
    import h5py

    with h5py.File(results) as f:
        keypoints = json.loads(f.attrs["keypoints"])
        result_views = json.loads(f.attrs["views"])
        frame_sizes = json.loads(f.attrs.get("frame_sizes", "{}"))
        n = f["pose2d/points"].shape[2]
        n = n if stop is None else min(n, stop)
        frames = np.arange(0, n, max(1, n // MAX_POSE_FRAMES))
        index = [keypoints.index(k) for k in names]
        points = f["pose2d/points"][0][:, frames][:, :, index].astype(np.float64)
        confidence = f["pose2d/conf"][0][:, frames][:, :, index].astype(np.float64)
    by_name = {name.lower(): spec for name, spec in views.items()}
    used = [
        i
        for i, view in enumerate(result_views)
        if view.lower() in by_name and not by_name[view.lower()].get("mirrored")
    ]
    if len(used) < 2:
        return None
    rows = []  # per view: (frames, points, 2 rows, 4)
    for i in used:
        spec = by_name[result_views[i].lower()]
        K, dist, P = _camera(spec, frame_sizes.get(result_views[i]))
        flat = points[i].reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(flat, K, dist).reshape(points[i].shape)
        ok = (confidence[i] >= MIN_CONFIDENCE) & np.isfinite(normalized).all(axis=-1)
        x, y = normalized[..., :1], normalized[..., 1:]
        view_rows = np.stack([x * P[2] - P[0], y * P[2] - P[1]], axis=-2)
        rows.append(np.where(ok[..., None, None], view_rows, 0.0))
    A = np.concatenate(rows, axis=-2)  # (frames, points, 2 * views, 4)
    seen = (np.abs(A).sum(axis=-1) > 0).sum(axis=-1) >= 4  # two views or more
    X = np.linalg.svd(A)[2][..., -1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen[..., None], X[..., :3] / X[..., 3:], np.nan)


def _pose_results(project: Path) -> Path | None:
    """The newest pose results of one animal with thorax-coxa points, if any."""
    folder = project / "results"
    if not folder.is_dir():
        return None
    import h5py

    newest = None
    for path in folder.glob("*.h5"):
        try:
            with h5py.File(path) as f:
                names = json.loads(f.attrs.get("keypoints", "[]"))
                ok = (
                    "pose2d/points" in f
                    and int(f.attrs.get("animals", 1)) == 1
                    and set(FRONT + HIND) <= set(names)
                )
                written = float(f.attrs.get("written", path.stat().st_mtime))
        except OSError, ValueError:
            continue
        if ok and (newest is None or written > newest[0]):
            newest = (written, path)
    return None if newest is None else newest[1]


def _camera(spec: dict, frame_hw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A view's intrinsic matrix, distortion and `[R | t]`, world to camera."""
    R = _world_to_camera(spec)
    if "rvec" in spec:
        t = np.asarray(spec.get("tvec", (0.0, 0.0, 0.0)), dtype=np.float64)
    else:  # an orbit placement: the camera at `distance` from `look_at`
        az, el = np.radians(
            [spec.get("azimuth_deg", 0.0), spec.get("elevation_deg", 0.0)]
        )
        toward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        center = (
            np.asarray(spec.get("look_at", (0.0, 0.0, 0.0))) + spec["distance"] * toward
        )
        t = -R @ center
    intr = spec.get("intr")
    if intr is not None and len(intr) == 4:
        fx, fy, cx, cy = intr
    elif intr is not None:
        fx = fy = intr[0]
        cx, cy = intr[1], intr[2]
    else:
        h, w = frame_hw if frame_hw else spec.get("image_size", (0, 0))
        fx = fy = spec["focal"]
        cx, cy = spec.get("cx", (w - 1) / 2), spec.get("cy", (h - 1) / 2)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist = np.asarray(spec.get("dist", ()), dtype=np.float64)
    return K, dist, np.column_stack([R, t])


def find_manifest(video) -> Path | None:
    """A deeperfly manifest next to `video` (or in `deeperfly/` there) listing it."""
    folder = Path(video).parent
    for manifest in (folder / MANIFEST, folder / "deeperfly" / MANIFEST):
        if manifest.is_file() and _view_of(manifest, video) is not None:
            return manifest
    return None


def check_size(view: CalibratedView, width: int, height: int) -> None:
    """Refuse a video whose frame is not the calibration's, up to a scale."""
    if view.image_size is None:
        return
    h, w = view.image_size
    if abs((width / height) / (w / h) - 1.0) > ASPECT_TOL:
        raise ValueError(
            f"{view.file}: view {view.name} was calibrated at {w}x{h}, which a "
            f"{width}x{height} video is not a scaled copy of"
        )


def _locate(path: Path) -> tuple[Path, Path | None]:
    """The calibration file `path` means, and the manifest it belongs to, if any."""
    if path.is_dir():
        for manifest in (path / MANIFEST, path / "deeperfly" / MANIFEST):
            if manifest.is_file():
                return _locate(manifest)
        raise ValueError(f"{path}: no {MANIFEST} in it or in its deeperfly/ folder")
    data = _toml(path)
    if "views" in data:  # a calibration; its project's manifest, if it has one
        manifest = path.parent.parent / MANIFEST
        return path, manifest if manifest.is_file() else None
    if "active_calibration" not in data:
        raise ValueError(f"{path}: neither a deeperfly calibration nor a manifest")
    cid = int(data["active_calibration"])
    for file in sorted((path.parent / "calibrations").glob("*.toml")):
        number = file.name.partition("-")[0]
        if number.isdigit() and int(number) == cid:
            return file, path
    raise ValueError(f"{path}: active calibration {cid} not found in calibrations/")


@functools.cache
def _toml_cached(path: Path, mtime: float) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from None


def _toml(path: Path) -> dict:
    return _toml_cached(path.resolve(), path.stat().st_mtime)


def _views(calibration: Path) -> dict[str, dict]:
    views = _toml(calibration).get("views")
    if not views:
        raise ValueError(f"{calibration}: no [views]")
    return views


def _view_of(manifest: Path, video) -> str | None:
    """The view whose videos include `video`, by path or else by file name."""
    target = Path(video).resolve()
    by_name = []
    for name, files in _toml(manifest).get("videos", {}).items():
        for file in [files] if isinstance(files, str) else files:
            path = (manifest.parent / file).resolve()
            if path == target:
                return name
            if path.name == target.name:
                by_name.append(name)
    return by_name[0] if len(by_name) == 1 else None


def _world_to_camera(spec: dict) -> np.ndarray:
    """The rotation `R` with `v_camera = R @ v_world`, for either form of a view."""
    if "rvec" in spec:
        return rotvec_to_matrix(np.asarray(spec["rvec"], dtype=np.float64))
    # An orbit placement, as deeperfly's `resolve_extrinsics` resolves it.
    az, el, roll = np.radians(
        [spec.get(k, 0.0) for k in ("azimuth_deg", "elevation_deg", "roll_deg")]
    )
    toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    z = -toward
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("an orbit view looking straight up or down has no roll")
    x /= np.linalg.norm(x)
    look = np.array([x, np.cross(z, x), z])
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ look


def _focal(spec: dict, calibration: Path, name: str) -> float:
    """The vertical focal length in pixels: `intr`'s `fy` (or `f`), or `focal`."""
    intr = spec.get("intr")
    if intr is not None:
        return float(intr[1] if len(intr) == 4 else intr[0])
    if "focal" in spec:
        return float(spec["focal"])
    raise ValueError(f"{calibration}: view {name} has no focal length")
