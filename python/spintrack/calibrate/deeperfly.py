"""A camera from a deeperfly calibration: its field of view and where it sits.

deeperfly (multi-camera pose estimation) keeps a project's calibrations under
`calibrations/NNNN-<label>.toml`, each a `[views.<name>]` table per camera: solved
(`rvec`, `tvec` world to camera as OpenCV has them, `intr` `[fx, fy, cx, cy]`,
`image_size` `[height, width]`) or an orbit placement (`azimuth_deg`, `elevation_deg`,
`roll_deg`, `focal`). Its manifest, `deeperfly.toml`, names the active one and each
view's videos. deeperfly's world is the rig's version of the animal frame here: x
forward (its front camera sits at azimuth 0), y left, z up.

The animal is never mounted exactly along the rig's x axis (on the lab's octacam, up to
12 deg off), so when the project holds pose results (`results/*.h5`), the animal's
thorax-coxa points, triangulated with the calibration, give its heading, and the camera
is placed relative to the animal's body rather than to the rig.
"""

from __future__ import annotations

import functools
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spintrack.calibrate.sliders import angles_from_camera_to_lab
from spintrack.geometry import rotvec_to_matrix

MANIFEST = "deeperfly.toml"
ASPECT_TOL = 0.01  # how far a video's aspect ratio may be from the calibration's
# The thorax-coxa points whose front and hind midpoints give the body's long axis.
FRONT = ("lf_thorax_coxa", "rf_thorax_coxa")
HIND = ("lh_thorax_coxa", "rh_thorax_coxa")
MIN_CONFIDENCE = 0.5  # a detection weaker than this is not triangulated
MAX_POSE_FRAMES = 400  # frames sampled for the heading; the animal is tethered


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

    The front and hind thorax-coxa midpoints, triangulated over up to
    `MAX_POSE_FRAMES` frames with `views` and their median taken, give the body's long
    axis; its horizontal part is the heading. None when there are no such results.
    """
    results = _pose_results(project)
    if results is None:
        return None
    import h5py

    with h5py.File(results) as f:
        names = json.loads(f.attrs["keypoints"])
        result_views = json.loads(f.attrs["views"])
        frame_sizes = json.loads(f.attrs.get("frame_sizes", "{}"))
        n = f["pose2d/points"].shape[2]
        frames = np.arange(0, n, max(1, n // MAX_POSE_FRAMES))
        index = [names.index(k) for k in FRONT + HIND]
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
    rows = []  # per view: (frames, 4 points, 2 rows, 4)
    for i in used:
        spec = by_name[result_views[i].lower()]
        K, dist, P = _camera(spec, frame_sizes.get(result_views[i]))
        flat = points[i].reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(flat, K, dist).reshape(points[i].shape)
        ok = (confidence[i] >= MIN_CONFIDENCE) & np.isfinite(normalized).all(axis=-1)
        x, y = normalized[..., :1], normalized[..., 1:]
        view_rows = np.stack([x * P[2] - P[0], y * P[2] - P[1]], axis=-2)
        rows.append(np.where(ok[..., None, None], view_rows, 0.0))
    A = np.concatenate(rows, axis=-2)  # (frames, 4, 2 * views, 4)
    seen = (np.abs(A).sum(axis=-1) > 0).sum(axis=-1) >= 4  # two views or more
    X = np.linalg.svd(A)[2][..., -1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(seen[..., None], X[..., :3] / X[..., 3:], np.nan)
    axis = 0.5 * (X[:, 0] + X[:, 1]) - 0.5 * (X[:, 2] + X[:, 3])  # front - hind
    axis = np.nanmedian(axis, axis=0)
    if not np.isfinite(axis[:2]).all() or np.hypot(*axis[:2]) == 0:
        return None
    return float(np.degrees(np.arctan2(axis[1], axis[0]))), results


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
