"""Scene specifications and dataset generation (video + ground truth + FicTrac
config).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.camera import source_camera
from spintrack.config import Config
from spintrack.geometry import matrix_to_rotvec
from spintrack.sphere import ball_outline
from spintrack_bench.synth.motion import (
    MotionSpec,
    lab_to_camera_increments,
    orientations,
)
from spintrack_bench.synth.render import (
    LightingSpec,
    OccluderSpec,
    Renderer,
    SensorSpec,
)
from spintrack_bench.synth.texture import TEXTURE_PRESETS, BlobTextureSpec

GENERATOR_VERSION = 1


@dataclass
class SceneSpec:
    name: str = "scene"
    seed: int = 0
    width: int = 640
    height: int = 480
    vfov_deg: float = 30.0
    fisheye: bool = False
    ball_azimuth_deg: float = (
        3.0  # ball center direction in the camera frame (+ = right)
    )
    ball_elevation_deg: float = -2.0  # (+ = up in the image)
    half_angle_deg: float = 11.0  # angular radius of the ball
    texture: str | BlobTextureSpec = "blobs"
    lighting: LightingSpec = field(default_factory=LightingSpec)
    sensor: SensorSpec = field(default_factory=SensorSpec)
    occluders: OccluderSpec = field(default_factory=OccluderSpec)
    motion: MotionSpec = field(default_factory=MotionSpec)
    # Optional movement of the ball itself in its holder, as
    # `{"kind": "bump", "start": int, "end": int, "amplitude_radii": float,
    #   "direction_deg": float}`: between `start` and `end` the center slides
    # `amplitude_radii` ball radii along `direction_deg` (90 = down the image) and comes
    # back, following a raised cosine. The animal does not move with it.
    ball_path: dict | None = None
    n_frames: int = 1000
    fps: float = 100.0
    codec: str = "h264"  # h264 | hevc | lossless
    crf: int = 18
    # Camera placement relative to the animal (see spintrack.calibrate.sliders).
    c2a_elevation_deg: float = 30.0
    c2a_azimuth_deg: float = 0.0
    c2a_twist_deg: float = 0.0
    # FicTrac config parameters written alongside the scene.
    q_factor: int = 6
    thr_ratio: float = 1.25
    thr_win_pc: float = 0.25

    def __post_init__(self):
        if isinstance(self.texture, str):
            self.texture = dataclasses.replace(TEXTURE_PRESETS[self.texture])
        for name, cls in (
            ("lighting", LightingSpec),
            ("sensor", SensorSpec),
            ("occluders", OccluderSpec),
            ("motion", MotionSpec),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))
        if isinstance(self.texture, dict):
            self.texture = BlobTextureSpec(**self.texture)
        self.motion.fps = self.fps

    def ball_center(self) -> np.ndarray:
        az, el = np.radians([self.ball_azimuth_deg, self.ball_elevation_deg])
        c = np.array([np.sin(az) * np.cos(el), -np.sin(el), np.cos(az) * np.cos(el)])
        return c / np.linalg.norm(c)

    def cam_to_lab(self) -> np.ndarray:
        return camera_to_lab_from_angles(
            self.c2a_elevation_deg, self.c2a_azimuth_deg, self.c2a_twist_deg
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["generator_version"] = GENERATOR_VERSION
        return json.dumps(d, indent=2, default=_json_default)

    @classmethod
    def from_json(cls, text: str) -> SceneSpec:
        d = json.loads(text)
        d.pop("generator_version", None)
        return cls(**d)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


class FfmpegWriter:
    """Encode gray uint8 frames with ffmpeg (libx264, libx265 or lossless x264)."""

    def __init__(
        self, path: Path, width: int, height: int, fps: float, codec: str, crf: int
    ):
        if codec == "h264":
            enc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
        elif codec == "hevc":
            enc = [
                "-c:v",
                "libx265",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-tag:v",
                "hvc1",
            ]
        elif codec == "lossless":
            enc = ["-c:v", "libx264", "-preset", "medium", "-qp", "0"]
        else:
            raise ValueError(f"unknown codec {codec!r}")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            *enc, "-pix_fmt", "yuv420p", str(path),
        ]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the synthetic video")


def build_renderer(spec: SceneSpec, rng: np.random.Generator) -> Renderer:
    camera = source_camera(spec.width, spec.height, spec.vfov_deg, spec.fisheye)
    texture = spec.texture.build(rng)
    return Renderer(
        camera,
        spec.ball_center(),
        np.radians(spec.half_angle_deg),
        texture,
        spec.lighting,
        spec.sensor,
        spec.occluders,
        spec.fps,
        rng,
    )


def fictrac_config(spec: SceneSpec, renderer: Renderer, video_name: str) -> Config:
    cfg = Config()
    cfg.src_fn = video_name
    cfg.vfov = float(spec.vfov_deg)
    cfg.fisheye = bool(spec.fisheye)
    cfg.src_fps = float(spec.fps)
    cfg.q_factor = int(spec.q_factor)
    cfg.thr_ratio = float(spec.thr_ratio)
    cfg.thr_win_pc = float(spec.thr_win_pc)
    cfg.do_display = False
    center = spec.ball_center()
    half = float(np.radians(spec.half_angle_deg))
    cfg.roi_c = [float(v) for v in center]
    cfg.roi_r = half
    rim = ball_outline(renderer.camera, center, half, n_points=8)
    cfg.roi_circ = [round(v) for v in rim.ravel()]
    body = renderer.body_polygon()
    cfg.roi_ignr = [body] if body else []
    cfg.c2a_r = [float(v) for v in matrix_to_rotvec(spec.cam_to_lab())]
    cfg.c2a_t = [0.0, 0.0, 1.0]
    return cfg


def center_path(spec: SceneSpec, renderer: Renderer) -> np.ndarray:
    """Per-frame ball center direction (n_frames, 3); constant unless `ball_path` is
    set.
    """
    center = spec.ball_center()
    path = np.repeat(center[None, :], spec.n_frames, axis=0)
    if not spec.ball_path:
        return path
    kind = spec.ball_path.get("kind", "bump")
    angle = np.radians(float(spec.ball_path.get("direction_deg", 90.0)))
    frames = np.arange(spec.n_frames)
    offset = np.zeros(spec.n_frames)
    moved = np.zeros(spec.n_frames, dtype=bool)
    if kind == "bump":
        start = int(spec.ball_path["start"])
        end = int(spec.ball_path["end"])
        amplitude = float(spec.ball_path["amplitude_radii"]) * renderer.radius_px
        inside = (frames >= start) & (frames < end)
        phase = 2.0 * np.pi * (frames[inside] - start) / max(end - start, 1)
        offset[inside] = amplitude * 0.5 * (1.0 - np.cos(phase))
        moved = inside
    elif kind == "steps":
        # `steps` is a list of `[start, frames, amplitude_radii]`: each moves the ball
        # by its amplitude along `direction_deg` over its frames with a raised-cosine
        # velocity profile, and the moves add up. A few short ones make the jerky drop
        # of AN07B017_260414_Fly4_004 (about a tenth of a radius in ten frames).
        for start, length, amplitude_radii in spec.ball_path["steps"]:
            amplitude = float(amplitude_radii) * renderer.radius_px
            phase = np.clip((frames - start) / max(int(length), 1), 0.0, 1.0)
            offset += amplitude * 0.5 * (1.0 - np.cos(np.pi * phase))
        moved = offset != 0.0
    else:
        raise ValueError(f"unknown ball_path kind {kind!r}")
    if not moved.any():
        return path
    cx, cy = renderer.center_px
    shift = offset[moved]
    rays = renderer.camera.rays(cx + shift * np.cos(angle), cy + shift * np.sin(angle))
    path[moved] = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return path


def generate(spec: SceneSpec, out_dir: Path, progress=None) -> Path:
    """Render the scene into `out_dir` (video.mp4, truth.npz, config.txt,
    scene.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    renderer = build_renderer(spec, rng)
    w_lab = spec.motion.generate(spec.n_frames, rng)
    w_lab[0] = 0.0  # frame 0 is the reference: no increment, as trackers report it
    cam_to_lab = spec.cam_to_lab()
    w_cam = lab_to_camera_increments(w_lab, cam_to_lab)
    R = orientations(w_cam)

    video = out_dir / "video.mp4"
    writer = FfmpegWriter(
        video, spec.width, spec.height, spec.fps, spec.codec, spec.crf
    )
    path = center_path(spec, renderer)
    R_prev = np.eye(3)
    for i in range(spec.n_frames):
        if i > 0 and not np.array_equal(path[i], path[i - 1]):
            renderer.set_center(path[i])
        frame = renderer.render(R_prev, R[i], i, spec.n_frames)
        writer.write(frame)
        R_prev = R[i]
        if i == 0:
            import cv2

            cv2.imwrite(str(out_dir / "preview.png"), frame)
        if progress is not None and i % 100 == 0:
            progress(i, spec.n_frames)
    writer.close()

    ts_ms = np.arange(spec.n_frames) / spec.fps * 1e3
    np.savez_compressed(
        out_dir / "truth.npz",
        R=R,
        w_cam=w_cam,
        w_lab=w_lab,
        ts_ms=ts_ms,
        cam_to_lab=cam_to_lab,
        center=spec.ball_center(),
        center_path=path,
        half_angle=np.radians(spec.half_angle_deg),
        fps=spec.fps,
    )
    fictrac_config(spec, renderer, video.name).save(out_dir / "config.txt")
    (out_dir / "scene.json").write_text(spec.to_json())
    return out_dir


def load_truth(dataset_dir: Path) -> dict:
    with np.load(Path(dataset_dir) / "truth.npz") as z:
        return {k: z[k] for k in z.files}
