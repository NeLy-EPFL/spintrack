"""Ray-cast renderer for a textured ball seen by a calibrated camera.

Only the ball's bounding box is ray-cast (supersampled); the rest of the frame is a static
background. Shading (Lambert + specular) depends on the surface normal, which is fixed per
pixel, so only the albedo lookup changes with the ball orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from spintrack.camera import Camera
from spintrack.geometry import matrix_to_rotvec, normalize, rotvec_to_matrix
from spintrack.sphere import ball_outline
from spintrack_bench.synth.texture import Texture


@dataclass
class LightingSpec:
    light_dir: tuple[float, float, float] = (-0.3, -0.7, -0.65)  # toward the light
    ambient: float = 0.35
    diffuse: float = 0.65
    specular: float = 0.15
    shininess: float = 25.0
    vignette: float = 0.25  # relative darkening at the image corners
    flicker_amp: float = 0.0  # fractional gain modulation
    flicker_hz: float = 100.0
    drift_amp: float = 0.0  # slow fractional gain drift over the clip
    # Shadow cast by the holder the ball sits in: the surface is progressively occluded
    # from the lamps below `holder_elevation` (as a fraction of the ball radius, measured
    # down the image from the center), losing up to `holder_shadow` of its illumination
    # over `holder_softness` radii. Fixed in the camera frame, like the rest of the
    # shading, so it darkens the texture that rotates past it rather than moving with it.
    holder_shadow: float = 0.0
    holder_elevation: float = 0.45
    holder_softness: float = 0.25
    holder_ambient: float = 0.0  # stray light in the shadow: kills contrast, not level


@dataclass
class SensorSpec:
    blur_sigma: float = 0.7  # px, defocus
    read_noise: float = 2.0  # gray levels
    shot_noise: float = 5.0  # gray levels at full scale
    exposure: float = 0.0  # fraction of the frame period integrated (motion blur)
    exposure_steps: int = 4
    supersample: int = 3
    background: float = 55.0  # gray level
    background_noise: float = 3.0  # static pattern std, gray levels


@dataclass
class OccluderSpec:
    legs: int = 0  # number of leg-like moving occluders (0 = none)
    leg_thickness_px: float = 0.035  # relative to the ball's image radius
    gait_hz: float = 12.0
    body: bool = False  # static body silhouette above the ball (exported as roi_ignr)
    dust: int = 0  # static translucent spots on the lens
    extra: dict = field(default_factory=dict)


class Renderer:
    def __init__(
        self,
        camera: Camera,
        center,
        half_angle: float,
        texture: Texture,
        lighting: LightingSpec,
        sensor: SensorSpec,
        occluders: OccluderSpec,
        fps: float,
        rng: np.random.Generator,
    ):
        self.camera = camera
        self.texture = texture
        self.lighting = lighting
        self.sensor = sensor
        self.occluders = occluders
        self.fps = fps
        self.rng = rng
        h, w = camera.height, camera.width

        self.set_center(center, half_angle)
        # The animal is tethered above the ball, so it stays where it was even if the
        # ball moves: its anchor and size are frozen at the first frame's geometry.
        self.body_anchor_px = self.center_px
        self.body_radius_px = self.radius_px

        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        self.vignette = (1.0 - lighting.vignette * r2).astype(np.float32)
        bg_rng = np.random.default_rng(rng.integers(0, 2**31))
        self.background = (
            sensor.background + bg_rng.normal(0.0, sensor.background_noise, (h, w))
        ).astype(np.float32)
        self.dust_gain = np.ones((h, w), np.float32)
        x0, y0, x1, y1 = self.bbox
        for _ in range(occluders.dust):
            dx, dy, rad = (
                bg_rng.uniform(x0, x1),
                bg_rng.uniform(y0, y1),
                bg_rng.uniform(4, 12),
            )
            spot = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * rad**2))
            self.dust_gain *= (1.0 - 0.5 * spot).astype(np.float32)

    def set_center(self, center, half_angle: float | None = None) -> None:
        """Point the ball at `center`, rebuilding everything that depends on where it is.

        Costs about 0.1 s at 3x supersampling, so scenes call it only on the frames where
        the ball has actually moved.
        """
        camera, sensor = self.camera, self.sensor
        h, w = camera.height, camera.width
        self.center = normalize(np.asarray(center, dtype=np.float64))
        if half_angle is not None:
            self.half_angle = float(half_angle)

        outline = ball_outline(camera, self.center, self.half_angle, 90)
        margin = int(np.ceil(4 * sensor.blur_sigma + 2))
        x0 = max(0, int(np.floor(outline[:, 0].min())) - margin)
        y0 = max(0, int(np.floor(outline[:, 1].min())) - margin)
        x1 = min(w, int(np.ceil(outline[:, 0].max())) + margin)
        y1 = min(h, int(np.ceil(outline[:, 1].max())) + margin)
        self.bbox = (x0, y0, x1, y1)
        cx, cy, _ = camera.project(self.center)
        self.center_px = (float(cx), float(cy))
        self.radius_px = float(np.max(np.hypot(outline[:, 0] - cx, outline[:, 1] - cy)))

        ss = sensor.supersample
        xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss
        ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss
        X, Y = np.meshgrid(xs, ys)
        rays = camera.rays(X, Y)
        radius = np.sin(self.half_angle)
        b = rays @ self.center
        disc = b * b - (1.0 - radius * radius)
        self.hit = disc >= 0.0
        t = b - np.sqrt(np.where(self.hit, disc, 0.0))
        points = t[..., None] * rays
        normals = normalize(points - self.center)
        normals[~self.hit] = 0.0
        self.normals = normals.astype(np.float32)

        light = normalize(np.asarray(self.lighting.light_dir, dtype=np.float64))
        ndotl = np.clip(normals @ light, 0.0, None)
        view = -rays
        reflect = 2.0 * ndotl[..., None] * normals - light
        spec = (
            np.clip(np.sum(reflect * view, axis=-1), 0.0, None)
            ** self.lighting.shininess
        )
        self.shade = (self.lighting.ambient + self.lighting.diffuse * ndotl) * self.hit
        lt = self.lighting
        if lt.holder_shadow > 0.0 or lt.holder_ambient > 0.0:
            ramp = self._holder_ramp(normals)
            self.shade = self.shade * (1.0 - lt.holder_shadow * ramp)
            self.glow = (lt.holder_ambient * ramp * self.hit).astype(np.float32)
        else:
            self.glow = np.zeros(self.hit.shape, dtype=np.float32)
        self.spec = (self.lighting.specular * spec * self.hit).astype(np.float32)

    def _holder_ramp(self, normals: np.ndarray) -> np.ndarray:
        """0 above the holder, rising smoothly to 1 in the shadow under the ball.

        The normal's image-down component is the height on the ball, so this is a
        horizontal band fixed in the camera frame: the texture rotates through it.
        """
        lt = self.lighting
        c = np.asarray(self.center, dtype=np.float64)
        down = normalize(np.cross(c, np.cross([0.0, 1.0, 0.0], c)))
        h = normals @ down
        t = (h - lt.holder_elevation) / max(lt.holder_softness, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # ----- ball appearance -----
    def albedo(self, R: np.ndarray) -> np.ndarray:
        """Albedo of every supersample for ball orientation `R` (body -> camera)."""
        body_dirs = self.normals @ R.astype(np.float32)  # R^T applied row-wise
        alb = self.texture.sample(body_dirs)
        alb[~self.hit] = 0.0
        return alb

    def gain(self, t_s: float, duration_s: float) -> float:
        lt = self.lighting
        g = 1.0 + lt.flicker_amp * np.sin(2 * np.pi * lt.flicker_hz * t_s)
        if duration_s > 0:
            g += lt.drift_amp * np.sin(2 * np.pi * t_s / duration_s)
        return float(g)

    def render(
        self, R_prev: np.ndarray, R: np.ndarray, index: int, n_frames: int
    ) -> np.ndarray:
        """uint8 frame for orientation `R`, with motion blur from `R_prev` if configured."""
        sensor = self.sensor
        if sensor.exposure > 0 and sensor.exposure_steps > 1:
            w = matrix_to_rotvec(R @ R_prev.T)
            taus = (
                1.0
                - sensor.exposure
                * (np.arange(sensor.exposure_steps) + 0.5)
                / sensor.exposure_steps
            )
            alb = np.mean(
                [self.albedo(rotvec_to_matrix(tau * w) @ R_prev) for tau in taus], 0
            )
        else:
            alb = self.albedo(R)
        patch = alb * self.shade + self.spec + self.glow
        ss = sensor.supersample
        hb, wb = patch.shape[0] // ss, patch.shape[1] // ss
        patch = patch.reshape(hb, ss, wb, ss).mean(axis=(1, 3))
        hit = self.hit.reshape(hb, ss, wb, ss).mean(axis=(1, 3))

        img = self.background.copy()
        x0, y0, x1, y1 = self.bbox
        region = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = region * (1.0 - hit) + 255.0 * patch
        t_s = index / self.fps
        self._draw_occluders(img, t_s)
        img *= self.vignette * self.gain(t_s, n_frames / self.fps)
        if sensor.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sensor.blur_sigma)
        noise_std = np.sqrt(
            sensor.read_noise**2 + sensor.shot_noise**2 * np.clip(img, 0, 255) / 255
        )
        img += self.rng.normal(0.0, 1.0, img.shape).astype(np.float32) * noise_std
        return np.clip(img, 0, 255).astype(np.uint8)

    # ----- occluders -----
    def body_polygon(self) -> list[int] | None:
        """FicTrac-style flat polygon of the body silhouette, or None."""
        if not self.occluders.body:
            return None
        cx, cy = self._body_center()
        axes = (int(0.38 * self.body_radius_px), int(0.22 * self.body_radius_px))
        pts = cv2.ellipse2Poly((int(cx), int(cy)), axes, 0, 0, 360, 20)
        return [int(v) for v in pts.ravel()]

    def _body_center(self) -> tuple[float, float]:
        cx, cy = self.body_anchor_px
        return cx, cy - 0.78 * self.body_radius_px

    def _draw_occluders(self, img: np.ndarray, t_s: float) -> None:
        occ = self.occluders
        cx, cy = self.body_anchor_px
        r = self.body_radius_px
        if occ.body:
            bx, by = self._body_center()
            axes = (int(0.38 * r), int(0.22 * r))
            cv2.ellipse(img, (int(bx), int(by)), axes, 0, 0, 360, 28.0, -1, cv2.LINE_AA)
        if occ.legs > 0:
            bx, by = self._body_center()
            thick = max(1, round(occ.leg_thickness_px * r))
            for k in range(occ.legs):
                side = -1.0 if k % 2 == 0 else 1.0
                rank = k // 2
                phase = (
                    np.pi * (k % 2) + 2 * np.pi * rank / 3.0
                )  # alternating tripod-ish
                swing = np.sin(2 * np.pi * occ.gait_hz * t_s + phase)
                ang = np.radians(-90 + side * (35 + 25 * rank) + 8 * swing)
                reach = r * (0.55 + 0.12 * rank + 0.06 * swing)
                foot = (cx + reach * np.cos(ang), cy - reach * np.sin(ang) * -1.0)
                knee = (
                    0.5 * (bx + foot[0]) + side * 0.18 * r,
                    0.5 * (by + foot[1]) - 0.12 * r,
                )
                pts = np.array([[bx, by], knee, foot], np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, 22.0, thick, cv2.LINE_AA)
        if self.occluders.dust:
            img *= self.dust_gain
