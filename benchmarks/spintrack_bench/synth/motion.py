"""Ground-truth ball motion: per-frame rotation increments in the lab frame.

Lab frame: x forward, y right, z down. Rotation about y is forward walking, about x is
sidestepping, about z is turning (see `spintrack.path`). All rates are rad/frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spintrack.geometry import rotvec_to_matrix


def ou_process(rng, n: int, sigma: float, tau_frames: float) -> np.ndarray:
    """Zero-mean Ornstein-Uhlenbeck series with stationary std `sigma`."""
    if tau_frames <= 0:
        return rng.normal(0.0, sigma, n)
    a = np.exp(-1.0 / tau_frames)
    noise = rng.normal(0.0, sigma * np.sqrt(1 - a * a), n)
    out = np.empty(n)
    x = rng.normal(0.0, sigma)
    for i in range(n):
        x = a * x + noise[i]
        out[i] = x
    return out


@dataclass
class MotionSpec:
    kind: str = "fly_walk"  # static | constant | random_walk | fly_walk | saccades
    axis: str = "y"  # for constant: lab axis x, y or z
    rate: float = 0.02  # for constant: rad/frame
    sigma: float = 0.02  # random_walk: stationary std per axis (rad/frame)
    tau_frames: float = 20.0  # random_walk correlation time
    fps: float = 100.0
    ball_radius_mm: float = 5.0
    speed_mm_s: tuple[float, float] = (4.0, 18.0)  # fly_walk forward speed range
    turn_sigma_deg_s: float = 120.0  # fly_walk slow turning std
    saccade_rate_hz: float = 0.7  # fly_walk/saccades: saccades per walking second
    saccade_peak_deg_s: tuple[float, float] = (400.0, 1200.0)
    saccade_ms: tuple[float, float] = (50.0, 100.0)
    bout_s: tuple[float, float] = (1.0, 5.0)
    stop_s: tuple[float, float] = (0.5, 3.0)
    extra: dict = field(default_factory=dict)

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, 3) lab-frame rotation increments, rad/frame."""
        if self.kind == "static":
            return np.zeros((n, 3))
        if self.kind == "constant":
            w = np.zeros((n, 3))
            w[:, "xyz".index(self.axis)] = self.rate
            return w
        if self.kind == "random_walk":
            return np.stack(
                [ou_process(rng, n, self.sigma, self.tau_frames) for _ in range(3)], 1
            )
        if self.kind in ("fly_walk", "saccades"):
            return self._fly_walk(n, rng, saccades_only=self.kind == "saccades")
        raise ValueError(f"unknown motion kind {self.kind!r}")

    def _fly_walk(self, n: int, rng, saccades_only: bool) -> np.ndarray:
        fps = self.fps
        rad_per_mm = 1.0 / self.ball_radius_mm
        w = np.zeros((n, 3))
        walking = np.zeros(n, bool)
        i = 0
        state = True  # start walking
        while i < n:
            dur = rng.uniform(*(self.bout_s if state else self.stop_s))
            j = min(n, i + round(dur * fps))
            walking[i:j] = state
            i = j
            state = not state
        # Forward speed: smooth modulation around a per-bout mean.
        speed = np.zeros(n)
        start = 0
        while start < n:
            end = start
            while end < n and walking[end] == walking[start]:
                end += 1
            if walking[start]:
                mean = rng.uniform(*self.speed_mm_s)
                speed[start:end] = mean * (
                    1 + 0.25 * ou_process(rng, end - start, 1.0, 15.0)
                )
            start = end
        speed = np.clip(speed, 0.0, None)
        forward = speed * rad_per_mm / fps  # rad/frame
        side = ou_process(rng, n, 0.15, 10.0) * forward
        turn = (
            np.radians(ou_process(rng, n, self.turn_sigma_deg_s, 12.0)) / fps * walking
        )
        # Saccades: brief bursts with a raised-cosine profile.
        n_sacc = rng.poisson(self.saccade_rate_hz * walking.sum() / fps)
        starts = np.flatnonzero(walking)
        for s in rng.choice(starts, size=min(n_sacc, len(starts)), replace=False):
            dur = round(rng.uniform(*self.saccade_ms) / 1000.0 * fps)
            peak = np.radians(rng.uniform(*self.saccade_peak_deg_s)) / fps
            sign = rng.choice([-1.0, 1.0])
            prof = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(dur) + 0.5) / dur))
            e = min(n, s + dur)
            turn[s:e] += sign * peak * prof[: e - s]
        if saccades_only:
            forward = forward * 0.3
        # Small jitter while standing (tethered flies never hold the ball perfectly still).
        jitter = rng.normal(0.0, 0.0005, (n, 3)) * (~walking)[:, None]
        # Lab-frame increments: x = -side (right is negative x rotation), y = forward,
        # z = -turn (heading increases for negative z), per spintrack.path conventions.
        w[:, 0] = -side
        w[:, 1] = forward
        w[:, 2] = -turn
        return w + jitter


def lab_to_camera_increments(w_lab: np.ndarray, cam_to_lab: np.ndarray) -> np.ndarray:
    """Rotate lab-frame rotation vectors into the camera frame (`v_lab = R @ v_cam`)."""
    return w_lab @ cam_to_lab  # equals (R^T @ w) row-wise


def orientations(w_cam: np.ndarray) -> np.ndarray:
    """Absolute ball orientations `R_t = exp(w_t) R_{t-1}` starting from identity, (n,3,3)."""
    out = np.empty((len(w_cam), 3, 3))
    R = np.eye(3)
    for i, w in enumerate(w_cam):
        R = rotvec_to_matrix(w) @ R
        out[i] = R
    return out
