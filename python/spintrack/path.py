"""Fictive path integration from per-frame ball rotations in the lab frame.

Lab frame (FicTrac convention): x forward, y right, z down. A ball rotation vector
`dr_lab` maps to animal motion as: `forward = dr_lab[1]`, `side = -dr_lab[0]` (positive
to the animal's right), `heading -= dr_lab[2]`. Positions are in radians of ball
rotation (multiply by the ball radius for distance). The world frame has x along the
initial heading and y to the initial right, matching FicTrac's output columns 15-21.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class PathStep:
    forward: float
    side: float
    step_mag: float
    step_dir: float  # direction of motion in the animal frame, [0, 2 pi)
    heading: float  # integrated heading, [0, 2 pi)
    pos_x: float
    pos_y: float
    int_x: float  # integrated forward motion, ignoring heading
    int_y: float  # integrated side motion, ignoring heading


class PathIntegrator:
    """Accumulate lab-frame rotation vectors into heading and 2-D position."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.heading = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.int_x = 0.0
        self.int_y = 0.0

    def step(self, dr_lab) -> PathStep:
        dr = np.asarray(dr_lab, dtype=np.float64)
        forward = float(dr[1])
        side = -float(dr[0])
        turn = -float(dr[2])
        step_mag = float(np.hypot(forward, side))
        step_dir = float(np.arctan2(side, forward)) % TWO_PI

        # Exact integral of a step whose world direction turns uniformly by `turn`.
        start = self.heading + step_dir
        if abs(turn) > 1e-12:
            mid = start + 0.5 * turn
            gain = np.sin(0.5 * turn) / (0.5 * turn)
        else:
            mid, gain = start, 1.0
        self.pos_x += step_mag * gain * np.cos(mid)
        self.pos_y += step_mag * gain * np.sin(mid)

        self.heading = (self.heading + turn) % TWO_PI
        self.int_x += forward
        self.int_y += side
        return PathStep(
            forward,
            side,
            step_mag,
            step_dir,
            self.heading,
            self.pos_x,
            self.pos_y,
            self.int_x,
            self.int_y,
        )


def integrate_path(dr_lab: np.ndarray) -> np.ndarray:
    """Vectorized: (N, 3) lab rotation vectors -> (N, 5) x, y, heading, int_x, int_y."""
    integ = PathIntegrator()
    out = np.empty((len(dr_lab), 5))
    for i, dr in enumerate(dr_lab):
        s = integ.step(dr)
        out[i] = (s.pos_x, s.pos_y, s.heading, s.int_x, s.int_y)
    return out
