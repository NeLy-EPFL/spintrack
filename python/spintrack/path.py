"""Fictive path integration from per-frame ball rotations in the lab frame.

Lab frame: x forward, y left, z up. The ball turns under the animal, so a ball rotation
vector `dr_lab` maps to animal motion as `forward = -dr_lab[1]`, `side = dr_lab[0]`
(positive to the animal's left) and `heading -= dr_lab[2]` (counterclockwise seen from
above, so positive turning left). Positions are in radians of ball rotation (multiply
by the ball radius for distance), with x along the initial heading and y to its left.

FicTrac's frame is x forward, y right, z down, and its columns 15-21 the mirror image of
these: `spintrack.io.records.to_fictrac` converts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi
SUBSTEPS = 4  # FicTrac's; more does not help


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
    """Accumulate lab-frame rotation vectors into heading and 2-D position.

    A line-for-line port of FicTrac's `Trackball::updatePath` in this lab frame, so that
    the path columns are FicTrac's mirrored, to print precision, given the same
    rotations: each step is walked in four substeps along a direction turning with the
    heading.
    """

    def __init__(self) -> None:
        self.int_x = 0.0
        self.int_y = 0.0
        self.reset()

    def reset(self) -> None:
        """Restart heading and position; `int_x` and `int_y` carry on, as in FicTrac."""
        self.heading = 0.0
        self.prev_heading = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

    def step(self, dr_lab) -> PathStep:
        forward = -float(dr_lab[1])
        side = float(dr_lab[0])
        step_mag = math.sqrt(forward * forward + side * side)
        step_dir = math.atan2(side, forward)
        if step_dir < 0:
            step_dir += TWO_PI
        self.int_x += forward
        self.int_y += side
        heading = self.heading - float(dr_lab[2])
        while heading < 0:
            heading += TWO_PI
        while heading >= TWO_PI:
            heading -= TWO_PI
        self.heading = heading
        # Walk the step in four substeps, turning from the previous heading to this one.
        step = step_mag / SUBSTEPS
        turn = heading - self.prev_heading
        while turn >= math.pi:
            turn -= TWO_PI
        while turn < -math.pi:
            turn += TWO_PI
        turn /= SUBSTEPS
        dx, dy = forward, side
        if step_mag != 0:
            inv = 1.0 / step_mag
            dx, dy = dx * inv, dy * inv
        dx, dy = _rotate(dx, dy, self.prev_heading + turn / 2.0)
        c, s = math.cos(turn), math.sin(turn)
        for _ in range(SUBSTEPS):
            self.pos_x += step * dx
            self.pos_y += step * dy
            dx, dy = dx * c - dy * s, dx * s + dy * c
        self.prev_heading = heading
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


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c


def integrate_path(dr_lab: np.ndarray) -> np.ndarray:
    """Vectorized: (N, 3) lab rotation vectors -> (N, 5) x, y, heading, int_x, int_y."""
    integ = PathIntegrator()
    out = np.empty((len(dr_lab), 5))
    for i, dr in enumerate(dr_lab):
        s = integ.step(dr)
        out[i] = (s.pos_x, s.pos_y, s.heading, s.int_x, s.int_y)
    return out
