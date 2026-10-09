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
SUBSTEPS = 4  # as FicTrac; more does not help


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

    A frame's step is walked in `SUBSTEPS` equal parts, each along the heading at its
    middle, while the heading turns uniformly from the last frame's value to this one's.
    This is FicTrac's scheme, so the path agrees with FicTrac's columns, mirrored, to
    rounding. The parts sum in closed form: a step `(forward, side)` with a turn `t`
    moves the animal by that step rotated to the frame's middle heading and scaled by
    the mean of `cos(p * t)` over the parts' offsets `p` from that middle.
    """

    def __init__(self) -> None:
        self.int_x = 0.0
        self.int_y = 0.0
        self.reset()

    def reset(self) -> None:
        """Restart heading and position; `int_x` and `int_y` carry on, as in FicTrac."""
        self.heading = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

    def step(self, dr_lab) -> PathStep:
        """Advance by one frame's lab-frame rotation vector."""
        forward = -float(dr_lab[1])
        side = float(dr_lab[0])
        self.int_x += forward
        self.int_y += side
        heading = _wrap(self.heading - float(dr_lab[2]))
        turn = math.remainder(heading - self.heading, TWO_PI)
        middle = self.heading + turn / 2.0
        gain = math.fsum(math.cos(p * turn) for p in _OFFSETS) / SUBSTEPS
        c, s = math.cos(middle), math.sin(middle)
        self.pos_x += gain * (forward * c - side * s)
        self.pos_y += gain * (forward * s + side * c)
        self.heading = heading
        return PathStep(
            forward,
            side,
            math.hypot(forward, side),
            _wrap(math.atan2(side, forward)),
            heading,
            self.pos_x,
            self.pos_y,
            self.int_x,
            self.int_y,
        )


# Each part's middle, as a fraction of the frame's turn, from the frame's middle.
_OFFSETS = tuple((k + 0.5) / SUBSTEPS - 0.5 for k in range(SUBSTEPS))


def _wrap(angle: float) -> float:
    """`angle` in [0, 2 pi)."""
    angle %= TWO_PI
    return 0.0 if angle >= TWO_PI else angle


def integrate_path(dr_lab: np.ndarray) -> np.ndarray:
    """Vectorized: (N, 3) lab rotation vectors -> (N, 5) x, y, heading, int_x, int_y."""
    integ = PathIntegrator()
    out = np.empty((len(dr_lab), 5))
    for i, dr in enumerate(dr_lab):
        s = integ.step(dr)
        out[i] = (s.pos_x, s.pos_y, s.heading, s.int_x, s.int_y)
    return out
