"""Separate the static illumination of the ball from its rotating surface texture.

The ball sits under fixed lamps and in a holder, so its shading is fixed in the camera
frame while the texture rotates with the ball. Two fields per window pixel describe it,
both fixed in the window:

- an additive bias, the running mean of what the map cannot explain,
  `obs - gain * map(R^T v)`, which the observation has subtracted;
- a gain, how much texture contrast that pixel delivers: in a shadow the texture is
  dimmer and the local normalization does not fully restore it. It is the running RMS of
  the bias-corrected observation relative to the best-lit part of the window, which
  needs no map because the ball's texture averages out as it turns. The observation is
  divided by it, and each pixel enters the map with weight `gain**2`, so dim views,
  whose noise the division amplifies, cannot wash out what the well-lit ones mapped.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6
TAU = 500.0  # frames; memory of the field and of its accumulators
UPDATE_EVERY = 25  # frames between field refreshes
# Fold in every n-th accepted frame: the field changes over hundreds of frames, so
# sampling costs it nothing and keeps the estimator off the hot path.
STRIDE = 4
WARMUP = 100  # frames before the first refresh
# Forget the field when a window re-fit moves it this far, as a fraction of the ball's
# radius: past that the ball has moved enough to be lit differently.
RESET_MOVE = 0.05
# The gain field: smoothed over this many window pixels so that the texture passing a
# pixel while the ball turns little does not enter it, referred to this percentile of
# the window, and clamped.
GAIN_SIGMA = 6.0
GAIN_REFERENCE = 90.0
GAIN_MIN = 0.2
GAIN_MAX = 1.5


class Photometry:
    """Static camera-frame bias field, estimated against the accumulated map."""

    def __init__(self, mask: np.ndarray, enabled: bool = True):
        self.mask = mask
        self.enabled = enabled
        shape = mask.shape
        self.bias = np.zeros(shape, dtype=np.float64)
        self.gain = np.ones(shape, dtype=np.float64)
        self.acc_n = np.zeros(shape)
        self.acc_res = np.zeros(shape)
        self.acc_e2 = np.zeros(shape)
        # A field loaded from another run and the accumulator weight it carries (see
        # `load`).
        self.prior_bias = np.zeros(shape)
        self.prior_n = np.zeros(shape)
        self.frames = 0
        self.seen_frames = 0

    def correct(self, obs: np.ndarray) -> np.ndarray:
        """Subtract the static bias from a normalized window and divide by the gain."""
        if not self.enabled:
            return obs
        return np.ascontiguousarray((obs - self.bias) / self.gain, dtype=np.float32)

    def weights(self) -> np.ndarray | None:
        """Per-pixel weight of a corrected window in the map, None if uniform."""
        if not self.enabled:
            return None
        return np.ascontiguousarray(self.gain**2, dtype=np.float32)

    def observe(self, core, R: np.ndarray, obs: np.ndarray) -> None:
        """Fold one accepted frame in, before it is splatted into the map.

        `obs` is the corrected window the solve used, so the residual is measured
        against the model the tracker actually believes.
        """
        self.seen_frames += 1
        if not self.enabled or self.seen_frames % STRIDE:
            return
        value, conf = core.render(R)
        seen = (conf > 0.0) & self.mask
        if not seen.any():
            return
        m = np.where(seen, value.astype(np.float64), 0.0)
        # Undo the gain, so both fields are estimated on the bias-corrected window.
        o = np.where(seen, obs.astype(np.float64) * self.gain, 0.0)
        # `TAU` is in frames, so a strided sample stands for `STRIDE` of them.
        decay = np.exp(-STRIDE / TAU)
        self.acc_n = decay * self.acc_n + seen
        self.prior_n = decay * self.prior_n
        # The residual of the *uncorrected* window, so that the accumulator is the field
        # itself and a refresh reads it off; stepping the field by the corrected
        # residual would make a damped oscillator of it.
        self.acc_res = decay * self.acc_res + (o - self.gain * m + self.bias) * seen
        self.acc_e2 = decay * self.acc_e2 + o * o
        self.frames += STRIDE
        if self.frames >= WARMUP and self.frames % UPDATE_EVERY < STRIDE:
            self.refresh()

    def refresh(self) -> None:
        """Read the field off the accumulators, a loaded prior included.

        The mask mean is projected out because the field and the map are separable only
        up to an additive constant. Pixels neither seen since the last reset nor carried
        by a prior keep their value.
        """
        mask = self.mask
        if not mask.any():
            return
        total = self.acc_n + self.prior_n
        ok = mask & (total > 1.0)
        field = np.where(
            ok,
            (self.acc_res + self.prior_bias * self.prior_n) / np.maximum(total, EPS),
            0.0,
        )
        field = np.where(ok, field - float(field[ok].mean()), 0.0)
        self.bias = np.where(ok, field, np.where(mask, self.bias, 0.0))
        seen = self.acc_n > 1.0
        if seen.sum() < 100:
            return
        rms = np.where(seen, self.acc_e2 / np.maximum(self.acc_n, EPS), 0.0)
        smooth = cv2.GaussianBlur(rms, (0, 0), GAIN_SIGMA)
        norm = cv2.GaussianBlur(seen.astype(np.float64), (0, 0), GAIN_SIGMA)
        rms = np.sqrt(np.where(seen, smooth / np.maximum(norm, EPS), 0.0))
        reference = float(np.percentile(rms[seen], GAIN_REFERENCE))
        if reference <= EPS:
            return
        gain = np.clip(rms / reference, GAIN_MIN, GAIN_MAX)
        self.gain = np.where(seen, gain, np.where(mask, self.gain, 1.0))

    def resample(
        self, remap_x: np.ndarray, remap_y: np.ndarray, mask: np.ndarray
    ) -> None:
        """Carry the field to a new window: it is window-fixed, not ball-fixed."""

        def warp(field):
            out = cv2.remap(
                field.astype(np.float32), remap_x, remap_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )  # fmt: skip
            return np.where(mask, out, 0.0).astype(np.float64)

        self.mask = mask
        for name in ("bias", "acc_n", "acc_res", "acc_e2", "prior_bias", "prior_n"):
            setattr(self, name, warp(getattr(self, name)))
        gain = warp(self.gain)
        self.gain = np.where(gain > 0.0, gain, 1.0)

    def forget(self) -> None:
        """Discard the field and start over, when what it describes has changed."""
        self.frames = 0
        for name in ("bias", "acc_n", "acc_res", "acc_e2", "prior_bias", "prior_n"):
            getattr(self, name)[:] = 0.0
        self.gain[:] = 1.0

    def report(self) -> dict | None:
        """Summary of what the lighting is doing, for the run sidecar."""
        if not self.enabled or not self.mask.any():
            return None
        return {
            "illum_peak": float(np.abs(self.bias[self.mask]).max()),
            "gain_min": float(self.gain[self.mask].min()),
        }

    def state(self) -> dict:
        return {
            "bias": self.bias.astype(np.float32),
            "gain": self.gain.astype(np.float32),
        }

    def load(self, bias=None, gain=None, prior_frames: float = 0.0) -> None:
        """Install a field measured elsewhere, `prior_frames` deep in the accumulator.

        Without that weight the first refresh would replace the field by a mean of the
        run's own first frames. With it the field is a prior, which new observations
        pull away from with the time constant `TAU`, and which keeps its value where a
        pixel is never seen.
        """
        if gain is not None:
            gain = np.asarray(gain, dtype=np.float64)
            self.gain = np.where(self.mask & (gain > 0.0), gain, 1.0)
        if bias is None:
            return
        self.bias = np.where(self.mask, np.asarray(bias, dtype=np.float64), 0.0)
        if prior_frames > 0.0:
            self.prior_bias = self.bias.copy()
            self.prior_n = np.where(self.mask, prior_frames / STRIDE, 0.0)
