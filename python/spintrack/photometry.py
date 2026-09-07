"""Separate the static illumination of the ball from its rotating surface texture.

The ball is lit by fixed lamps and sits in a holder, so the shading at a given window
pixel is fixed in the *camera* frame while the texture it falls on rotates with the ball.
Anything camera-fixed that is left in the observation is therefore not texture, and
splatting it into the ball-fixed map smears it over the surface: on the lab recordings the
band just above the holder sits about 0.9 low in normalized-intensity units (map values
span roughly +/-3) and keeps only a third of the texture contrast of the rest of the ball.

Three per-window-pixel fields describe what the lighting does to a pixel, all held in the
window frame and all inert at their defaults:

    residual = (obs - bias) - gain * map(R^T v)      weighted by  rho * map_conf * wt
    splat      value (obs - bias) / gain             footprint  *= wt * gain^2

`bias` is additive (a shadow floor, a stray reflection), `gain` multiplicative (shading
scales the texture contrast), and `wt` is the inverse-noise-variance weight that follows
from them, so a pixel whose texture the shadow has buried counts for less instead of
injecting noise.

The estimator has to go through the map rather than averaging frames directly. A per-pixel
temporal mean of the frames themselves only separates lighting from texture if the ball
turns enough for the texture to average away, and it does not: on the lab recordings a
window pixel sees about three effectively independent patches of surface over two thousand
frames, so a plain temporal mean is mostly texture and subtracting it would take real
signal out of the map. Taking the mean of `obs - map(R^T v)` instead removes the texture
explicitly, and needs no spatial smoothing to be usable.

`flat` is a fourth, independent correction that acts earlier, on the raw window before
`TrackEngine.normalize` sees it. The local box z-score in `normalize` is itself part of
the problem: its box straddles the sharp edge of the holder shadow, which is what turns a
smooth darkening into the overshoot-and-undershoot pattern the map inherits. Dividing the
raw window by a smoothed temporal flat field removes the edge before the box filter meets
it. It is the one arm that can repair contrast the normalizer destroyed, and the one whose
estimate is exposed to the texture leakage above, hence the spatial smoothing.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def _smooth(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores pixels outside the mask."""
    if sigma <= 0:
        return field
    k = max(3, int(6 * sigma) | 1)
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(field.astype(np.float32) * m, (k, k), sigma)
    den = cv2.GaussianBlur(m, (k, k), sigma)
    return np.where(mask, num / np.maximum(den, EPS), 0.0).astype(np.float64)


class FlatField:
    """Temporal flat field of the raw window, applied before local normalization."""

    def __init__(self, mask: np.ndarray, tau: float, sigma: float, warmup: int):
        self.mask = mask
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.warmup = int(warmup)
        shape = mask.shape
        self.n = np.zeros(shape)
        self.s1 = np.zeros(shape)
        self.s2 = np.zeros(shape)
        self.frames = 0
        self.frozen = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def apply(self, window: np.ndarray) -> np.ndarray:
        """Flatten `window`, then fold it into the running statistics."""
        img = window.astype(np.float64)
        out = window
        if self._mean is not None:
            flat = (img - self._mean) / self._std * self._std_ref + self._mean_ref
            out = np.where(self.mask, flat, img)
        decay = np.exp(-1.0 / self.tau)
        self.n = decay * self.n + self.mask
        self.s1 = decay * self.s1 + img * self.mask
        self.s2 = decay * self.s2 + img * img * self.mask
        self.frames += 1
        if not self.frozen and self.frames >= self.warmup and self.frames % 25 == 0:
            self._refresh()
        return out

    def _refresh(self) -> None:
        n = np.maximum(self.n, EPS)
        mean = self.s1 / n
        var = np.maximum(self.s2 / n - mean * mean, 0.0)
        mean = _smooth(mean, self.mask, self.sigma)
        std = np.sqrt(_smooth(var, self.mask, self.sigma))
        # A pixel with no measurable temporal contrast carries no texture; the floor keeps
        # the division from turning its noise into signal.
        floor = 0.2 * float(np.median(std[self.mask])) if self.mask.any() else 1.0
        self._mean = mean
        self._std = np.maximum(std, max(floor, EPS))
        self._mean_ref = float(np.median(mean[self.mask]))
        self._std_ref = float(np.median(self._std[self.mask]))


class Photometry:
    """Static camera-frame photometric fields, estimated against the accumulated map."""

    def __init__(self, mask: np.ndarray, params):
        self.mask = mask
        self.p = params
        shape = mask.shape
        self.bias = np.zeros(shape, dtype=np.float64)
        self.gain = np.ones(shape, dtype=np.float64)
        self.wt = np.ones(shape, dtype=np.float64)
        self.acc_n = np.zeros(shape)
        self.acc_res = np.zeros(shape)
        self.acc_res2 = np.zeros(shape)
        self.acc_om = np.zeros(shape)
        self.acc_mm = np.zeros(shape)
        self.frames = 0
        self.seen_frames = 0
        self.frozen = False
        self.flat = (
            FlatField(mask, params.illum_flat_tau, params.illum_flat_sigma,
                      params.illum_warmup)
            if params.illum_flat
            else None
        )  # fmt: skip

    @property
    def active(self) -> bool:
        p = self.p
        return bool(p.illum_bias or p.illum_gain or p.illum_weight or p.illum_measure)

    # ----- application -----
    def correct(self, obs: np.ndarray) -> np.ndarray:
        """Subtract the static bias from a normalized window."""
        if not self.p.illum_bias:
            return obs
        return np.ascontiguousarray(obs - self.bias, dtype=np.float32)

    def push(self, core) -> None:
        """Hand the multiplicative fields to the solver and the map update."""
        if not self.p.illum_gain and not self.p.illum_weight:
            return
        gain = self.gain if self.p.illum_gain else None
        wt = self.wt if self.p.illum_weight else None
        core.set_photometric(
            None if gain is None else np.ascontiguousarray(gain, dtype=np.float32),
            None if wt is None else np.ascontiguousarray(wt, dtype=np.float32),
        )

    # ----- estimation -----
    def observe(self, core, R: np.ndarray, obs: np.ndarray) -> None:
        """Fold one accepted frame in, before it is splatted into the map.

        `obs` is the corrected window the solve used, so the residual is measured against
        the model the tracker actually believes.
        """
        p = self.p
        self.seen_frames += 1
        stride = max(1, int(p.illum_stride))
        if not self.active or self.seen_frames % stride:
            return
        value, conf = core.render(R, p.w_min, p.w_sat)
        seen = (conf > 0.0) & self.mask
        if not seen.any():
            return
        m = np.where(seen, value.astype(np.float64), 0.0)
        o = np.where(seen, obs.astype(np.float64), 0.0)
        res = o - self.gain * m
        # `illum_tau` is in frames, so a strided sample stands for `stride` of them.
        decay = np.exp(-stride / p.illum_tau)
        self.acc_n = decay * self.acc_n + seen
        # The residual of the *uncorrected* window, so that the accumulator is the bias
        # field itself (an exponential mean of what the map cannot explain) and a refresh
        # reads it off. Stepping the field by a fraction of the corrected residual instead
        # made a damped oscillator of it: 16% overshoot, 63% of the way only 450 frames
        # after the warmup, against the first refresh here.
        self.acc_res = decay * self.acc_res + (res + self.bias) * seen
        if p.illum_weight:
            self.acc_res2 = decay * self.acc_res2 + res * res * seen
        if p.illum_gain:
            self.acc_om = decay * self.acc_om + o * m
            self.acc_mm = decay * self.acc_mm + m * m
        self.frames += stride
        if (
            self.frames >= p.illum_warmup
            and self.frames % p.illum_update_every < stride
        ):
            self.refresh(core)

    def residual_field(self) -> np.ndarray:
        """Mean of `obs - model` per window pixel: what is left that does not rotate.

        Zero once the lighting has been separated out; a shadow the map has absorbed
        shows up here as a static patch. Needs no ground truth, so it reads the same on
        synthetic scenes and on recordings.
        """
        ok = self.mask & (self.acc_n > 1.0)
        field = np.where(
            ok, self.acc_res / np.maximum(self.acc_n, EPS) - self.bias, 0.0
        )
        return np.where(ok, field - (field[ok].mean() if ok.any() else 0.0), 0.0)

    def refresh(self, core=None) -> None:
        """Recompute the fields from the accumulators."""
        p = self.p
        mask = self.mask
        if not mask.any():
            return
        if self.frozen or not (p.illum_bias or p.illum_gain or p.illum_weight):
            return  # measure-only: the accumulators are the whole point
        n = np.maximum(self.acc_n, EPS)
        seen = self.acc_n > 1.0
        ok = mask & seen
        if p.illum_bias:
            # The accumulator is an exponential mean of the uncorrected residual with the
            # time constant `illum_tau`, so the field is read off it. The mask mean is
            # projected out because the field and the map are only separable up to one
            # additive constant; pixels not seen since the last reset keep their value.
            field = np.where(ok, self.acc_res / n, 0.0)
            field = _smooth(field, ok, p.illum_smooth)
            field = np.where(ok, field - float(field[ok].mean()), 0.0)
            self.bias = np.where(ok, field, np.where(mask, self.bias, 0.0))
        if p.illum_gain:
            # Regress the observation on the model. Only the shape of the field is
            # identifiable - the overall scale trades off against the map's amplitude -
            # so it is renormalized, damped and clamped every time.
            g_hat = np.where(ok, self.acc_om / np.maximum(self.acc_mm, EPS), 1.0)
            g_hat = _smooth(g_hat, ok, p.illum_smooth) if p.illum_smooth > 0 else g_hat
            g_hat = np.where(ok, g_hat, 1.0)
            a = p.illum_gain_damping
            gain = (1.0 - a) * self.gain + a * g_hat
            ref = float(np.median(gain[ok])) if ok.any() else 1.0
            gain = np.clip(gain / max(ref, EPS), p.illum_gain_min, p.illum_gain_max)
            self.gain = np.where(mask, gain, 1.0)
        if p.illum_weight:
            # Inverse noise variance: the Fisher information a pixel carries about the
            # rotation is (gain * texture gradient)^2 / residual variance.
            mean = self.acc_res / n
            var = np.maximum(self.acc_res2 / n - mean * mean, EPS)
            info = np.where(ok, self.gain * self.gain / var, 0.0)
            info = _smooth(info, ok, p.illum_smooth) if p.illum_smooth > 0 else info
            ref = float(np.median(info[ok])) if ok.any() else 1.0
            wt = np.clip(info / max(ref, EPS), p.illum_weight_min, p.illum_weight_max)
            self.wt = np.where(ok, wt, 1.0)
        if core is not None:
            self.push(core)

    # ----- window moves -----
    def resample(
        self, remap_x: np.ndarray, remap_y: np.ndarray, mask: np.ndarray
    ) -> None:
        """Carry the fields to a new window: they are window-fixed, not ball-fixed."""

        def warp(field, fill):
            out = cv2.remap(
                field.astype(np.float32), remap_x, remap_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
            )  # fmt: skip
            return np.where(mask, out, fill).astype(np.float64)

        self.bias = warp(self.bias, 0.0)
        self.gain = warp(self.gain, 1.0)
        self.wt = warp(self.wt, 1.0)
        self.mask = mask
        for name, fill in (("acc_n", 0.0), ("acc_res", 0.0), ("acc_res2", 0.0),
                           ("acc_om", 0.0), ("acc_mm", 0.0)):  # fmt: skip
            setattr(self, name, warp(getattr(self, name), fill))
        if self.flat is not None:
            self.flat = FlatField(
                mask,
                self.p.illum_flat_tau,
                self.p.illum_flat_sigma,
                self.p.illum_warmup,
            )

    # ----- inspection -----
    def forget(self) -> None:
        """Discard the fields and start over, when what they describe has changed."""
        self.bias[:] = 0.0
        self.gain[:] = 1.0
        self.wt[:] = 1.0
        self.frames = 0
        for name in ("acc_n", "acc_res", "acc_res2", "acc_om", "acc_mm"):
            getattr(self, name)[:] = 0.0
        if self.flat is not None:
            self.flat = FlatField(
                self.mask,
                self.p.illum_flat_tau,
                self.p.illum_flat_sigma,
                self.p.illum_warmup,
            )

    def freeze(self) -> None:
        """Stop learning but keep applying what was learned, and restart the accumulators.

        Scoring the rest of a recording after this is the test of whether an arm separated
        lighting from texture: a field that has quietly absorbed texture cannot transfer to
        frames it never saw.
        """
        self.frozen = True
        if self.flat is not None:
            self.flat.frozen = True
        for name in ("acc_n", "acc_res", "acc_res2", "acc_om", "acc_mm"):
            getattr(self, name)[:] = 0.0

    def report(self) -> dict | None:
        """Summary of what the lighting is doing, for the run sidecar."""
        if not self.active or not self.mask.any():
            return None
        field = self.bias if self.p.illum_bias else self.residual_field()
        report = {"illum_peak": float(np.abs(field[self.mask]).max())}
        if self.p.illum_weight:
            report["illum_dim_frac"] = float((self.wt[self.mask] < 0.5).mean())
        return report

    def state(self) -> dict:
        return {"bias": self.bias.astype(np.float32),
                "gain": self.gain.astype(np.float32),
                "wt": self.wt.astype(np.float32)}  # fmt: skip

    def load(self, bias=None, gain=None, wt=None) -> None:
        if bias is not None:
            self.bias = np.where(self.mask, np.asarray(bias, dtype=np.float64), 0.0)
        if gain is not None:
            self.gain = np.where(self.mask, np.asarray(gain, dtype=np.float64), 1.0)
        if wt is not None:
            self.wt = np.where(self.mask, np.asarray(wt, dtype=np.float64), 1.0)
