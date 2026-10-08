"""Named synthetic scene families used for the benchmark."""

from __future__ import annotations

import dataclasses

from spintrack_bench.synth.dataset import SceneSpec

_LAB_SENSOR = {"blur_sigma": 0.8, "read_noise": 2.0, "shot_noise": 5.0}
# Matched to the lab recordings: the shadowed band is about 45% darker and keeps about
# half its relative contrast, the stray light being what flattens the rest.
_HOLDER = {"holder_shadow": 0.7, "holder_ambient": 0.08, "holder_elevation": 0.32,
           "holder_softness": 0.1}  # fmt: skip


_JERKS = [[300, 10, 0.10], [340, 8, 0.13], [372, 10, 0.12], [600, 300, -0.35]]


def _fly(name: str, **kw) -> SceneSpec:
    base = {"name": name, "motion": {"kind": "fly_walk"}}
    base.update(kw)
    return SceneSpec(**base)


FAMILIES: dict[str, SceneSpec] = {
    "clean_fly": _fly("clean_fly"),
    "static": SceneSpec(name="static", motion={"kind": "static"}),
    "constant_forward": SceneSpec(
        name="constant_forward", motion={"kind": "constant", "axis": "y", "rate": 0.02}
    ),
    "constant_side": SceneSpec(
        name="constant_side", motion={"kind": "constant", "axis": "x", "rate": 0.02}
    ),
    "constant_turn": SceneSpec(
        name="constant_turn", motion={"kind": "constant", "axis": "z", "rate": 0.02}
    ),
    "fast_forward": SceneSpec(
        name="fast_forward", motion={"kind": "constant", "axis": "y", "rate": 0.12}
    ),
    "random_walk": SceneSpec(
        name="random_walk", motion={"kind": "random_walk", "sigma": 0.03, "tau_frames": 15}
    ),
    "saccades": _fly("saccades", motion={"kind": "saccades"}),
    "low_contrast": _fly("low_contrast", texture="low_contrast"),
    "speckle": _fly("speckle", texture="speckle"),
    "sparse": _fly("sparse", texture="sparse"),
    "occluded": _fly("occluded", occluders={"legs": 6, "body": True, "dust": 3}),
    "lighting": _fly(
        "lighting",
        lighting={"flicker_amp": 0.15, "flicker_hz": 47.0, "drift_amp": 0.2, "vignette": 0.4,
                  "specular": 0.35},
    ),
    "motion_blur": _fly(
        "motion_blur", sensor={"exposure": 0.8, "exposure_steps": 4, "blur_sigma": 1.5}
    ),
    "noisy": _fly("noisy", sensor={"read_noise": 8.0, "shot_noise": 14.0, "blur_sigma": 1.0}),
    "offaxis": _fly("offaxis", ball_azimuth_deg=16.0, ball_elevation_deg=9.0, half_angle_deg=8.0),
    # The ball sinks in its holder and comes back, as in AN07B017_260414_Fly4_004.
    "ball_drop": _fly(
        "ball_drop", occluders={"legs": 6, "body": True},
        ball_path={"kind": "bump", "start": 300, "end": 840, "amplitude_radii": 0.4,
                   "direction_deg": 90.0},
    ),
    # The ball drops in jerks, as 004's actually does: three of about a tenth of the
    # radius over ten frames each, then back over 300 frames. `ball_drop` moves 0.002
    # radii a frame at most, which no follower lags; these move up to 0.026.
    "jerky_drop": _fly(
        "jerky_drop", occluders={"legs": 6, "body": True},
        ball_path={"kind": "steps", "steps": _JERKS, "direction_deg": 90.0},
    ),
    # Shorter and longer jerks, at an angle, on another texture.
    "jerky_diag": _fly(
        "jerky_diag", seed=7, occluders={"legs": 6, "body": True},
        ball_path={"kind": "steps", "direction_deg": 60.0,
                   "steps": [[250, 12, 0.08], [290, 6, 0.08], [330, 14, 0.15],
                             [650, 200, -0.31]]},
    ),
    # The jerky drop at the lab rig's own geometry: a 506 px ball at a 1.2 deg
    # half-angle in a 1600 x 1008 frame, which the window decimates 8:1.
    "lab_jerky_drop": _fly(
        "lab_jerky_drop", width=1600, height=1008, vfov_deg=2.3893, half_angle_deg=1.2,
        ball_azimuth_deg=-0.4, ball_elevation_deg=0.03, codec="hevc", crf=18,
        q_factor=12, thr_ratio=0.7, sensor=_LAB_SENSOR,
        occluders={"legs": 6, "body": True},
        ball_path={"kind": "steps", "steps": _JERKS, "direction_deg": 90.0},
    ),
    # The ball holder shadows the bottom of the ball: a darkening fixed in the camera
    # frame that the texture rotates through, as in AN07B017_260414_Fly4_003.
    "holder_shadow": _fly("holder_shadow", lighting=_HOLDER),
    # JSP-like geometry: small ball in a narrow-FOV frame, HEVC, q_factor 12.
    "lab_small_ball": _fly(
        "lab_small_ball", width=864, height=512, vfov_deg=2.0, half_angle_deg=0.31,
        ball_azimuth_deg=-0.4, ball_elevation_deg=0.03, codec="hevc", crf=18, q_factor=12,
        thr_ratio=0.7, sensor=_LAB_SENSOR, occluders={"legs": 6, "body": True},
    ),
    # The same rig with the holder shadow: the scene this whole correction is aimed at.
    "holder_shadow_lab": _fly(
        "holder_shadow_lab", width=864, height=512, vfov_deg=2.0, half_angle_deg=0.31,
        ball_azimuth_deg=-0.4, ball_elevation_deg=0.03, codec="hevc", crf=18, q_factor=12,
        thr_ratio=0.7, sensor=_LAB_SENSOR, occluders={"legs": 6, "body": True},
        lighting=_HOLDER,
    ),
    # As above but with the ball sitting low enough to be cut by the bottom of the frame,
    # as it is on the real rig: the shadow then falls right at the edge of the mask, where
    # the local normalization has no bright pixels left to balance it against.
    "holder_shadow_cut": _fly(
        "holder_shadow_cut", width=864, height=512, vfov_deg=2.0, half_angle_deg=0.31,
        ball_azimuth_deg=-0.4, ball_elevation_deg=-0.86, codec="hevc", crf=18, q_factor=12,
        thr_ratio=0.7, sensor=_LAB_SENSOR, occluders={"legs": 6, "body": True},
        lighting=_HOLDER,
    ),
}  # fmt: skip


def get_family(
    name: str, n_frames: int | None = None, seed: int | None = None
) -> SceneSpec:
    spec = dataclasses.replace(FAMILIES[name])
    if n_frames is not None:
        spec.n_frames = n_frames
    if seed is not None:
        spec.seed = seed
    return spec
