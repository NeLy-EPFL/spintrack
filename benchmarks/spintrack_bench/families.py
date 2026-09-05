"""Named synthetic scene families used for the benchmark."""

from __future__ import annotations

import dataclasses

from spintrack_bench.synth.dataset import SceneSpec

_LAB_SENSOR = {"blur_sigma": 0.8, "read_noise": 2.0, "shot_noise": 5.0}


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
    # JSP-like geometry: small ball in a narrow-FOV frame, HEVC, q_factor 12.
    "lab_small_ball": _fly(
        "lab_small_ball", width=864, height=512, vfov_deg=2.0, half_angle_deg=0.31,
        ball_azimuth_deg=-0.4, ball_elevation_deg=0.03, codec="hevc", crf=18, q_factor=12,
        thr_ratio=0.7, sensor=_LAB_SENSOR, occluders={"legs": 6, "body": True},
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
