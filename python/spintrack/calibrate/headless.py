"""Calibration steps that need no window: for lab servers and remote sessions.

`spintrack calibrate CONFIG --c2a-angles ELEV AZIM TWIST` writes the camera-to-animal
transform from the three angles `spintrack.calibrate.sliders` documents, and
`--auto` fits the ball (and `vfov`) from the recording itself. Both write the same keys
the GUI writes, so a config stays interchangeable between the two paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spintrack.autofit import prepare_config, resolve_source
from spintrack.calibrate.sliders import c2a_from_angles
from spintrack.config import Config
from spintrack.detect import DetectionError

log = logging.getLogger("spintrack")


def write_c2a_angles(
    config_path: str | Path, elevation: float, azimuth: float, twist: float
) -> Config:
    """Set `c2a_r` from the camera position angles and save the config in place."""
    path = Path(config_path)
    cfg = Config.load(path)
    cfg.c2a_r = c2a_from_angles(elevation, azimuth, twist)
    cfg.c2a_src = "sliders"
    cfg.extra["c2a_angles"] = [float(elevation), float(azimuth), float(twist)]
    cfg.save(path)
    log.info(
        "c2a_r : { %.6f, %.6f, %.6f }  (elevation %g, azimuth %g, twist %g) -> %s",
        *cfg.c2a_r,
        elevation,
        azimuth,
        twist,
        path,
    )
    return cfg


def write_auto_geometry(config_path: str | Path, src=None, n_frames: int = 100) -> int:
    """Fit the ball (and, when `vfov` is `auto`, the field of view) and save the config.

    Returns a process exit code: 2 when nothing trustworthy could be measured, so a
    scripted setup fails loudly instead of tracking against a guessed ball.
    """
    path = Path(config_path)
    cfg = Config.load(path)
    try:
        prepared = prepare_config(
            cfg, resolve_source(path, cfg, src), n_frames=n_frames
        )
    except (DetectionError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    detection = prepared.detection
    cfg.comments.append(
        f"# ball fitted by spintrack calibrate --auto from {detection.n_frames} frames "
        f"(confidence {detection.confidence:.2f}, rim {detection.rim_fraction:.2f}, "
        f"residual {detection.residual_px:.2f} px)"
    )
    if prepared.vfov is not None:
        cfg.comments.append(
            f"# vfov fitted by spintrack calibrate --auto: {prepared.vfov.line()}"
        )
    cfg.save(path)
    log.info("ball: %s", prepared.line())
    if prepared.vfov is not None:
        log.info("vfov: %s", prepared.vfov.line())
    for note in prepared.notes:
        log.info("%s", note)
    log.info("wrote %s", path)
    return 0
