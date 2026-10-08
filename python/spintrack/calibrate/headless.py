"""Calibration steps that need no window: for lab servers and remote sessions.

`spintrack calibrate CONFIG --c2a-angles ELEV AZIM TWIST` writes the camera-to-animal
transform from the three angles `spintrack.calibrate.sliders` documents, and `--auto`
fits the ball (and `vfov`) from the recording itself. Both write the keys the GUI
writes, and both create CONFIG when it does not exist and `--src` names the video.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spintrack.autofit import prepare_config
from spintrack.calibrate.sliders import c2a_from_angles
from spintrack.config import Config
from spintrack.detect import DetectionError

log = logging.getLogger("spintrack")

AUTO_COMMENTS = ("# ball fitted by spintrack", "# vfov fitted by spintrack")


def open_config(path: str | Path, src=None) -> Config:
    """The config at `path`, or a new one for the video `src` when there is none."""
    path = Path(path)
    if path.exists():
        return Config.load(path)
    if src is None:
        raise ValueError(f"{path} does not exist; pass --src VIDEO to create it")
    log.info("creating %s for %s", path, src)
    return Config(src_fn=str(src), path=path)


def write_c2a_angles(
    config_path: str | Path, elevation: float, azimuth: float, twist: float, src=None
) -> Config:
    """Set `c2a_r` from the camera position angles and save the config in place."""
    cfg = open_config(config_path, src)
    cfg.c2a_r = c2a_from_angles(elevation, azimuth, twist)
    cfg.c2a_src = "sliders"
    cfg.extra["c2a_angles"] = [float(elevation), float(azimuth), float(twist)]
    path = cfg.save()
    log.info(
        "c2a_r : { %.6f, %.6f, %.6f }  (elevation %g, azimuth %g, twist %g) -> %s",
        *cfg.c2a_r, elevation, azimuth, twist, path,
    )  # fmt: skip
    return cfg


def write_auto_geometry(config_path: str | Path, src=None, n_frames: int = 100) -> int:
    """Fit the ball (and, when `vfov` is `auto`, the field of view) and save the config.

    Returns a process exit code: 2 when nothing trustworthy could be measured, so a
    scripted setup fails loudly instead of tracking against a guessed ball.
    """
    try:
        cfg = open_config(config_path, src)
        prepared = prepare_config(cfg, cfg.source(src), n_frames=n_frames)
    except (DetectionError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    # One note per fit, replacing the previous run's rather than piling up.
    cfg.comments = [c for c in cfg.comments if not c.startswith(AUTO_COMMENTS)]
    detection = prepared.detection
    if detection is not None:
        cfg.comments.append(
            f"{AUTO_COMMENTS[0]} calibrate --auto from {detection.n_frames} frames "
            f"(confidence {detection.confidence:.2f}, rim {detection.rim_fraction:.2f}"
            f", residual {detection.residual_px:.2f} px)"
        )
    if prepared.vfov is not None:
        cfg.comments.append(
            f"{AUTO_COMMENTS[1]} calibrate --auto: {prepared.vfov.line()}"
        )
    path = cfg.save()
    log.info("ball: %s", prepared.line())
    if prepared.vfov is not None:
        log.info("vfov: %s", prepared.vfov.line())
    for note in prepared.notes:
        log.info("%s", note)
    log.info("wrote %s", path)
    return 0
