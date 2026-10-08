"""Calibration steps that need no window: for lab servers and remote sessions.

`spintrack calibrate CONFIG --camera-position ELEV AZIM TWIST` writes where the camera
sits, in the angles `spintrack.calibrate.sliders` documents, and `--auto` fits the ball
(and the field of view) from the recording itself. Both create CONFIG when it does not
exist and `--src` names the video.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spintrack.autofit import prepare_config
from spintrack.config import Config, leading_comments
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
    if path.suffix.lower() != ".toml":
        raise ValueError(f"{path}: a config is a .toml file")
    log.info("creating %s for %s", path, src)
    return Config(video=int(src) if str(src).isdigit() else str(src))


def save_config(cfg: Config, path: str | Path, notes: list[str] | None = None) -> Path:
    """Write `cfg` to `path`, keeping the comments at the top of the file there.

    `notes`, from `calibrate --auto`, replace the ones an earlier fit left.
    """
    path = Path(path)
    comments = leading_comments(path) if path.exists() else []
    if notes is not None:
        comments = [c for c in comments if not c.startswith(AUTO_COMMENTS)] + notes
    return cfg.save(path, comments)


def write_camera_position(
    config_path: str | Path, elevation: float, azimuth: float, twist: float, src=None
) -> Config:
    """Set the camera position (degrees) and save the config in place."""
    cfg = open_config(config_path, src)
    cfg.camera.rotation = None
    cfg.camera.position_deg = (elevation, azimuth, twist)
    path = save_config(cfg, config_path)
    log.info(
        "camera position: elevation %g, azimuth %g, twist %g -> %s",
        elevation, azimuth, twist, path,
    )  # fmt: skip
    return cfg


def write_auto_geometry(config_path: str | Path, src=None, n_frames: int = 100) -> int:
    """Fit the ball (and the field of view, when unknown) and save the config.

    Returns a process exit code: 2 when nothing trustworthy could be measured, so a
    scripted setup fails loudly instead of tracking against a guessed ball.
    """
    try:
        cfg = open_config(config_path, src)
        spec = src if src is not None else cfg.video
        if spec is None:
            raise ValueError(f"{config_path} names no video; pass --src VIDEO")
        prepared = prepare_config(cfg, str(spec), n_frames=n_frames)
    except (DetectionError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    notes = []
    detection = prepared.detection
    if detection is not None:
        notes.append(
            f"{AUTO_COMMENTS[0]} calibrate --auto from {detection.n_frames} frames "
            f"(confidence {detection.confidence:.2f}, rim {detection.rim_fraction:.2f}"
            f", residual {detection.residual_px:.2f} px)"
        )
    if prepared.vfov is not None:
        notes.append(f"{AUTO_COMMENTS[1]} calibrate --auto: {prepared.vfov.line()}")
    path = save_config(cfg, config_path, notes)
    log.info("ball: %s", prepared.line())
    if prepared.vfov is not None:
        log.info("vfov: %s", prepared.vfov.line())
    for note in prepared.notes:
        log.info("%s", note)
    log.info("wrote %s", path)
    return 0
