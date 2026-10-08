"""The tracking config: a TOML file, validated by a pydantic model.

    video = "trial3.mp4"

    [camera]
    vfov_deg = 2.3893
    position_deg = [0, 180, 0]  # elevation, azimuth, twist

    [ball]
    rim = [[656.5, 411.0], [142.5, 411.5], [145.0, 321.0], [255.0, 156.0]]

Every key is optional: a run finds the ball, the field of view and the camera position
in the recording when the config leaves them out. `docs/guide.md` lists the keys. Paths
are relative to the file. An unknown key is an error, so that a typo cannot pass for a
default. On the command line, `KEY=VALUE` arguments (`tracking.window_px=80`) override
the config's keys.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tomllib
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.geometry import rotvec_to_matrix

# A command-line override: a dotted key, `=`, and a value.
OVERRIDE = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)=(.*)", re.DOTALL)
# The types of a key that takes only text, which an override passes verbatim.
TEXT = {str, type(None)}
# Keys that stand in for others: an override of one clears them, unless also set.
REPLACES = {
    "camera.position_deg": ("camera.rotation",),
    "camera.rotation": ("camera.position_deg",),
    "camera.calibration": ("camera.position_deg", "camera.rotation", "camera.vfov_deg"),
    "camera.azimuth_deg": ("camera.position_deg", "camera.rotation"),
}
# The keys that hold paths, as (table, key); None is the top level.
PATHS = (
    (None, "video"),
    ("camera", "calibration"),
    ("tracking", "initial_map"),
    ("tracking", "initial_illumination"),
)


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CameraConfig(_Table):
    """The lens, and where the camera sits around the animal."""

    # Vertical field of view; None until `spintrack calibrate --auto` fits it.
    vfov_deg: float | None = Field(None, gt=0)
    fisheye: bool = False  # an equidistant (f-theta) lens rather than a pinhole
    fps: float | None = Field(None, gt=0)  # for a source that does not report one
    # The camera-to-animal transform: the camera's elevation, azimuth and twist
    # (`spintrack.calibrate.sliders`), or a rotation vector, as the calibration square
    # gives. One of the two, or neither until it is known.
    position_deg: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None
    # A deeperfly calibration (a calibration file, a manifest or a project folder) and
    # its view that filmed the video, found from the manifest's videos when not given.
    # It gives the field of view and the camera position the keys above leave out.
    calibration: str | None = None
    view: str | None = None
    # Where the camera sits around the animal when nothing above says: 0 in front, 90
    # at its right, 180 behind, -90 at its left. The elevation and the twist then come
    # from where the animal stands on the ball.
    azimuth_deg: float | None = Field(None, ge=-360.0, le=360.0)

    @model_validator(mode="after")
    def _one_transform(self) -> Self:
        if self.position_deg is not None and self.rotation is not None:
            raise ValueError("position_deg and rotation are alternatives; keep one")
        return self

    def to_animal(self) -> np.ndarray | None:
        """The rotation `R` with `v_animal = R @ v_camera`, or None when not set."""
        if self.position_deg is not None:
            return camera_to_lab_from_angles(*self.position_deg)
        if self.rotation is not None:
            return rotvec_to_matrix(np.asarray(self.rotation, dtype=np.float64))
        return None


class BallConfig(_Table):
    """The ball's rim: points on its outline in the image, in pixels."""

    rim: list[tuple[float, float]] = Field(default_factory=list)

    @field_validator("rim")
    @classmethod
    def _enough(cls, rim):
        if 0 < len(rim) < 3:
            raise ValueError("at least three rim points are needed")
        return rim


class MaskConfig(_Table):
    """Image regions the tracker ignores: the animal, the tether, the holder's edge."""

    ignore: list[list[tuple[int, int]]] = Field(default_factory=list)  # polygons, px

    @field_validator("ignore")
    @classmethod
    def _polygons(cls, ignore):
        if any(len(polygon) < 3 for polygon in ignore):
            raise ValueError("a polygon needs at least three points")
        return ignore


class TrackingConfig(_Table):
    window_px: int = Field(60, gt=0)  # side of the square tracking window
    global_search: bool = False  # relocalize against the whole map when lost
    # Lost frames in a row before tracking restarts; None: never.
    max_bad_frames: int | None = Field(None, ge=0)
    max_step_rad: float = Field(0.35, gt=0)  # the largest rotation accepted per frame
    # Side of the local brightness normalization, as a fraction of the window.
    norm_window: float = Field(0.25, gt=0)
    forget_outside_view: bool = False  # forget map cells the window does not see
    illumination: bool = True  # separate the rig's static lighting from the texture
    # Start from a saved map (a spintrack .npz or a FicTrac sphere-map .png).
    initial_map: str | None = None
    freeze_map: bool = False  # never update `initial_map`
    # Start the lighting fields from a map .npz saved on this rig, without its ball.
    initial_illumination: str | None = None


class OutputConfig(_Table):
    name: str | None = None  # the output folder is NAME_spintrack; default the video's
    debug_video: bool = False  # also write debug.mp4
    debug_axes: bool = False  # draw the ball's axes in debug.mp4
    debug_codec: str = "h264"


class StreamConfig(_Table):
    """Where to stream the records live, in FicTrac's line format."""

    udp: str | None = Field(None, pattern=r"^.*:\d+$")  # HOST:PORT
    serial: str | None = Field(None, pattern=r"^[^:]+(:\d+)?$")  # PORT[:BAUD]


class Config(_Table):
    """A rig and, through `video`, a recording of it."""

    video: str | int | None = None  # a path, or a camera index
    camera: CameraConfig = Field(default_factory=CameraConfig)
    ball: BallConfig = Field(default_factory=BallConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Read a `.toml` config, resolving its relative paths against its folder."""
        path = Path(path)
        if path.suffix.lower() != ".toml":
            raise ValueError(
                f"{path}: configs are TOML files; docs/fictrac.md translates a FicTrac "
                f"config.txt"
            )
        try:
            cfg = cls.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: {exc}") from None
        except ValidationError as exc:
            raise ValueError(_explain(path, exc)) from None
        for table, key in PATHS:
            owner = getattr(cfg, table) if table else cfg
            value = getattr(owner, key)
            if isinstance(value, str):
                setattr(owner, key, os.path.normpath(path.parent / value))
        return cfg

    def save(
        self, path: str | Path, comments: Sequence[str] = (), *, full: bool = False
    ) -> Path:
        """Write the config to `path` as TOML, under `comments` (lines from `#`).

        Only keys that differ from the defaults are written unless `full`. Paths are
        written relative to the file, unless the two share no folder but the root.
        """
        path = Path(path)
        data = self.model_dump(exclude_defaults=not full, exclude_none=True)
        for table, key in PATHS:
            value = getattr(getattr(self, table) if table else self, key)
            if isinstance(value, str):
                (data[table] if table else data)[key] = _relative(value, path.parent)
        header = "".join(f"{c}\n" for c in comments) + ("\n" if comments else "")
        path.write_text(header + _toml(data), "utf-8")
        return path


def is_override(arg: str) -> bool:
    """Whether a command-line argument is a `KEY=VALUE` override rather than a path."""
    return OVERRIDE.fullmatch(arg) is not None


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """A copy of `cfg` with `KEY=VALUE` overrides applied, validated as a whole.

    A value is read as TOML (`80`, `true`, `[0, 180, 0]`); a list may drop its brackets
    (`0,180,0`), `none` restores a key's default, and a key that takes only text takes
    the value verbatim. Relative paths are relative to the current directory.
    """
    changes = {}
    for item in overrides:
        match = OVERRIDE.fullmatch(item)
        if match is None:
            raise ValueError(f"{item!r} is not KEY=VALUE")
        key, raw = match.groups()
        changes[key] = _override_value(key, raw)
    return with_changes(cfg, changes, "command line")


def with_changes(cfg: Config, changes: dict, where: str = "changes") -> Config:
    """A copy of `cfg` with values set by dotted key, validated as a whole.

    Setting `camera.position_deg` clears `camera.rotation`, and the other way around;
    setting `camera.calibration` clears both and `camera.vfov_deg` (`REPLACES`).
    Raises `ValueError`, naming `where` and each wrong key.
    """
    data = cfg.model_dump()
    for key in changes.keys() & REPLACES.keys():
        for other in REPLACES[key]:
            if other not in changes:
                table, name = other.split(".")
                data[table][name] = None
    for key, value in changes.items():
        *tables, name = key.split(".")
        owner = data
        for table in tables:
            if type(owner.get(table)) is not dict:
                owner[table] = {}  # not a table: validation names the key
            owner = owner[table]
        owner[name] = value
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_explain(where, exc)) from None


def _override_value(key: str, raw: str):
    """An override's value, typed by its key: see `apply_overrides`."""
    model, field = Config, None
    for part in key.split("."):
        field = getattr(model, "model_fields", {}).get(part)
        if field is None:
            break  # an unknown key: validation names it
        model = field.annotation
    if raw.strip().lower() in ("none", "null"):
        return None if field is None else field.get_default(call_default_factory=True)
    if field is not None and set(typing.get_args(model) or (model,)) <= TEXT:
        return raw
    for text in (raw, f"[{raw}]"):
        try:
            return tomllib.loads(f"v = {text}")["v"]
        except tomllib.TOMLDecodeError:
            pass
    return raw


def leading_comments(path: str | Path) -> list[str]:
    """The comment lines at the top of a config file, for `Config.save` to keep."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            break
        if line.startswith("#"):
            out.append(line)
    return out


def _relative(path: str, folder: Path) -> str:
    """`path` relative to `folder`, or absolute when the two share only the root."""
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    try:
        shared = os.path.commonpath([path, folder])
    except ValueError:  # on another drive
        shared = ""
    if shared == os.path.dirname(shared):
        return Path(path).as_posix()
    return Path(os.path.relpath(path, folder)).as_posix()


def _toml(data: dict) -> str:
    """TOML for a dict of values and tables of values.

    An array of arrays gets one item per line, any other array one line (tomli-w would
    give every number of a rim point a line of its own).
    """
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items() if type(v) is not dict]
    for name, table in data.items():
        if type(table) is dict and table:
            lines += ["", f"[{name}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in table.items()]
    return "\n".join(lines).lstrip("\n") + "\n"


def _toml_value(value, nested: bool = False) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a TOML basic string, except that TOML escapes DEL too.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    items = [_toml_value(v, True) for v in value]
    if not nested and value and isinstance(value[0], (list, tuple)):
        return "[\n" + "".join(f"    {item},\n" for item in items) + "]"
    return "[" + ", ".join(items) + "]"


def _explain(path: Path | str, exc: ValidationError) -> str:
    """The validation errors, one per line, with a suggestion for unknown keys."""
    tables = {
        name: field.annotation.model_fields
        for name, field in Config.model_fields.items()
        if isinstance(field.annotation, type) and issubclass(field.annotation, _Table)
    }
    known = {name: name for name in Config.model_fields}
    for name, keys in tables.items():
        known.update({key: f"{name}.{key}" for key in keys})
    lines = [f"{path}: invalid config"]
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        if error["type"] != "extra_forbidden":
            lines.append(f"  {where}: {error['msg'].removeprefix('Value error, ')}")
            continue
        *table, key = error["loc"]
        close = difflib.get_close_matches(str(key), known, n=1)
        if close:
            hint = f"did you mean {known[close[0]]}?"
        elif table:
            hint = f"[{table[0]}] takes {', '.join(tables[table[0]])}"
        else:
            hint = f"the top level takes {', '.join(Config.model_fields)}"
        lines.append(f"  {where}: unknown key; {hint}")
    return "\n".join(lines)
