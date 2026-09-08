"""Tracking configuration: FicTrac-compatible `config.txt`, YAML and TOML.

The FicTrac text format is `key : value` per line. Vectors are written `{ a, b, c }`,
nested vectors `{ { a, b }, { c, d } }`, booleans `y`/`n`. Lines starting with `##` are
headers and dropped; other lines starting with `#` or `%` are comments and preserved.
Unknown keys are kept in `Config.extra` and written back unchanged.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TRUE = {"y", "yes", "true", "1"}
_FALSE = {"n", "no", "false", "0"}
_AUTO = {"auto", "none", ""}


@dataclass
class Config:
    """FicTrac's documented parameters plus the lab fork's `accumulate_map`.

    Defaults follow FicTrac's documentation. `vfov` has no default: it must be set.
    """

    src_fn: str = ""
    vfov: float | None = None  # None or `auto` in the config: fitted from the recording
    fisheye: bool = False
    q_factor: int = 6
    src_fps: float = -1.0
    max_bad_frames: int = -1
    do_display: bool = True
    save_debug: bool = False
    save_raw: bool = False
    vid_codec: str = "h264"
    thr_ratio: float = 1.25
    thr_win_pc: float = 0.25
    opt_do_global: bool = False
    opt_max_err: float = -1.0
    opt_max_evals: int = 50
    opt_bound: float = 0.35
    opt_tol: float = 1e-3
    roi_circ: list[int] = field(default_factory=list)
    roi_c: list[float] | None = None
    roi_r: float | None = None
    roi_ignr: list[list[int]] = field(default_factory=list)
    c2a_src: str = ""
    c2a_cnrs_xy: list[int] = field(default_factory=list)
    c2a_cnrs_yz: list[int] = field(default_factory=list)
    c2a_cnrs_xz: list[int] = field(default_factory=list)
    c2a_r: list[float] | None = None
    c2a_t: list[float] | None = None
    sphere_map_fn: str = ""
    sock_host: str = "127.0.0.1"
    sock_port: int = -1
    com_port: str = ""
    com_baud: int = 115200
    accumulate_map: bool = True
    output_fn: str = ""  # output base name; FicTrac defaults to the video name
    map_frozen: bool = False  # spintrack: never update a loaded map (sphere_map_fn)
    # spintrack: separate the rig's static illumination from the ball's texture
    # instead of letting it accumulate in the surface map (see `photometry.py`).
    illumination: bool = True
    # spintrack: start from illumination fields measured on this rig before, read out of
    # a map `.npz` (`--save-map` writes them beside the map). Only the fields are taken,
    # so the rig's lighting can be carried to a new ball without a previous ball's
    # surface map - which `sphere_map_fn` would load along with them.
    illumination_fn: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)

    # ----- loading -----
    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a FicTrac `config.txt`, or a `.yaml`/`.yml`/`.toml` file, by suffix."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_mapping(yaml.safe_load(path.read_text()) or {})
        if suffix == ".toml":
            return cls.from_mapping(tomllib.loads(path.read_text()))
        return cls.from_text(path.read_text())

    @classmethod
    def from_text(cls, text: str) -> Config:
        """Parse the FicTrac text format."""
        values: dict[str, Any] = {}
        comments: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("##"):
                continue
            if line[0] in "#%":
                comments.append(line)
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            values[key.strip()] = _parse_text_value(value.strip())
        cfg = cls.from_mapping(values)
        cfg.comments = comments
        return cfg

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Config:
        """Build from already-typed values (YAML/TOML) or text-parsed ones."""
        cfg = cls()
        for key, value in values.items():
            if key in _FIELD_TYPES:
                setattr(cfg, key, _coerce(key, value))
            else:
                cfg.extra[key] = value
        return cfg

    # ----- saving -----
    def to_text(
        self, header: str = "## spintrack config file (FicTrac compatible)"
    ) -> str:
        """Serialize in the FicTrac text format; keys are sorted, defaults included."""
        rows: dict[str, str] = {}
        for f in dataclasses.fields(self):
            if f.name in ("extra", "comments"):
                continue
            value = getattr(self, f.name)
            if value is None or value == "" or value == []:
                continue
            rows[f.name] = _format_text_value(value)
        for key, value in self.extra.items():
            rows[key] = _format_text_value(value)
        lines = [header] + [f"{key:<16} : {rows[key]}" for key in sorted(rows)]
        lines.extend(self.comments)
        return "\n".join(lines) + "\n"

    def to_mapping(self) -> dict[str, Any]:
        """Plain dict of set values (for YAML/TOML output or inspection)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in ("extra", "comments"):
                continue
            value = getattr(self, f.name)
            if value is None or value == "" or value == []:
                continue
            out[f.name] = value
        out.update(self.extra)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self.to_mapping(), sort_keys=True))
        else:
            path.write_text(self.to_text())

    # ----- derived -----
    def c2a_source(self) -> str | None:
        """Which key defines the camera-to-animal transform, or None if none does."""
        if self.c2a_r is not None and len(self.c2a_r) == 3:
            return "c2a_r"
        if self.c2a_src.startswith("c2a_cnrs_"):
            corners = getattr(self, self.c2a_src, None)
            if corners and len(corners) == 8:
                return self.c2a_src
        return None

    def has_ball(self) -> bool:
        return (self.roi_c is not None and self.roi_r is not None) or len(
            self.roi_circ
        ) >= 6

    def window_size(self) -> int:
        """Side of the square tracking window in pixels (FicTrac: 10 * q_factor)."""
        return 10 * int(self.q_factor)


# Field name -> coercion kind.
_FIELD_TYPES: dict[str, str] = {
    "src_fn": "str",
    "vfov": "auto_float",
    "fisheye": "bool",
    "q_factor": "int",
    "src_fps": "float",
    "max_bad_frames": "int",
    "do_display": "bool",
    "save_debug": "bool",
    "save_raw": "bool",
    "vid_codec": "str",
    "thr_ratio": "float",
    "thr_win_pc": "float",
    "opt_do_global": "bool",
    "opt_max_err": "float",
    "opt_max_evals": "int",
    "opt_bound": "float",
    "opt_tol": "float",
    "roi_circ": "ints",
    "roi_c": "floats",
    "roi_r": "float",
    "roi_ignr": "int_lists",
    "c2a_src": "str",
    "c2a_cnrs_xy": "ints",
    "c2a_cnrs_yz": "ints",
    "c2a_cnrs_xz": "ints",
    "c2a_r": "floats",
    "c2a_t": "floats",
    "sphere_map_fn": "str",
    "sock_host": "str",
    "sock_port": "int",
    "com_port": "str",
    "com_baud": "int",
    "accumulate_map": "bool",
    "output_fn": "str",
    "map_frozen": "bool",
    "illumination": "bool",
    "illumination_fn": "str",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return round(float(value))


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _coerce(key: str, value: Any) -> Any:
    kind = _FIELD_TYPES[key]
    if kind == "str":
        return "" if value is None else str(value)
    if kind == "float":
        return float(value)
    if kind == "auto_float":
        # `auto` (or a missing key) means "measure it from the recording"; see
        # `spintrack.autofit.fit_vfov`.
        if value is None or (isinstance(value, str) and value.strip().lower() in _AUTO):
            return None
        return float(value)
    if kind == "int":
        return _to_int(value)
    if kind == "bool":
        return _to_bool(value)
    if kind == "ints":
        return [_to_int(v) for v in _flatten(value)]
    if kind == "floats":
        return [float(v) for v in _flatten(value)]
    if kind == "int_lists":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a list of polygons")
        if value and not isinstance(value[0], (list, tuple)):
            value = [value]  # a single polygon written without the outer braces
        return [[_to_int(v) for v in _flatten(poly)] for poly in value]
    raise AssertionError(kind)


def _parse_text_value(text: str) -> Any:
    """Parse one FicTrac text value: nested braces, y/n, number, or string."""
    text = text.strip().rstrip(",").strip()
    if text.startswith("{"):
        value, rest = _parse_braces(text)
        if rest.strip():
            raise ValueError(f"trailing text after vector: {text!r}")
        return value
    return _parse_scalar(text)


def _parse_scalar(text: str) -> Any:
    lower = text.lower()
    if lower in ("y", "n"):
        return lower == "y"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_braces(text: str) -> tuple[list[Any], str]:
    """Parse `{ ... }` (nested allowed) at the start of `text`; return (value, rest)."""
    assert text[0] == "{"
    items: list[Any] = []
    rest = text[1:]
    while True:
        rest = rest.lstrip().lstrip(",").lstrip()
        if not rest:
            raise ValueError("unterminated vector in config value")
        if rest[0] == "}":
            return items, rest[1:]
        if rest[0] == "{":
            value, rest = _parse_braces(rest)
            items.append(value)
            continue
        end = len(rest)
        for i, ch in enumerate(rest):
            if ch in ",}":
                end = i
                break
        token = rest[:end].strip()
        if token:
            items.append(_parse_scalar(token))
        rest = rest[end:]


def _format_text_value(value: Any) -> str:
    if isinstance(value, bool):
        return "y" if value else "n"
    if isinstance(value, (list, tuple)):
        return "{ " + ", ".join(_format_text_value(v) for v in value) + " }"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
