"""Tracking configuration: FicTrac-compatible `config.txt`, YAML and TOML.

The FicTrac text format is `key : value` per line. Vectors are written `{ a, b, c }`,
nested vectors `{ { a, b }, { c, d } }`, booleans `y`/`n`. Lines starting with `#` or
`%` are comments; a `#` after whitespace starts an inline comment. Unknown keys are kept
in `Config.extra`. Saving a config that was loaded from text rewrites only the lines of
the keys that changed and appends new ones, so the user's layout and comments survive.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("spintrack")

_TRUE = {"y", "yes", "true", "1"}
_FALSE = {"n", "no", "false", "0"}
_AUTO = {"auto", "none", ""}
# Keys FicTrac reads that spintrack has no use for, and keys spintrack writes itself.
_OTHER_KNOWN_KEYS = {"enh_cfg_disp", "reconfig", "thr_rgb_tfrm", "c2a_angles"}
_INTERNAL = ("extra", "comments", "path", "_source")


@dataclass
class Config:
    """FicTrac's documented parameters plus spintrack's own.

    Defaults follow FicTrac. `vfov` has no default: it must be set, or `auto` for
    `spintrack calibrate --auto` to fit it.
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
    accumulate_map: bool = True  # the lab fork's key
    output_fn: str = ""  # output base name; FicTrac defaults to the video name
    map_frozen: bool = False  # spintrack: never update a loaded map (sphere_map_fn)
    # spintrack: separate the rig's static illumination from the ball's texture.
    illumination: bool = True
    # spintrack: start from the illumination fields of a map `.npz` saved on this rig,
    # without that map's ball surface.
    illumination_fn: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    # The file this config was loaded from; relative paths in it resolve against it.
    path: Path | None = field(default=None, repr=False, compare=False)
    # The text it was parsed from and the values it held, for an in-place save.
    _source: tuple | None = field(default=None, repr=False, compare=False)

    # ----- loading -----
    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a FicTrac `config.txt`, or a `.yaml`/`.yml`/`.toml` file, by suffix."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            cfg = cls.from_mapping(yaml.safe_load(path.read_text()) or {})
        elif suffix == ".toml":
            cfg = cls.from_mapping(tomllib.loads(path.read_text()))
        else:
            cfg = cls.from_text(path.read_text())
        cfg.path = path
        return cfg

    @classmethod
    def from_text(cls, text: str) -> Config:
        """Parse the FicTrac text format."""
        values: dict[str, Any] = {}
        comments: list[str] = []
        lines = text.splitlines()
        for raw in lines:
            line = raw.strip()
            if line[:1] in ("#", "%") and not line.startswith("##"):
                comments.append(line)
                continue
            parsed = _key_value(line)
            if parsed is None:
                continue
            key, value = parsed
            # String keys keep their text: `output_fn : 003` is not the number 3.
            if _FIELD_TYPES.get(key) == "str":
                values[key] = value
            else:
                values[key] = _parse_text_value(value)
        cfg = cls.from_mapping(values)
        cfg.comments = comments
        cfg._source = (lines, copy.deepcopy(cfg._keyed()))
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
        unknown = sorted(set(cfg.extra) - _OTHER_KNOWN_KEYS)
        if unknown:
            log.warning(
                "unknown config keys, kept but not used: %s", ", ".join(unknown)
            )
        if cfg.q_factor <= 0:
            log.warning(
                "q_factor %d is not positive; using 6, as FicTrac does", cfg.q_factor
            )
            cfg.q_factor = 6
        if cfg.opt_max_err >= 0:
            log.warning(
                "opt_max_err is ignored: spintrack does not drop frames on the "
                "matching error (see the run summary's hard-tracking episodes)"
            )
        return cfg

    # ----- saving -----
    def to_text(
        self, header: str = "## spintrack config file (FicTrac compatible)"
    ) -> str:
        """Serialize in the FicTrac text format.

        A config parsed from text keeps its lines: only the keys whose values changed
        are rewritten, keys no longer set are dropped, and new keys and comments are
        appended. Any other config is written as the keys that differ from the defaults.
        """
        current = self._keyed()
        if self._source is None:
            defaults = Config()._keyed()
            rows = {k: v for k, v in current.items() if defaults.get(k) != v}
            lines = [header] + [_format_line(k, rows[k]) for k in sorted(rows)]
            return "\n".join(lines + self.comments) + "\n"
        lines, loaded = self._source
        out: list[str] = []
        written: set[str] = set()
        old_comments: set[str] = set()
        for raw in lines:
            line = raw.strip()
            if line[:1] in ("#", "%") and not line.startswith("##"):
                old_comments.add(line)
                if line in self.comments:
                    out.append(raw)
                continue
            parsed = _key_value(line)
            if parsed is None:
                out.append(raw)
                continue
            key = parsed[0]
            if key in written:
                continue
            written.add(key)
            if key not in current:
                if key not in loaded:
                    out.append(raw)  # empty then, empty now
                continue
            if loaded.get(key) == current[key]:
                out.append(raw)
            else:
                out.append(_format_line(key, current[key]))
        defaults = Config()._keyed()
        for key, value in current.items():
            if key not in written and defaults.get(key) != value:
                out.append(_format_line(key, value))
        out.extend(c for c in self.comments if c not in old_comments)
        return "\n".join(out) + "\n"

    def _keyed(self) -> dict[str, Any]:
        """Every key that holds a value, spintrack's and the extra ones alike."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in _INTERNAL:
                continue
            value = getattr(self, f.name)
            if value is None or value == "" or value == []:
                continue
            out[f.name] = value
        out.update(self.extra)
        return out

    def to_mapping(self) -> dict[str, Any]:
        """Plain dict of set values (for YAML output or inspection)."""
        return dict(self._keyed())

    def save(self, path: str | Path | None = None) -> Path:
        """Write the config (to where it was loaded from by default)."""
        path = Path(path) if path is not None else self.path
        if path is None:
            raise ValueError("no path to save the config to")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self.to_mapping(), sort_keys=True))
        elif suffix == ".toml":
            raise ValueError(
                f"{path}: TOML configs are read-only; save as .txt or .yaml"
            )
        else:
            path.write_text(self.to_text())
        return path

    # ----- derived -----
    def source(self, override=None) -> str:
        """The source: a camera index, or a path, relative ones to the config file's."""
        spec = override if override is not None else self.src_fn
        if not spec:
            raise ValueError("no source: set src_fn in the config or pass --src")
        if str(spec).isdigit():
            return str(spec)
        spec = Path(spec)
        if not spec.is_absolute() and self.path is not None:
            spec = self.path.parent / spec
        return str(spec)

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


def _key_value(line: str) -> tuple[str, str] | None:
    """`(key, value text)` of a `key : value` line, or None for any other line."""
    line = line.strip()
    if not line or line[0] in "#%" or ":" not in line:
        return None
    key, _, value = line.partition(":")
    value = re.split(r"\s#", value, maxsplit=1)[0]
    return key.strip(), value.strip()


def _format_line(key: str, value: Any) -> str:
    return f"{key:<16} : {_format_text_value(value)}"


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
        # `auto` (or a missing key) means "measure it from the recording".
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
