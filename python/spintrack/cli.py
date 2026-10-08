"""Command line entry point (`spintrack`)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from logging.handlers import MemoryHandler
from pathlib import Path

from spintrack import __version__

log = logging.getLogger("spintrack")

RUN_DESCRIPTION = """\
Track the ball in a video (or camera index), or in the one a config names in `video`.
Writes tracks.parquet (the records), summary.json (the run quality), log.txt and
config.toml (the config as run) into one folder: --out, or NAME_spintrack next to the
video, where NAME is the config's output name or the video's name. `spintrack VIDEO
...` is short for `spintrack run VIDEO ...`.
"""
# The first lines of the config.toml a run writes.
RUN_CONFIG_HEADER = (
    "# The config this run used, with the command line's changes and any ball it",
    "# detected. `spintrack run` on this file with `--out` another folder, and the",
    "# run's --two-pass and --max-frames if any, tracks the same video the same way.",
)


def _add_camera_position(p, what: str) -> None:
    p.add_argument(
        "--camera-position",
        nargs=3,
        type=float,
        default=None,
        metavar=("ELEV", "AZIM", "TWIST"),
        help=f"{what} (degrees; a camera directly behind the animal, level with the "
        f"ball, is 0 180 0)",
    )


def _add_run(sub, common) -> None:
    p = sub.add_parser(
        "run",
        parents=[common],
        help="track a video, or the recording or camera a config names",
        description=RUN_DESCRIPTION,
    )
    p.add_argument(
        "input",
        metavar="VIDEO|CONFIG",
        help="a video or camera index, or a config (.toml) that names one",
    )
    p.add_argument(
        "--config", default=None, help="the rig's config, for a VIDEO given directly"
    )
    _add_camera_position(p, "where the camera sits, instead of the config's")
    p.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="the output folder (default: NAME_spintrack next to the video)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace outputs of an earlier run"
    )
    p.add_argument(
        "--two-pass",
        action="store_true",
        help="map the ball in a first pass, then re-track from that map",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N", help="stop after N frames"
    )
    p.add_argument(
        "--debug-video",
        action="store_true",
        help="also write an annotated video, debug.mp4",
    )
    p.add_argument(
        "--debug-axes",
        action="store_true",
        help="draw the ball's axes in the debug video (implies --debug-video)",
    )
    p.add_argument(
        "--save-map", action="store_true", help="also write the final map, map.npz"
    )
    p.add_argument(
        "--load-map",
        default=None,
        metavar="PATH",
        help="start from a saved map (.npz or a FicTrac sphere-map .png)",
    )
    live = p.add_argument_group("streaming (FicTrac's line format)")
    live.add_argument(
        "--udp", default=None, metavar="HOST:PORT", help="stream records over UDP"
    )
    live.add_argument(
        "--tcp", default=None, metavar="HOST:PORT", help="stream records over TCP"
    )
    live.add_argument(
        "--serial", default=None, metavar="PORT[:BAUD]", help="stream over serial"
    )
    live.add_argument(
        "--print", action="store_true", help="print records to the terminal"
    )
    p.set_defaults(func=cmd_run)


def _add_map(sub, common) -> None:
    p = sub.add_parser(
        "map", parents=[common], help="render a saved surface map as an image"
    )
    p.add_argument("map", help="a spintrack .npz map or a FicTrac sphere-map .png")
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="output image (default: MAP.png for a .npz, MAP-render.png for a .png)",
    )
    p.add_argument(
        "--layout",
        choices=("grid", "cube"),
        default="grid",
        help="grid: an equal-area rectangle, as FicTrac draws its maps (default); "
        "cube: the unfolded cube the map is stored on, which keeps the poles square",
    )
    p.add_argument(
        "--w-min",
        type=float,
        default=0.1,
        metavar="W",
        help="weight below which a cell counts as never seen",
    )
    p.set_defaults(func=cmd_map)


def cmd_map(args) -> int:
    import cv2
    import numpy as np

    from spintrack.maps import load_map, render_map

    path = Path(args.map)
    npz = path.suffix.lower() == ".npz"
    if args.out:
        out = Path(args.out)
    else:
        out = (
            path.with_suffix(".png")
            if npz
            else path.with_name(f"{path.stem}-render.png")
        )
    if out.resolve() == path.resolve():
        raise ValueError(f"{out} is the map itself; pass another --out")
    if npz:
        with np.load(path) as z:
            shape = z["mean"].shape
    else:  # a FicTrac template: convert it on its own grid
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"could not read {path} as an image")
        shape = image.shape
    mean, weight = load_map(path, shape)
    cv2.imwrite(str(out), render_map(mean, weight, args.w_min, args.layout))
    seen = 100.0 * float(np.mean(weight >= args.w_min))
    log.info("%s: %.0f%% of the ball seen -> %s", path, seen, out)
    return 0


def _add_calibrate(sub, common) -> None:
    p = sub.add_parser(
        "calibrate",
        parents=[common],
        help="interactive ball / animal-frame calibration",
    )
    p.add_argument(
        "config", help="the config.toml to update (created when missing, with --src)"
    )
    p.add_argument(
        "--src",
        default=None,
        help="the video (or camera index) to use instead of the config's",
    )
    _add_camera_position(p, "write where the camera sits, without a window")
    p.add_argument(
        "--auto",
        action="store_true",
        help="fit the ball, and the field of view if missing, from the recording; "
        "no window",
    )
    p.add_argument(
        "--frames", type=int, default=100, help="frames to detect the ball from"
    )
    p.set_defaults(func=cmd_calibrate)


def cmd_calibrate(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.camera_position is not None:
        from spintrack.calibrate.headless import write_camera_position

        try:
            write_camera_position(args.config, *args.camera_position, src=args.src)
        except ValueError as exc:
            logging.getLogger("spintrack").error("%s", exc)
            return 2
        if not args.auto:
            return 0
    if args.auto:
        from spintrack.calibrate.headless import write_auto_geometry

        return write_auto_geometry(args.config, args.src, args.frames)
    from spintrack.calibrate.gui import calibrate

    return calibrate(args.config, args.src)


def _host_port(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "127.0.0.1", int(port)


def _inputs(args) -> tuple[str | None, str | None]:
    """The config and the source `run` was given: its positional is either.

    A `.txt` or YAML positional is taken for an old config, which `Config.load`
    refuses with a pointer to the TOML format.
    """
    if Path(args.input).suffix.lower() not in (".toml", ".txt", ".yaml", ".yml"):
        return args.config, args.input
    if args.config:
        raise ValueError(
            f"{args.input} is a config; --config goes with a video: "
            f"spintrack run VIDEO --config CONFIG"
        )
    return args.input, None


def _outputs(args, cfg, src: str) -> dict[str, Path]:
    """The files this run writes, by kind; refuses to replace any unless --overwrite.

    They go into one folder: `--out`, or NAME_spintrack next to the video (in the
    current directory for a camera), where NAME is `output.name` or the video's name.
    """
    if args.out:
        folder = Path(args.out)
    else:
        camera = src.isdigit()
        if cfg.output.name:
            name = Path(cfg.output.name).name
        else:
            name = "camera" if camera else Path(src).stem
        folder = (Path() if camera else Path(src).parent) / f"{name}_spintrack"
    paths = {
        "tracks": folder / "tracks.parquet",
        "summary": folder / "summary.json",
        "log": folder / "log.txt",
        "config": folder / "config.toml",
    }
    if cfg.output.debug_video:
        paths["debug"] = folder / "debug.mp4"
    if args.save_map:
        paths["map"] = folder / "map.npz"
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise ValueError(
            f"outputs of an earlier run in {folder}: {', '.join(existing)} "
            f"(--overwrite replaces them)"
        )
    return paths


def _streams(args, cfg) -> list:
    """The sockets, serial port and terminal the records are streamed to."""
    from spintrack.io.recorders import (
        SerialRecorder,
        TcpRecorder,
        TerminalRecorder,
        UdpRecorder,
    )

    out = []
    stream = cfg.stream
    if args.udp or (stream.udp and not args.tcp):
        out.append(UdpRecorder(*_host_port(args.udp or stream.udp)))
    if args.tcp:
        out.append(TcpRecorder(*_host_port(args.tcp)))
    if args.serial or stream.serial:
        port, _, baud = (args.serial or stream.serial).partition(":")
        out.append(SerialRecorder(port, int(baud) if baud else 115200))
    if args.print:
        out.append(TerminalRecorder())
    return out


def _provenance(args, config, cfg, src: str, prepared) -> tuple[dict, dict]:
    """What the sidecar records about the inputs, and the summary lines they add."""
    camera = cfg.camera
    identity = camera.rotation is not None and not any(camera.rotation)
    provenance = {
        "config": config,
        "source": src,
        "vfov": {"value": camera.vfov_deg, "source": "config"},
        "ball": {"source": "config"},
        "camera_position": {
            "source": "command line" if args.camera_position else "config",
            "identity": identity,
        },
    }
    checks = {"camera position": "identity (explicit)"} if identity else {}
    if prepared is not None:
        provenance["ball"] = prepared.report()
        checks["ball"] = prepared.line()
        if prepared.vfov is not None:
            provenance["vfov"] = prepared.vfov.report()
            checks["vfov"] = prepared.vfov.line()
    return provenance, checks


def _clock(seconds: float) -> str:
    minutes, s = divmod(round(seconds), 60)
    hours, m = divmod(minutes, 60)
    return f"{hours}:{m:02d}:{s:02d}" if hours else f"{m}:{s:02d}"


def _progress(stats) -> None:
    done = f"{stats.frames} frames"
    if stats.total and stats.fps > 0:
        left = max(stats.total - stats.frames, 0) / stats.fps
        done = (
            f"{100 * stats.frames / stats.total:.0f}% ({stats.frames}/{stats.total} "
            f"frames, {_clock(left)} left)"
        )
    log.info("%s, %d dropped, %.0f fps", done, stats.dropped, stats.fps)


class _RunLog:
    """The run's log.txt: the lines the terminal shows at INFO, and warnings.

    Attached for the whole command: lines logged before the output folder exists are
    held back and written once `open` names the file (a run refused before that leaves
    no files), and the error that ends a run is in it too.
    """

    def __init__(self):
        # With no target it keeps every record; with one, it writes each through.
        self._buffer = MemoryHandler(1, flushLevel=logging.NOTSET)
        self._buffer.setLevel(logging.INFO)
        self._loggers = (log, logging.getLogger("py.warnings"))
        for logger in self._loggers:
            logger.addHandler(self._buffer)

    def open(self, path: Path) -> None:
        target = logging.FileHandler(path, "w", encoding="utf-8")
        target.setFormatter(logging.Formatter("%(message)s"))
        self._buffer.setTarget(target)
        self._buffer.flush()

    def close(self) -> None:
        for logger in self._loggers:
            logger.removeHandler(self._buffer)
        target = self._buffer.target
        self._buffer.close()
        if target is not None:
            target.close()


def cmd_run(args) -> int:
    import cv2

    from spintrack.io.parquet import ParquetWriter
    from spintrack.io.sources import open_source
    from spintrack.pipeline import open_config, run
    from spintrack.quality import format_summary, write_sidecar

    # OpenCV spreads its remaps and filters over every core by default; two threads
    # track as fast and leave the rest of the machine to other runs.
    cv2.setNumThreads(2)
    config, src = _inputs(args)
    cfg, src = open_config(config, src, args.two_pass, args.camera_position)
    if args.load_map:
        cfg.tracking.initial_map = args.load_map
    if args.debug_video or args.debug_axes:
        cfg.output.debug_video = True
    outputs = _outputs(args, cfg, src)
    source = open_source(src)
    recorders = []
    try:
        prepared = None
        if not cfg.ball.rim:
            from spintrack.autofit import prepare_config

            prepared = prepare_config(cfg, src)
        provenance, checks = _provenance(args, config, cfg, src, prepared)
        # Streams first: one that cannot connect then fails before any file exists.
        recorders = _streams(args, cfg)
        outputs["tracks"].parent.mkdir(parents=True, exist_ok=True)
        args.run_log.open(outputs["log"])
        cfg.save(outputs["config"], RUN_CONFIG_HEADER, full=True)
        recorders.append(ParquetWriter(outputs["tracks"], provenance))
        size = f"{source.width}x{source.height}, {source.fps:g} fps"
        n = getattr(source, "n_frames", None)
        log.info(
            "spintrack %s: %s (%s%s)",
            __version__, src, size, f", {n} frames" if n else "",
        )  # fmt: skip
        t0 = time.perf_counter()
        stats = run(
            cfg,
            source,
            recorders,
            two_pass_source=(lambda: open_source(src)) if args.two_pass else None,
            max_frames=args.max_frames,
            progress=_progress,
            debug_video=outputs.get("debug"),
            debug_axes=args.debug_axes,
            save_map=outputs.get("map"),
        )
        elapsed = time.perf_counter() - t0
    finally:
        source.close()
        for rec in recorders:
            rec.close()
    log.info(
        "done: %d frames, %d dropped, %.1f s (%.0f fps)",
        stats.frames, stats.dropped, elapsed, stats.frames / max(elapsed, 1e-9),
    )  # fmt: skip
    log.debug("tracking: %.2f ms/frame", stats.tracking_ms_per_frame)
    if stats.quality is None:
        del outputs["summary"]
    else:
        stats.quality.checks = {**checks, **stats.quality.checks}
        log.info("%s", format_summary(stats.quality))
        sidecar = {**provenance, "geometry": stats.geometry}
        write_sidecar(outputs["summary"], stats.quality, sidecar)
    written = ", ".join(p.name for p in outputs.values())
    log.info("wrote %s in %s", written, outputs["tracks"].parent)
    return 0


def _message(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.filename and exc.strerror:
        return f"{exc.filename}: {exc.strerror}"
    return str(exc)


def _is_input(arg: str) -> bool:
    """Whether a first argument is a `run` input: a path, or a camera index.

    A path counts even if missing, for `run` to report it, but a bare word must name a
    file, so that a misspelled command still gets argparse's list of commands.
    """
    path = Path(arg)
    return bool(path.suffix) or path.name != arg or arg.isdigit() or path.is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spintrack",
        epilog="spintrack VIDEO ... is short for spintrack run VIDEO ...",
    )
    parser.add_argument(
        "--version", action="version", version=f"spintrack {__version__}"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug messages, and a traceback on errors",
    )
    sub = parser.add_subparsers(dest="command")
    _add_run(sub, common)
    _add_map(sub, common)
    _add_calibrate(sub, common)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] not in sub.choices and _is_input(argv[0]):
        argv.insert(0, "run")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.captureWarnings(True)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    from spintrack.detect import DetectionError

    args.run_log = _RunLog()
    # A kill, or the hangup of a closed terminal, stops a run as Ctrl-C does, so the
    # records so far are still written.
    kills = [getattr(signal, n) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
    handlers = {s: signal.signal(s, signal.default_int_handler) for s in kills}
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except (OSError, ValueError, DetectionError) as exc:
        log.error("error: %s", _message(exc), exc_info=args.verbose)
        return 2
    finally:
        args.run_log.close()
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    sys.exit(main())
