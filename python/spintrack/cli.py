"""Command line entry point: `spintrack run` and `spintrack gui`."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import time
from logging.handlers import MemoryHandler
from pathlib import Path

from spintrack import __version__
from spintrack.config import is_override

log = logging.getLogger("spintrack")

RUN_DESCRIPTION = """\
Track the ball in each VIDEO (a video file or a camera index), or in the video a config
(.toml) names. The ball, the field of view and the camera position come from the config
(-c) when it has them, and from the recording when not. KEY=VALUE arguments override
the config's keys: tracking.window_px=80, camera.position_deg=0,180,0. Each video gets a
folder, NAME_spintrack next to it (or --out), with tracks.parquet, summary.json, log.txt
and config.toml, the config as run. `spintrack VIDEO ...` is short for `spintrack run
VIDEO ...`.
"""
GUI_DESCRIPTION = """\
Open a page that tracks VIDEO live while you fix the ball, the camera position and the
tracking parameters, and save them as a config for `spintrack run -c CONFIG`. KEY=VALUE
arguments override the config's keys, as for `spintrack run`.
"""
# The first lines of the config.toml a run writes.
RUN_CONFIG_HEADER = (
    "# The config this run used, with the command line's changes and what it found",
    "# in the recording. `spintrack run` on this file with `--out` another folder, and",
    "# the run's --two-pass and --max-frames if any, tracks the video the same way.",
)
# What a positional argument ending so is taken for: a config, not a video.
CONFIG_SUFFIXES = (".toml", ".txt", ".yaml", ".yml")


def _add_inputs(p, what: str) -> None:
    p.add_argument(
        "inputs",
        nargs="*",
        metavar="VIDEO|KEY=VALUE",
        help=f"{what}, and config overrides such as tracking.window_px=80",
    )
    p.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="CONFIG",
        help="a config (.toml) to start from, for instance one `spintrack gui` saved",
    )
    p.add_argument("--port", type=int, default=None, help="the page's port")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug messages, and a traceback on errors",
    )


def _run_parser(sub) -> None:
    p = sub.add_parser(
        "run",
        help="track videos (spintrack VIDEO is short for spintrack run VIDEO)",
        description=RUN_DESCRIPTION,
    )
    _add_inputs(p, "videos or camera indices, or configs that name their video")
    p.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="DIR",
        help="the output folder, for one video (default: NAME_spintrack next to it)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace outputs of an earlier run"
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="serve no preview page (by default its link is printed)",
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
        help="also write an annotated video, debug.mp4 (output.debug_video=true)",
    )
    p.add_argument(
        "--save-map", action="store_true", help="also write the final map, map.npz"
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


def _gui_parser(sub) -> None:
    p = sub.add_parser(
        "gui",
        help="fix the ball and the parameters on a video while it tracks",
        description=GUI_DESCRIPTION,
    )
    _add_inputs(p, "the video, or a config that names it")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="print the page's link without opening a browser",
    )
    p.set_defaults(func=cmd_gui)


def _jobs(args) -> tuple[list[tuple[str | None, str | None]], list[str]]:
    """What to track, as (config, source) pairs, and the overrides for all of them.

    A positional argument is an override (`KEY=VALUE`), a config that names its video,
    or a video (or camera index), which `--config` then describes.
    """
    overrides = [a for a in args.inputs if is_override(a)]
    jobs = []
    for item in args.inputs:
        if is_override(item):
            continue
        if Path(item).suffix.lower() in CONFIG_SUFFIXES:
            if args.config:
                raise ValueError(
                    f"{item} is a config; --config goes with videos: spintrack run "
                    f"VIDEO... --config CONFIG"
                )
            jobs.append((item, None))
        else:
            jobs.append((args.config, item))
    if not jobs:
        if not args.config:
            raise ValueError("name a video: spintrack run VIDEO")
        jobs.append((args.config, None))  # the config names its video
    if getattr(args, "out", None) and len(jobs) > 1:
        raise ValueError("--out names one video's folder; leave it out for several")
    return jobs, overrides


def _host_port(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "127.0.0.1", int(port)


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


def _provenance(config, overrides, cfg, src: str, prepared) -> tuple[dict, dict]:
    """What the sidecar records about the inputs, and the summary lines they add."""
    camera = cfg.camera
    identity = camera.rotation is not None and not any(camera.rotation)
    given = {item.partition("=")[0] for item in overrides}

    def origin(*keys: str) -> str:
        return "command line" if given & set(keys) else "config"

    position = origin("camera.position_deg", "camera.rotation")
    checks = {"camera position": "identity (explicit)"} if identity else {}
    if prepared is not None and prepared.camera is not None:
        position = prepared.camera_position
        checks["camera position"] = prepared.camera.line()
    provenance = {
        "config": config,
        "overrides": list(overrides),
        "source": src,
        "vfov": {"value": camera.vfov_deg, "source": origin("camera.vfov_deg")},
        "ball": {"source": origin("ball.rim")},
        "camera_position": {
            "source": position,
            "position_deg": camera.position_deg,
            "identity": identity,
        },
    }
    if prepared is not None and prepared.camera is not None:
        provenance["camera_position"]["fit"] = prepared.camera.report()
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
    """A run's log.txt: the lines the terminal shows at INFO, and warnings.

    Attached for the whole run: lines logged before the output folder exists are held
    back and written once `open` names the file (a run refused before that leaves no
    files), and the error that ends a run is in it too.
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


def _start_page(args, mode: str, controls=None):
    """The page's live view, serving at `view.url`; None when no port could be had."""
    from spintrack.web.live import LiveView
    from spintrack.web.server import serve

    view = LiveView(mode)
    try:
        view.url, view.stop_server = serve(view, controls, port=args.port)
    except OSError as exc:
        log.warning("%s page: off (%s)", mode, exc)
        view.close()
        return None
    port = view.url.split("/")[2].rsplit(":", 1)[1]
    log.info(
        "%s: %s (from another machine: ssh -L %s:localhost:%s %s)",
        "preview" if mode == "run" else "gui",
        view.url, port, port, socket.gethostname(),
    )  # fmt: skip
    return view


def cmd_run(args) -> int:
    import cv2

    # OpenCV spreads its remaps and filters over every core by default; two threads
    # track as fast and leave the rest of the machine to other runs.
    cv2.setNumThreads(2)
    jobs, overrides = _jobs(args)
    view = None if args.no_preview else _start_page(args, "run")
    failed = 0
    try:
        for i, (config, src) in enumerate(jobs):
            if len(jobs) > 1:
                log.info("[%d/%d] %s", i + 1, len(jobs), src or config)
            if view is not None:
                view.video(src or config, i, len(jobs))
            failed += not _run_job(args, config, src, overrides, view)
            if view is not None and view.stop_requested:
                break
    finally:
        if view is not None:
            view.close()
    if failed and len(jobs) > 1:
        log.error("%d of %d videos failed", failed, len(jobs))
    return 2 if failed else 0


def _run_job(args, config, src, overrides, view) -> bool:
    """Track one video into its folder; False, with the error logged, if it failed."""
    from spintrack.detect import DetectionError

    run_log = _RunLog()
    try:
        _track_job(args, config, src, overrides, view, run_log)
        return True
    except (OSError, ValueError, DetectionError) as exc:
        log.error("error: %s", _message(exc), exc_info=args.verbose)
        if view is not None:
            view.finish("failed", _message(exc))
        return False
    except KeyboardInterrupt:
        log.error("interrupted")
        raise
    finally:
        run_log.close()


def _track_job(args, config, src, overrides, view, run_log) -> None:
    from spintrack.autofit import complete_config
    from spintrack.io.parquet import ParquetWriter
    from spintrack.io.sources import open_source
    from spintrack.pipeline import open_config, run
    from spintrack.quality import format_summary, write_sidecar

    cfg, src = open_config(config, src, overrides, args.two_pass)
    if view is not None:
        view.rename(src)
    if args.debug_video:
        cfg.output.debug_video = True
    outputs = _outputs(args, cfg, src)
    source = open_source(src)
    recorders = []
    try:
        if view is not None:
            view.status("finding what the config leaves open")
        prepared = complete_config(cfg, src)
        provenance, checks = _provenance(config, overrides, cfg, src, prepared)
        # Streams first: one that cannot connect then fails before any file exists.
        recorders = _streams(args, cfg)
        outputs["tracks"].parent.mkdir(parents=True, exist_ok=True)
        run_log.open(outputs["log"])
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
            save_map=outputs.get("map"),
            view=view,
            masks=prepared.masks() if prepared is not None else (),
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
    summary = None
    if stats.quality is None:
        del outputs["summary"]
    else:
        stats.quality.checks = {**checks, **stats.quality.checks}
        summary = format_summary(stats.quality)
        log.info("%s", summary)
        sidecar = {**provenance, "geometry": stats.geometry}
        write_sidecar(outputs["summary"], stats.quality, sidecar)
    written = ", ".join(p.name for p in outputs.values())
    log.info("wrote %s in %s", written, outputs["tracks"].parent)
    if view is not None:
        view.finish("stopped" if view.stop_requested else "done", summary)


def cmd_gui(args) -> int:
    from spintrack.web.gui import GuiSession

    jobs, overrides = _jobs(args)
    if len(jobs) > 1:
        raise ValueError("the gui opens one video")
    config, src = jobs[0]
    # Where Save writes: the config the gui started from, or one next to the video.
    if config is not None:
        save_to = Path(config)
    else:
        save_to = (Path(src).parent if not src.isdigit() else Path()) / "spintrack.toml"
    session = GuiSession(config, src, overrides, save_to)
    view = _start_page(args, "gui", session)
    if view is None:
        return 2
    try:
        if not args.no_browser and _graphical():
            import webbrowser

            webbrowser.open(view.url)
        session.serve(view)  # until Ctrl-C or the page's Quit
    finally:
        session.close()
        view.close()
    return 0


def _graphical() -> bool:
    """Whether a browser opened here would show up on a screen."""
    import os

    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _message(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.filename and exc.strerror:
        return f"{exc.filename}: {exc.strerror}"
    return str(exc)


def _is_input(arg: str) -> bool:
    """Whether a first argument is a `run` input: a path, a camera index or KEY=VALUE.

    A path counts even if missing, for `run` to report it, but a bare word must name a
    file, so that a misspelled command still gets argparse's list of commands.
    """
    path = Path(arg)
    return (
        bool(path.suffix)
        or path.name != arg
        or arg.isdigit()
        or is_override(arg)
        or path.is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spintrack",
        description="Track the rotation of a trackball from video.",
        epilog="spintrack VIDEO ... is short for spintrack run VIDEO ...",
    )
    parser.add_argument(
        "--version", action="version", version=f"spintrack {__version__}"
    )
    sub = parser.add_subparsers(dest="command", title="commands")
    _run_parser(sub)
    _gui_parser(sub)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] not in sub.choices and _is_input(argv[0]):
        argv.insert(0, "run")
    if not argv or argv[0] not in sub.choices:
        parser.parse_args(argv)  # --help, --version, or an unknown command
        parser.print_help()
        return 0
    # Intermixed, so that options may come between the videos and the overrides.
    args = sub.choices[argv[0]].parse_intermixed_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.captureWarnings(True)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    from spintrack.detect import DetectionError

    # A kill, or the hangup of a closed terminal, stops a run as Ctrl-C does, so the
    # records so far are still written.
    kills = [getattr(signal, n) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
    handlers = {s: signal.signal(s, signal.default_int_handler) for s in kills}
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, DetectionError) as exc:
        log.error("error: %s", _message(exc), exc_info=args.verbose)
        return 2
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    sys.exit(main())
