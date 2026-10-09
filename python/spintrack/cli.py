"""The command line: `spintrack run`, `spintrack gui` and `spintrack doctor`.

Typer, and the modules a command needs, are imported here when it runs:
`import spintrack` loads neither.
"""

import json
import logging
import signal
import socket
import sys
import time
from logging.handlers import MemoryHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer
from typer.core import TyperGroup

from spintrack import __version__
from spintrack.config import is_override

log = logging.getLogger("spintrack")

# The first lines of the config.toml a run writes.
RUN_CONFIG_HEADER = (
    "# The config this run used, with the command line's changes and what it found",
    "# in the recording. `spintrack run` on this file with `--out` another folder, and",
    "# the run's --two-pass and --max-frames if any, tracks the video the same way.",
)
# What a positional argument ending so is taken for: a config, not a video.
CONFIG_SUFFIXES = (".toml", ".txt", ".yaml", ".yml")
STREAMING = "Streaming (FicTrac's line format)"
# Interfaces a page served on is reached from this machine only, through `ssh -L`
# from another.
LOOPBACK = ("127.0.0.1", "localhost", "::1")


class _Commands(TyperGroup):
    """The commands, with a hint for a video given without `run`."""

    def resolve_command(self, ctx, args):
        if args and self.get_command(ctx, args[0]) is None and _is_input(args[0]):
            ctx.fail(
                f"{args[0]!r} is not a command; to track it: spintrack run {args[0]}"
            )
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=_Commands,
    help="Track the rotation of a trackball from video.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
    context_settings={"help_option_names": ["-h", "--help"]},
)

ConfigOption = Annotated[
    str | None,
    typer.Option(
        "-c",
        "--config",
        metavar="CONFIG",
        help="A config (`.toml`) to start from, such as one `spintrack gui` saved.",
    ),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        metavar="KEY=VALUE",
        help="Set a config key, such as `tracking.window_px=80` or "
        "`camera.position_deg=0,180,0`; `none` restores its default. Repeatable.",
    ),
]
HostOption = Annotated[
    str,
    typer.Option(
        "--host",
        metavar="HOST",
        help="The interface to serve the page on; by default, this machine.",
    ),
]
PortOption = Annotated[
    int | None,
    typer.Option(
        "--port",
        min=1,
        max=65535,
        metavar="PORT",
        show_default=False,
        help="The page's port (default: 8300). A busy port is passed over for the "
        "next free one.",
    ),
]
VerboseOption = Annotated[
    bool,
    typer.Option("-v", "--verbose", help="Debug messages, and a traceback on errors."),
]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"spintrack {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Track the rotation of a trackball from video."""


def _jobs(args) -> list[tuple[str | None, str | None]]:
    """What to track, as (config, source) pairs.

    A positional argument is a config that names its video, or a video (or camera
    index), which `--config` then describes.

    A misuse is a usage error, through `args.ctx`: status 2, with the usage line.
    """
    for item in args.overrides:
        if not is_override(item):
            args.ctx.fail(f"--set takes KEY=VALUE, not {item!r}")
    jobs: list[tuple[str | None, str | None]] = []
    for item in args.inputs:
        if is_override(item):
            args.ctx.fail(f"{item!r} sets a config key: --set {item}")
        if Path(item).suffix.lower() in CONFIG_SUFFIXES:
            if args.config:
                args.ctx.fail(
                    f"{item} is a config; --config goes with videos: spintrack "
                    f"{args.command} VIDEO --config CONFIG"
                )
            jobs.append((item, None))
        else:
            jobs.append((args.config, item))
    if not jobs:
        if not args.config:
            args.ctx.fail(f"name a video: spintrack {args.command} VIDEO")
        jobs.append((args.config, None))  # the config names its video
    if getattr(args, "out", None) and len(jobs) > 1:
        args.ctx.fail("--out names one video's folder; leave it out for several")
    return jobs


def _host_port(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "127.0.0.1", int(port)


def _outputs(args, cfg, src: str) -> dict[str, Path]:
    """The files this run writes, by kind; refuses to replace any unless --force.

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
    if existing and not args.force:
        raise ValueError(
            f"outputs of an earlier run in {folder}: {', '.join(existing)} "
            f"(--force replaces them)"
        )
    return paths


def _streams(args, cfg) -> list:
    """The sockets, serial port and terminal the records are streamed to."""
    from spintrack.io.recorders import (
        Recorder,
        SerialRecorder,
        TcpRecorder,
        TerminalRecorder,
        UdpRecorder,
    )

    out: list[Recorder] = []
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

    name = "live view" if mode == "run" else "gui"
    view = LiveView(mode)
    try:
        view.url, view.stop_server = serve(
            view, controls, port=args.port, host=args.host
        )
    except OSError as exc:
        log.warning("%s: off (%s)", name, exc)
        view.close()
        return None
    port = int(view.url.split("/")[2].rsplit(":", 1)[1])
    if args.port and port != args.port:
        log.info("port %d is busy; serving on %d", args.port, port)
    if args.host in LOOPBACK:
        log.info(
            "%s: %s (from another machine: ssh -L %d:localhost:%d %s)",
            name, view.url, port, port, socket.gethostname(),
        )  # fmt: skip
    else:
        log.info("%s: %s", name, view.url)
    return view


def _run(args) -> int:
    import cv2

    # OpenCV spreads its remaps and filters over every core by default; two threads
    # track as fast and leave the rest of the machine to other runs.
    cv2.setNumThreads(2)
    jobs = _jobs(args)
    overrides = args.overrides
    view = None if args.no_live else _start_page(args, "run")
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
    return 1 if failed else 0


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


def _gui(args) -> int:
    from spintrack.web.gui import GuiSession

    ((config, src),) = _jobs(args)
    overrides = args.overrides
    # Where Save writes: the config the gui started from, or one next to the video.
    if config is not None:
        save_to = Path(config)
    else:
        assert src is not None  # without a config, a job names its video
        save_to = (Path(src).parent if not src.isdigit() else Path()) / "spintrack.toml"
    session = GuiSession(config, src, overrides, save_to)
    view = _start_page(args, "gui", session)
    if view is None:
        return 1
    try:
        if not args.no_browser and _graphical() and view.url:
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

    A path counts even if missing, but a bare word must name a file, so that a
    misspelled command still gets Click's suggestion of the command meant.
    """
    path = Path(arg)
    return (
        bool(path.suffix)
        or path.name != arg
        or arg.isdigit()
        or is_override(arg)
        or path.is_file()
    )


def _invoke(command, args) -> None:
    """Run a command's body with logging and signals set up, and exit with its status.

    An error the user can act on (a missing file, a wrong config key) is one line, with
    the traceback under `--verbose`; status 1. Ctrl-C, a kill or a hangup stops the
    command as Ctrl-C does; status 130.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.captureWarnings(True)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    from spintrack.detect import DetectionError

    # A kill, or the hangup of a closed terminal, stops a run as Ctrl-C does, so the
    # records so far are still written.
    kills = [getattr(signal, n) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
    handlers = {s: signal.signal(s, signal.default_int_handler) for s in kills}
    try:
        status = command(args)
    except KeyboardInterrupt:
        status = 130
    except (OSError, ValueError, DetectionError) as exc:
        log.error("error: %s", _message(exc), exc_info=args.verbose)
        status = 1
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
    if status:
        raise typer.Exit(status)


@app.command()
def run(
    ctx: typer.Context,
    videos: Annotated[
        list[str] | None,
        typer.Argument(
            show_default=False,
            help="Videos or camera indices, or configs (`.toml`) that name their "
            "video.",
        ),
    ] = None,
    config: ConfigOption = None,
    set_: SetOption = None,
    out: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--out",
            metavar="DIR",
            help="The output folder, for one video (default: `NAME_spintrack` next "
            "to it).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the outputs of an earlier run.")
    ] = False,
    two_pass: Annotated[
        bool,
        typer.Option(
            "--two-pass",
            help="Map the ball in a first pass, then re-track from that map.",
        ),
    ] = False,
    max_frames: Annotated[
        int | None, typer.Option(min=1, metavar="N", help="Stop after N frames.")
    ] = None,
    debug_video: Annotated[
        bool,
        typer.Option(
            "--debug-video",
            help="Also write an annotated video, `debug.mp4` "
            "(`--set output.debug_video=true`).",
        ),
    ] = False,
    save_map: Annotated[
        bool, typer.Option("--save-map", help="Also write the final map, `map.npz`.")
    ] = False,
    no_live: Annotated[
        bool,
        typer.Option(
            "--no-live", help="Serve no live view (by default its link is printed)."
        ),
    ] = False,
    host: HostOption = "127.0.0.1",
    port: PortOption = None,
    udp: Annotated[
        str | None,
        typer.Option(
            metavar="HOST:PORT",
            help="Stream records over UDP.",
            rich_help_panel=STREAMING,
        ),
    ] = None,
    tcp: Annotated[
        str | None,
        typer.Option(
            metavar="HOST:PORT",
            help="Stream records over TCP.",
            rich_help_panel=STREAMING,
        ),
    ] = None,
    serial: Annotated[
        str | None,
        typer.Option(
            metavar="PORT[:BAUD]",
            help="Stream records over a serial port (the `serial` extra).",
            rich_help_panel=STREAMING,
        ),
    ] = None,
    print_: Annotated[
        bool,
        typer.Option(
            "--print",
            help="Print records to the terminal.",
            rich_help_panel=STREAMING,
        ),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Track the ball in each video.

    The ball, the field of view and the camera position come from the config (`-c`)
    when it has them, and from the recording when not. `--set` changes the config's
    keys for this run.

    Each video gets a folder, `NAME_spintrack` next to it (or `--out`), with
    `tracks.parquet`, `summary.json`, `log.txt` and `config.toml`, the config as run.
    The live view, whose link is printed first, shows the run as it goes.
    """
    args = SimpleNamespace(
        ctx=ctx,
        command="run",
        inputs=videos or [],
        config=config,
        overrides=set_ or [],
        out=out,
        force=force,
        two_pass=two_pass,
        max_frames=max_frames,
        debug_video=debug_video,
        save_map=save_map,
        no_live=no_live,
        host=host,
        port=port,
        udp=udp,
        tcp=tcp,
        serial=serial,
        print=print_,
        verbose=verbose,
    )
    _invoke(_run, args)


@app.command()
def gui(
    ctx: typer.Context,
    video: Annotated[
        str | None,
        typer.Argument(
            show_default=False,
            help="The video or camera index, or a config (`.toml`) that names its "
            "video.",
        ),
    ] = None,
    config: ConfigOption = None,
    set_: SetOption = None,
    host: HostOption = "127.0.0.1",
    port: PortOption = None,
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser", help="Print the page's link without opening a browser."
        ),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Fix a video's config while it tracks.

    Opens a page that tracks the video live while you fix the ball, the camera
    position and the tracking parameters, and saves them as a config for
    `spintrack run -c CONFIG`.
    """
    args = SimpleNamespace(
        ctx=ctx,
        command="gui",
        inputs=[video] if video is not None else [],
        config=config,
        overrides=set_ or [],
        host=host,
        port=port,
        no_browser=no_browser,
        verbose=verbose,
    )
    _invoke(_gui, args)


# Each status's marker and color, as `octacam doctor` shows them.
MARKERS = {
    "ok": ("\N{CHECK MARK}", "green"),
    "warn": ("\N{WARNING SIGN}", "yellow"),
    "error": ("\N{BALLOT X}", "red"),
    "info": ("\N{BULLET}", "cyan"),
}


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json", help="Print the report as JSON, for scripts, instead of text."
        ),
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Exit with status 1 on warnings too, not only on errors."
        ),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Report what this installation can do.

    Checks Python, the compiled core, PyTorch and the device SAM 3 runs on, whether
    SAM 3's checkpoint is cached, PyAV's FFmpeg and the serial extra. It downloads
    nothing. Exits with status 1 on errors (and on warnings with `--check`).
    """
    args = SimpleNamespace(json=json_output, check=check, verbose=verbose)
    _invoke(_doctor, args)


def _doctor(args) -> int:
    from spintrack.doctor import report

    sections = report()
    statuses = [status for s in sections for status, _ in s.findings]
    errors, warnings = statuses.count("error"), statuses.count("warn")
    if args.json:
        payload = {
            "spintrack_version": __version__,
            "sections": [
                {
                    "title": s.title,
                    "findings": [{"status": st, "text": t} for st, t in s.findings],
                }
                for s in sections
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"spintrack doctor -- spintrack {__version__}")
        for s in sections:
            typer.echo(f"\n{s.title}")
            for status, text in s.findings:
                marker, color = MARKERS[status]
                typer.echo("  " + typer.style(marker, fg=color) + f" {text}")
        typer.echo()
        if errors or warnings:
            typer.echo(f"{errors} error(s), {warnings} warning(s).")
        else:
            typer.secho("All checks passed.", fg="green", bold=True)
    return int(bool(errors or (args.check and warnings)))


def main(argv: list[str] | None = None) -> int:
    """Run the command line on `argv` (by default `sys.argv[1:]`); the exit status."""
    try:
        app(args=argv, prog_name="spintrack")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else int(exc.code is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
