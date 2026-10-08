"""Command line entry point (`spintrack`)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from spintrack import __version__

log = logging.getLogger("spintrack")

RUN_DESCRIPTION = """\
Track the ball in the recording (or camera) a FicTrac config describes. Writes
NAME.dat (FicTrac's 25 columns), NAME.parquet (the same records, with named columns)
and NAME-summary.json (the run quality) next to the config, where NAME is the config's
output_fn or the video's name.
"""


def _add_run(sub, common) -> None:
    p = sub.add_parser(
        "run",
        parents=[common],
        help="track a video or camera described by a FicTrac config",
        description=RUN_DESCRIPTION,
    )
    p.add_argument("config", help="config.txt (FicTrac format), .yaml or .toml")
    p.add_argument(
        "--src", default=None, help="override src_fn: video path or camera index"
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="DIR|PATH",
        help="a directory to write into, or a .dat or .parquet path that names "
        "every output",
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
        help="also write an annotated video, NAME-debug.mp4",
    )
    p.add_argument(
        "--debug-axes",
        action="store_true",
        help="draw the ball's axes in the debug video (implies --debug-video)",
    )
    p.add_argument(
        "--save-map", action="store_true", help="also write the final map, NAME-map.npz"
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
        "config", help="config.txt to update (created when missing, with --src)"
    )
    p.add_argument(
        "--src", default=None, help="override src_fn: video path or camera index"
    )
    p.add_argument(
        "--c2a-angles",
        nargs=3,
        type=float,
        default=None,
        metavar=("ELEV", "AZIM", "TWIST"),
        help="write c2a_r from the camera position in degrees, without a window "
        "(a camera directly behind the animal, level with the ball, is 0 180 0)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="fit the ball from the recording and write it to the config, no window",
    )
    p.add_argument(
        "--frames", type=int, default=100, help="frames to detect the ball from"
    )
    p.set_defaults(func=cmd_calibrate)


def cmd_calibrate(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.c2a_angles is not None:
        from spintrack.calibrate.headless import write_c2a_angles

        try:
            write_c2a_angles(args.config, *args.c2a_angles, src=args.src)
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


def _outputs(args, cfg, src: str) -> dict[str, Path]:
    """The files this run writes, by kind; refuses to replace any unless --overwrite."""
    out = Path(args.out) if args.out else None
    if out is not None and out.suffix.lower() in (".dat", ".parquet"):
        base = out.with_suffix("")
    else:
        name = cfg.output_fn or ("camera" if src.isdigit() else Path(src).stem)
        base = out / Path(name).name if out else Path(args.config).parent / name
    paths = {
        "dat": Path(f"{base}.dat"),
        "parquet": Path(f"{base}.parquet"),
        "summary": Path(f"{base}-summary.json"),
    }
    if args.debug_video or args.debug_axes or cfg.save_debug:
        paths["debug"] = Path(f"{base}-debug.mp4")
    if args.save_map:
        paths["map"] = Path(f"{base}-map.npz")
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise ValueError(
            f"outputs of an earlier run in {base.parent}: {', '.join(existing)} "
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
    if args.udp or (cfg.sock_port > 0 and not args.tcp):
        host, port = (
            _host_port(args.udp) if args.udp else (cfg.sock_host, cfg.sock_port)
        )
        out.append(UdpRecorder(host, port))
    if args.tcp:
        out.append(TcpRecorder(*_host_port(args.tcp)))
    if args.serial or cfg.com_port:
        spec = args.serial or f"{cfg.com_port}:{cfg.com_baud}"
        port, _, baud = spec.partition(":")
        out.append(SerialRecorder(port, int(baud) if baud else 115200))
    if args.print:
        out.append(TerminalRecorder())
    return out


def _provenance(args, cfg, src: str, prepared) -> tuple[dict, dict]:
    """What the sidecar records about the inputs, and the summary lines they add."""
    c2a = cfg.c2a_source()
    angles = cfg.extra.get("c2a_angles") if cfg.c2a_src == "sliders" else None
    identity = c2a == "c2a_r" and not any(cfg.c2a_r)
    provenance = {
        "config": str(args.config),
        "source": src,
        "vfov": {"value": cfg.vfov, "source": "config"},
        "ball": {"source": "config", "roi_c": cfg.roi_c, "roi_r": cfg.roi_r},
        "c2a": {
            "source": c2a,
            "identity": identity,
            "angles": list(angles) if angles else None,
        },
    }
    checks = {"c2a_r": "identity (explicit)"} if identity else {}
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


def cmd_run(args) -> int:
    import cv2

    from spintrack.io.parquet import ParquetWriter
    from spintrack.io.recorders import FileRecorder
    from spintrack.io.sources import open_source
    from spintrack.pipeline import open_config, run
    from spintrack.quality import format_summary, write_sidecar

    # OpenCV spreads its remaps and filters over every core by default; two threads
    # track as fast and leave the rest of the machine to other runs.
    cv2.setNumThreads(2)
    cfg, src = open_config(args.config, args.src, args.two_pass)
    if args.load_map:
        cfg.sphere_map_fn = args.load_map
    outputs = _outputs(args, cfg, src)
    source = open_source(src)
    recorders = []
    try:
        prepared = None
        if not cfg.has_ball():
            from spintrack.autofit import prepare_config

            prepared = prepare_config(cfg, src)
        provenance, checks = _provenance(args, cfg, src, prepared)
        # Streams first: one that cannot connect then fails before any file exists.
        recorders = _streams(args, cfg)
        outputs["dat"].parent.mkdir(parents=True, exist_ok=True)
        recorders += [
            FileRecorder(outputs["dat"]),
            ParquetWriter(outputs["parquet"], provenance),
        ]
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
    log.info("wrote %s in %s", written, outputs["dat"].parent)
    return 0


def _message(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.filename and exc.strerror:
        return f"{exc.filename}: {exc.strerror}"
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spintrack")
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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.setLevel(logging.DEBUG if args.verbose else logging.NOTSET)
    from spintrack.detect import DetectionError

    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except (OSError, ValueError, DetectionError) as exc:
        log.error("error: %s", _message(exc), exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    sys.exit(main())
