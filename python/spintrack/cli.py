"""Command line entry point (`spintrack`)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from spintrack import __version__


def _add_run(sub) -> None:
    p = sub.add_parser(
        "run", help="track a video or camera described by a FicTrac config"
    )
    p.add_argument("config", help="config.txt (FicTrac format), .yaml or .toml")
    p.add_argument(
        "--src", default=None, help="override src_fn: video path or camera index"
    )
    p.add_argument(
        "--out", default=None, help="output .dat path (default: next to the video)"
    )
    p.add_argument(
        "--udp", default=None, metavar="HOST:PORT", help="stream records over UDP"
    )
    p.add_argument(
        "--tcp", default=None, metavar="HOST:PORT", help="stream records over TCP"
    )
    p.add_argument(
        "--serial", default=None, metavar="PORT[:BAUD]", help="stream over serial"
    )
    p.add_argument("--print", action="store_true", help="print records to the terminal")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--all-pixels", action="store_true", help="solve on every window pixel"
    )
    p.add_argument(
        "--no-prefetch", action="store_true", help="decode in the tracking thread"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)


def _add_calibrate(sub) -> None:
    p = sub.add_parser("calibrate", help="interactive ball / animal-frame calibration")
    p.add_argument("config", help="config.txt to read and update (needs vfov)")
    p.add_argument(
        "--src", default=None, help="override src_fn: video path or camera index"
    )
    p.set_defaults(func=cmd_calibrate)


def cmd_calibrate(args) -> int:
    from spintrack.calibrate.gui import calibrate

    return calibrate(args.config, args.src)


def _host_port(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "127.0.0.1", int(port)


def cmd_run(args) -> int:
    from spintrack.config import Config
    from spintrack.engine import TrackParams
    from spintrack.io.recorders import (
        FileRecorder,
        SerialRecorder,
        TcpRecorder,
        TerminalRecorder,
        UdpRecorder,
    )
    from spintrack.io.sources import open_source
    from spintrack.pipeline import run

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    log = logging.getLogger("spintrack")
    config_path = Path(args.config)
    cfg = Config.load(config_path)
    src_spec = args.src if args.src is not None else cfg.src_fn
    if not src_spec:
        log.error("no source: set src_fn in the config or pass --src")
        return 2
    if not str(src_spec).isdigit():
        src_path = Path(src_spec)
        if not src_path.is_absolute():
            src_path = config_path.parent / src_path
        src_spec = str(src_path)
    source = open_source(src_spec)

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = cfg.output_fn or (
            Path(src_spec).stem if not str(src_spec).isdigit() else "camera"
        )
        out_dir = config_path.parent if not Path(base).is_absolute() else Path()
        out_path = out_dir / f"{base}-{stamp}.dat"
    recorders = [FileRecorder(out_path)]
    if args.udp or (cfg.sock_port > 0 and not args.tcp):
        host, port = (
            _host_port(args.udp) if args.udp else (cfg.sock_host, cfg.sock_port)
        )
        recorders.append(UdpRecorder(host, port))
    if args.tcp:
        recorders.append(TcpRecorder(*_host_port(args.tcp)))
    if args.serial or cfg.com_port:
        spec = args.serial or f"{cfg.com_port}:{cfg.com_baud}"
        port, _, baud = spec.partition(":")
        recorders.append(SerialRecorder(port, int(baud) if baud else 115200))
    if args.print:
        recorders.append(TerminalRecorder())

    params = TrackParams(max_pixels=None) if args.all_pixels else None
    log.info("spintrack %s: %s -> %s", __version__, src_spec, out_path)

    def progress(stats):
        log.info(
            "%d frames, %d dropped, %.0f fps (%.2f ms/frame tracking)",
            stats.frames, stats.dropped, stats.fps, stats.tracking_ms_per_frame,
        )  # fmt: skip

    try:
        stats = run(
            cfg,
            source,
            recorders,
            params=params,
            max_frames=args.max_frames,
            prefetch=not args.no_prefetch,
            progress=progress,
        )
    finally:
        source.close()
        for rec in recorders:
            rec.close()
    log.info(
        "done: %d frames, %d tracked, %d dropped, %.1f s (%.0f fps, %.2f ms/frame tracking)",
        stats.frames, stats.tracked, stats.dropped, stats.wall_s, stats.fps,
        stats.tracking_ms_per_frame,
    )  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spintrack")
    parser.add_argument(
        "--version", action="version", version=f"spintrack {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    _add_run(sub)
    _add_calibrate(sub)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
