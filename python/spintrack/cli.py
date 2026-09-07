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
        "--save-map", default=None, metavar="PATH", help="write the final map (.npz)"
    )
    p.add_argument(
        "--load-map",
        default=None,
        metavar="PATH",
        help="start from a saved map (.npz or a FicTrac sphere-map .png)",
    )
    p.add_argument(
        "--frozen-map", action="store_true", help="never update the loaded map"
    )
    p.add_argument(
        "--map-projection",
        choices=("equal_area", "cube"),
        default=None,
        help="how the surface map tiles the sphere (default: cube)",
    )
    p.add_argument(
        "--debug-video",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="write an annotated debug video (default path: <out>-debug.mp4)",
    )
    p.add_argument(
        "--refine",
        type=int,
        default=0,
        metavar="SWEEPS",
        help="after tracking, re-estimate all frames against the complete map",
    )
    p.add_argument(
        "--refine-out",
        default=None,
        metavar="PATH",
        help="refined .dat (default: <out>-refined.dat)",
    )
    p.add_argument(
        "--no-prefetch", action="store_true", help="decode in the tracking thread"
    )
    p.add_argument(
        "--no-scale-check",
        action="store_true",
        help="skip the inner/outer check on the ball's assumed radius",
    )
    p.add_argument(
        "--no-illumination",
        action="store_true",
        help="do not separate the rig's static lighting from the ball's texture",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="do not write the run quality sidecar (<out>-summary.json)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)


def _add_summarize(sub) -> None:
    p = sub.add_parser("summarize", help="run quality summary of an existing .dat")
    p.add_argument("dat", help="a FicTrac-format .dat written by spintrack or FicTrac")
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="frame rate, if the .dat has no timestamps",
    )
    p.add_argument(
        "--json", default=None, metavar="PATH", help="also write the sidecar"
    )
    p.set_defaults(func=cmd_summarize)


def cmd_summarize(args) -> int:
    from spintrack.quality import format_summary, summary_from_dat, write_sidecar

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    quality = summary_from_dat(args.dat, args.fps)
    print(format_summary(quality))
    if args.json:
        write_sidecar(args.json, quality, {"dat": str(args.dat)})
    return 0


def _add_map(sub) -> None:
    p = sub.add_parser("map", help="render a saved surface map as an image")
    p.add_argument("map", help="a spintrack .npz map or a FicTrac sphere-map .png")
    p.add_argument(
        "--out", default=None, metavar="PATH", help="output image (default: <map>.png)"
    )
    p.add_argument(
        "--layout",
        choices=("grid", "cube"),
        default="grid",
        help="the map's own equal-area rectangle, or an unfolded cube",
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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("spintrack")
    path = Path(args.map)
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            shape = z["mean"].shape
    else:  # a FicTrac template: convert it on its own grid
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
    mean, weight = load_map(path, shape)
    out = Path(args.out) if args.out else path.with_suffix(".png")
    cv2.imwrite(str(out), render_map(mean, weight, args.w_min, args.layout))
    log.info(
        "%s: %dx%d cells, %.0f%% seen -> %s",
        path,
        shape[0],
        shape[1],
        100.0 * float(np.mean(weight >= args.w_min)),
        out,
    )
    return 0


def _add_calibrate(sub) -> None:
    p = sub.add_parser("calibrate", help="interactive ball / animal-frame calibration")
    p.add_argument("config", help="config.txt to read and update (needs vfov)")
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

        write_c2a_angles(args.config, *args.c2a_angles)
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


def _c2a_provenance(cfg) -> dict:
    """Where the camera-to-animal transform came from, for the sidecar."""
    source = cfg.c2a_source()
    angles = cfg.extra.get("c2a_angles") if cfg.c2a_src == "sliders" else None
    return {
        "source": source,
        "identity": source == "c2a_r" and not any(cfg.c2a_r),
        "angles": list(angles) if angles else None,
    }


def _c2a_line(prov: dict) -> str:
    if prov["identity"]:
        return "identity (explicit)"
    if prov["angles"]:
        el, az, tw = prov["angles"]
        return f"from config (sliders: elevation {el:g}, azimuth {az:g}, twist {tw:g})"
    if prov["source"] == "c2a_r":
        return "from config"
    return f"from config ({prov['source']})"


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
    if args.load_map:
        cfg.sphere_map_fn = args.load_map
    if args.frozen_map:
        cfg.map_frozen = True
    src_spec = args.src if args.src is not None else cfg.src_fn
    if not src_spec:
        log.error("no source: set src_fn in the config or pass --src")
        return 2
    if cfg.c2a_source() is None:
        log.error(
            "no camera-to-animal transform (c2a_r): the lab-frame and forward/side "
            "columns would be\ncamera-frame values in disguise. Fix: spintrack "
            "calibrate CONFIG --c2a-angles ELEV AZIM TWIST\n(a camera directly behind "
            "the animal, level with the ball, is 0 180 0), or write\n"
            "`c2a_r : { 0, 0, 0 }` to use the identity explicitly."
        )
        return 2
    if not str(src_spec).isdigit():
        src_path = Path(src_spec)
        if not src_path.is_absolute():
            src_path = config_path.parent / src_path
        src_spec = str(src_path)
    prepared = None
    if not cfg.has_ball() or cfg.vfov is None:
        from spintrack.autofit import prepare_config
        from spintrack.detect import DetectionError

        try:
            prepared = prepare_config(cfg, src_spec)
        except (DetectionError, ValueError) as exc:
            log.error("%s", exc)
            return 2
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

    if args.no_illumination:
        cfg.illumination = False
    tuned = args.all_pixels or args.no_scale_check or args.map_projection
    params = TrackParams() if tuned else None
    if params is not None:
        params.max_pixels = None if args.all_pixels else params.max_pixels
        params.scale_check_stride = (
            0 if args.no_scale_check else params.scale_check_stride
        )
        params.map_projection = args.map_projection or params.map_projection
    debug_video = args.debug_video
    if debug_video is None and cfg.save_debug:
        debug_video = "auto"
    if debug_video == "auto":
        debug_video = str(out_path.with_name(out_path.stem + "-debug.mp4"))
    refined_out = args.refine_out
    if args.refine > 0 and refined_out is None:
        refined_out = str(out_path.with_name(out_path.stem + "-refined.dat"))
    summary_out = None
    if not args.no_summary:
        summary_out = str(out_path.with_name(out_path.stem + "-summary.json"))
    provenance = {
        "config": str(config_path),
        "source": str(src_spec),
        "vfov": (
            prepared.vfov.report()
            if prepared is not None and prepared.vfov is not None
            else {"value": cfg.vfov, "source": "config"}
        ),
        "ball": (
            prepared.report()
            if prepared is not None
            else {"source": "config", "roi_c": cfg.roi_c, "roi_r": cfg.roi_r}
        ),
        "c2a": _c2a_provenance(cfg),
    }
    checks = {"c2a_r": _c2a_line(provenance["c2a"])}
    if prepared is not None:
        checks["ball"] = prepared.line()
        if prepared.vfov is not None:
            checks["vfov"] = prepared.vfov.line()
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
            save_map=args.save_map,
            debug_video=debug_video,
            refine_sweeps=args.refine,
            refined_out=refined_out,
            summary_out=summary_out,
            provenance=provenance,
            checks=checks,
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
    if stats.refine:
        log.info("refined: %s -> %s", stats.refine, refined_out)
    if debug_video:
        log.info("debug video: %s", debug_video)
    if stats.quality is not None:
        from spintrack.quality import format_summary

        log.info("%s", format_summary(stats.quality))
        if summary_out:
            log.info("summary: %s", summary_out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spintrack")
    parser.add_argument(
        "--version", action="version", version=f"spintrack {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    _add_run(sub)
    _add_map(sub)
    _add_calibrate(sub)
    _add_summarize(sub)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
