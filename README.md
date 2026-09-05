# spintrack

Track the 3D rotation of a spherical treadmill ("trackball") under a tethered animal from a
single camera, and reconstruct the animal's fictive path. A pip-installable successor to
[FicTrac](https://github.com/rjdmoore/fictrac) (Moore et al. 2014): same configuration files
and output format, a compiled Rust core, and a solver that is 4-8x more precise on synthetic
ground truth and several times faster at the lab's settings.

## Install

```bash
uv add spintrack      # or: pip install spintrack
```

Prebuilt wheels for Linux, macOS and Windows (Python >= 3.11). No CMake, OpenCV, NLopt or
Boost to build.

## Use

```bash
spintrack run config.txt                 # FicTrac config, writes <video>-<timestamp>.dat
spintrack run config.txt --src ball.mp4  # override the source (video path or camera index)
spintrack run config.txt --udp 127.0.0.1:1111 --print
spintrack run config.txt --debug-video --refine 2 --save-map ball.npz
spintrack calibrate config.txt           # click rim points, ignore regions, animal axes
```

`--debug-video` writes an annotated video (ball orientation, tracking window, map, path).
`--refine N` re-estimates every frame offline against the map built from the whole
recording, which removes drift and recovers dropped frames. `--save-map` / `--load-map`
(also FicTrac sphere-map PNGs) carry a surface map between runs; `--frozen-map` keeps it
fixed.

```python
from spintrack import Config, Tracker
from spintrack.io.sources import VideoSource

cfg = Config.load("config.txt")
src = VideoSource("trial.mp4")
tracker = Tracker(cfg, src.width, src.height)
for frame in src:
    result = tracker.process_frame(frame.image, frame.ts_ms)  # None when the frame is dropped
    if result is not None:
        print(result.heading, result.w_lab)
```

Live cameras: feed grayscale frames from your own capture code (pypylon, PySpin, OpenCV, ...)
to `Tracker.process_frame`. spintrack does not bundle camera SDKs.

## How it works

FicTrac binarizes the tracking window and searches a rotation that best matches a binary
surface map with a derivative-free optimiser. spintrack instead normalizes the window
photometrically, keeps a floating-point surface map, and aligns the two with Gauss-Newton
iterations using analytic derivatives (sub-pixel, a few iterations per frame), with robust
weights against occluders, a coarse-to-fine fallback for saccades, and an optional global
relocalisation. Details in [docs/algorithm.md](docs/algorithm.md).

## Accuracy and speed

Synthetic scenes with exact ground truth (1000 frames each, 100 fps; FicTrac 2.1.2 run as a
black box on the same videos). Median per-frame rotation error in degrees; full tables in
[docs/benchmark.md](docs/benchmark.md).

| scene            | FicTrac | spintrack |
|------------------|--------:|----------:|
| clean fly walk   |   0.243 |     0.041 |
| lab-like, q12    |   0.165 |     0.049 |
| low contrast     |   0.346 |     0.046 |
| occluded by legs |   0.275 |     0.046 |
| saccades         |   0.242 |     0.034 |
| motion blur      |   0.304 |     0.098 |
| fine speckle     |  90.266 |     0.037 |

Tracking time per frame at the lab's settings (120x120 window, one core): FicTrac ~7 ms,
spintrack ~1 ms (~0.5 ms at FicTrac's default 60x60 window). On six real 60 s trials
(1600x1008 HEVC, 100 fps) spintrack agrees with FicTrac to a median 0.1 deg per frame and
runs at ~400 fps including decoding.

## Develop

```bash
uv sync               # builds the Rust extension via maturin (Python >= 3.11)
uv run pytest -q
uv run ruff check . && cargo fmt --check && cargo clippy --all-targets -- -D warnings
uv run --group bench python benchmarks/bench.py synth   # render the benchmark scenes
uv run --group bench python benchmarks/bench.py run --systems fictrac spintrack --pin-cpu 2
```

## License

Apache-2.0. spintrack shares no code with FicTrac; it reads FicTrac's `config.txt` files and
writes the same 25-column `.dat` output so existing pipelines keep working. See
[docs/migration-from-fictrac.md](docs/migration-from-fictrac.md).
