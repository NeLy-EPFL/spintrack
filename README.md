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
spintrack run config.txt --two-pass      # map the ball, then track it again from the map
spintrack calibrate config.txt --auto    # find the ball (and vfov) in the recording
spintrack calibrate config.txt --c2a-angles 0 180 0   # camera behind the animal
spintrack calibrate config.txt           # click rim points, ignore regions, animal axes
spintrack summarize camera.dat --fps 100 # run quality of an existing .dat
spintrack map ball.npz --layout cube     # look at the ball's surface map
```

`--debug-video` writes an annotated video (ball orientation, tracking window, map,
illumination, path);
it encodes H.264 with PyAV's bundled FFmpeg, so no system `ffmpeg` is needed.
`--refine N` re-estimates every frame offline against the map built from the whole
recording, which removes drift and recovers dropped frames. `--save-map` / `--load-map`
(also FicTrac sphere-map PNGs) carry a surface map between runs; `--frozen-map` keeps it
fixed. `--two-pass` maps the ball in a throwaway first pass over the recording and tracks
it again from the finished map, so the opening frames see a whole ball instead of one
visible cap; it fixes the cold start, not the drift that accumulates afterwards, which is
what `--refine` is for. The two compose. If the ball moved in its holder, the second pass
also places the window on the trajectory the first pass measured, without the online
follower's delay. `spintrack map` renders a saved map as a picture,
either as a Lambert equal-area rectangle or unfolded onto a cube.

The surface map is an equi-angular cubemap; `--map-projection equal_area` switches to
FicTrac's Lambert cylindrical grid, which is what a like-for-like comparison against
FicTrac wants. Maps convert between the two on load.

### The lighting stays out of the ball's texture

A ball sitting in a holder is darker near the holder, and that darkness belongs to the rig,
not to the ball. Because the lamps do not turn with the ball, spintrack can tell the two
apart: it measures what stays put in the camera frame and subtracts it, so the surface map
holds texture rather than a shadow smeared around the sphere. `--no-illumination` turns it
off. See [docs/algorithm.md](docs/algorithm.md#static-illumination).

### The geometry does not have to be hand-measured

A config without `roi_circ`/`roi_c` gets its ball found in the recording, and `vfov : auto`
gets the field of view fitted from the photometric cost. `spintrack calibrate CONFIG --auto`
does the same and writes the numbers back, so a headless machine never needs the click-based
calibrator. Detection refuses rather than guessing when it is not sure: a ball radius that is
5% wrong makes every reported speed about 7% wrong. `vfov` is only fitted when the cost has a
real minimum; where the ball is small in the frame the cost is flat and the run says so,
because there any value in the flat range tracks identically.

`c2a_r` is now required: without it the lab-frame and forward/side columns would be
camera-frame values wearing a different name. Write it with `--c2a-angles`, or
`c2a_r : { 0, 0, 0 }` to say "camera frame is the animal frame" on purpose.

### Every run says how it went

Each run ends with a quality block and writes `<out>-summary.json`: cost percentiles,
dropped frames, solver effort, map coverage, and the frame ranges where tracking was
measurably harder than in the rest of the same run. It also reports two checks that need no
ground truth - whether the ball's assumed radius is consistent with what the inner and outer
parts of the window each imply, and whether the ball moved in its holder mid-recording. If it
did, the tracking window follows it instead of quietly turning the movement into rotation.
`spintrack summarize X.dat` produces the same summary from an existing file (without the
solver-effort part, which a `.dat` does not carry).

```python
from spintrack import Config, Tracker
from spintrack.io.sources import VideoSource

cfg = Config.load("config.txt")
src = VideoSource("trial.mp4")
tracker = Tracker(cfg, src.width, src.height)
for frame in src:
    # None when the frame is dropped
    result = tracker.process_frame(frame.image, frame.ts_ms)
    if result is not None:
        print(result.heading, result.w_lab)
```

Live cameras: feed grayscale frames from your own capture code (pypylon, PySpin, OpenCV, ...)
to `Tracker.process_frame`. spintrack does not bundle camera SDKs.

## How it works

FicTrac binarizes the tracking window and searches a rotation that best matches a binary
surface map with a derivative-free optimiser. spintrack instead normalizes the window
photometrically, keeps a floating-point surface map, and aligns the two with Gauss-Newton
iterations using analytic derivatives (sub-pixel, a few iterations per frame), with an
anti-aliased window, robust weights against occluders, a coarse-to-fine fallback for
saccades, an estimate of the rig's static illumination that keeps it out of the map, a
window that follows a ball moving in its holder, and an optional global relocalisation.
Details in [docs/algorithm.md](docs/algorithm.md).

## Accuracy and speed

Synthetic scenes with exact ground truth (1000 frames each, 100 fps; FicTrac 2.1.2 run as a
black box on the same videos). Median per-frame rotation error in degrees; full tables in
[docs/benchmark.md](docs/benchmark.md).

| scene            | FicTrac | spintrack |
|------------------|--------:|----------:|
| clean fly walk   |   0.243 |     0.019 |
| lab-like, q12    |   0.165 |     0.046 |
| low contrast     |   0.346 |     0.029 |
| occluded by legs |   0.275 |     0.036 |
| saccades         |   0.242 |     0.017 |
| motion blur      |   0.304 |     0.092 |
| fine speckle     |  90.266 |     0.020 |

Tracking time per frame at the lab's settings (120x120 window, one core): FicTrac ~7 ms,
spintrack ~2.5 ms with every check on, ~1.7 ms with the ball follower and the radius check
off (~2 ms and ~1.4 ms at FicTrac's default 60x60 window). On six real 60 s trials
(1600x1008 HEVC, 100 fps) spintrack agrees with FicTrac to a median 0.1 deg per frame and
runs at ~240 fps including decoding (~3.5 ms of tracking per frame, of which the silhouette
measurement that follows a moving ball is about 1.5).

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
