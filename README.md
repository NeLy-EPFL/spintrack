# spintrack

spintrack measures the 3D rotation of a trackball (spherical treadmill) under a tethered animal from a single camera and integrates the animal's fictive path. It is a successor to [FicTrac](https://github.com/rjdmoore/fictrac) (Moore et al. 2014): it streams FicTrac's 25-field records over UDP, TCP or serial, so existing closed-loop rigs keep working, and it writes the same records to Parquet with named columns. Instead of matching a thresholded window against a binary map, it aligns every frame photometrically against a floating-point map of the ball's surface, with sub-pixel Gauss-Newton steps in a Rust core. On synthetic scenes with exact ground truth its median per-frame error is 0.014-0.053 deg where FicTrac's is 0.17-0.50 deg, and at the lab's window size it needs about a third of FicTrac's time per frame.

## Install

Install spintrack as a command-line tool with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install --torch-backend auto git+https://github.com/NeLy-EPFL/spintrack
```

`--torch-backend auto` picks the PyTorch build that matches the machine's GPU driver (CUDA or ROCm), or the CPU build when there is none. PyTorch runs SAM 3, which finds the ball; its checkpoint (3.4 GB) downloads on the first run that needs it, without a login (see [finding the ball](docs/guide.md#find-the-ball)). uv fetches Python 3.14 if needed. Until wheels come with the first release, the install builds the Rust extension, so it needs a [Rust toolchain](https://rustup.rs), and while the repository is private, GitHub access (for instance through `gh auth login`). To update, run the same command with `--reinstall`; `uv tool upgrade` would not keep the PyTorch build.

Streaming over a serial port needs the `serial` extra: install `"spintrack[serial] @ git+https://github.com/NeLy-EPFL/spintrack"` instead. To work on spintrack itself, clone the repository and run `uv sync`.

## Quick start

The repository includes a 10.5 s example: frames 850-1899 of trial ANXXX049_251125_Fly1_003, a fly filmed by a camera behind it, at half resolution. The fly walks and turns enough in it to map the whole ball. A second example, `examples/ball_drop/`, shows the tracking window following a ball that sinks in its holder and comes back up. In a clone of the repository:

```console
$ spintrack examples/sample/sample.mp4 camera.azimuth_deg=180
preview: http://127.0.0.1:8300/?token=... (from another machine: ssh -L 8300:localhost:8300 HOST)
loading SAM 3 on cuda (the first use downloads 3.4 GB)
fitting the field of view: tracking 1000 frames at each candidate
spintrack 0.1.0: examples/sample/sample.mp4 (800x504, 100 fps, 1050 frames)
done: 1050 frames, 0 dropped, 1.9 s (559 fps)
run quality: 1050 frames, 1050 tracked, 0 dropped
ball: detected at (400.5, 371.5) r 260.8 px by SAM 3 (score 0.75), rim confirms 86% of its outline
vfov: 3.31 deg (fitted; not identifiable, the cost is flat over 1-11 deg, over which the rotation scale changes by 6%)
camera position: azimuth 180 from camera.azimuth_deg; elevation 0, twist -0.6 deg from where the animal stands (it stands on the ball's outline, where the elevation does not show and level is assumed; SAM 3 score 0.87)
walking: net 3.9 ball radii, 7 deg right of forward
ball moved: no
radius: silhouette 258.3 px, config 260.1 px (-0.7%)
hard tracking: none
wrote tracks.parquet, summary.json, log.txt, config.toml in examples/sample/sample_spintrack
```

`spintrack VIDEO` is short for `spintrack run VIDEO`. A run finds the ball (with SAM 3), fits the field of view and takes the camera's elevation and twist from where the fly stands; the summary says what it found. What a video cannot tell is the fly's front from its back, so say where the camera sits around the fly, as FicTrac also requires: `camera.azimuth_deg=180` behind it, `0` in front, `90` at its right, `-90` at its left. A rig's config (`-c`) can hold it. A [deeperfly](https://github.com/NeLy-EPFL/deeperfly) project tracks its ball with deeperfly's `[ball]` stage instead, which runs spintrack with the project's calibration and 3D pose and writes the ball in the project's coordinates; no azimuth is needed there. The link opens a live view of the run: the frame with the ball and the fly's trail over it, the tracking window, the surface map, the path and the speeds. It costs the run nothing until opened.

When the run gets something wrong, fix it while watching the tracking:

```bash
spintrack gui examples/sample/sample.mp4
```

The gui tracks a 10 s clip over and over while you drag the ball's outline, set the camera position, mask the fly and change the tracking parameters, and saves them as a config (`spintrack.toml` next to the video). Use that config for every video of the rig, and change any key for one run on the command line:

```bash
spintrack run session/*.mp4 -c spintrack.toml
spintrack run trial3.mp4 -c spintrack.toml tracking.window_px=80 --debug-video
```

The [user guide](docs/guide.md) explains each step.

## What you get

Each run writes one folder, `NAME_spintrack` next to the video (or `--out DIR`), where NAME is the video's name or the config's `output.name`:

- `tracks.parquet`: one row per tracked frame, FicTrac's 25 columns by name, then the ball's position in the image and how far the tracking window was from it, with units.
- `summary.json`: run quality and the provenance of every geometric input.
- `log.txt`: what the run printed.
- `config.toml`: the config as run, with the command line's changes and what the run found in the recording; `spintrack run` on it, with `--out` another folder and the run's `--two-pass` and `--max-frames`, repeats the run.
- `debug.mp4` with `--debug-video`: the frame, the tracking window, the surface map and the path, for checking a run by eye.
- `map.npz` with `--save-map`: the ball's surface map, to start a later run from.

[docs/output.md](docs/output.md) lists every column with its unit, frame and sign.

## Python

```python
import spintrack

track = spintrack.track("examples/sample/config.toml")  # the run, without the files
df = track.to_polars()  # columns by name: forward_total, heading, ...
print(track.quality.n_dropped, df.select("forward_total", "heading").tail())
```

For a live camera, feed frames from your own capture code to a `Tracker`. The config must already describe the ball, the field of view and the camera position, so track a recorded clip of the rig first and use the `config.toml` the run writes.

```python
from spintrack import Config, Tracker

cfg = Config.load("config.toml")
tracker = Tracker(cfg, width, height)
for gray, ts_ms in frames():  # your camera SDK: 2-D uint8 images, timestamps in ms
    res = tracker.process_frame(gray, ts_ms)
    if res is not None:  # None: the frame could not be tracked
        print(res.forward, res.side, res.turn)  # rad this frame; side, turn: left +
```

## Accuracy and speed

| | FicTrac 2.1.2 | spintrack |
|---|---|---|
| Median per-frame rotation error, synthetic scenes | 0.17-0.50 deg | 0.014-0.053 deg |
| Synthetic scenes lost | `speckle` (fine texture, 90 deg error) | none |
| Turning on real recordings, against an independent referee | about 2% high | within about 1% |
| Forward walking, same recordings | 0.6-1.4% high | within about 1% |
| Tracking time per frame, one core, 120x120 window | about 7 ms | about 2 ms |

The synthetic scenes have exact ground truth; the error ranges leave out `static` and `motion_blur`, and FicTrac was not run on four of the 24 scenes. On a ball rendered without any of spintrack's geometry code, the reported rotation is 1.0000 times the true one, within 0.05%. The real recordings are six 60 s trials of one rig, scored against a direct rigid-sphere image registration that shares no code with either tracker. End to end, including H.264 decoding of 1600x1008 frames, spintrack runs at about 400 fps on a 32-core workstation. Scenes, methods and full tables: [docs/benchmark.md](docs/benchmark.md).

## FicTrac compatibility

- The config is a TOML file with readable names in tables, checked on load: an unknown key is an error. [docs/fictrac.md](docs/fictrac.md) translates a FicTrac `config.txt` key by key.
- The UDP, TCP and serial streams have FicTrac's 25 fields, line format and signs, so FicTrac's clients work unchanged. `tracks.parquet` has the same columns by name, plus four for the ball's position, in spintrack's lab frame: x forward, y left, z up, where FicTrac's is y right, z down, so the lab rotations' y and z and the path's y, heading and sideways motion have the opposite sign. The path is integrated exactly as FicTrac does, mirrored. There is no `.dat` file.
- The rotation columns are in the true camera and lab frames. FicTrac writes its tracking-window frame as the camera frame, which inflates its turning wherever sideslip and turning are correlated, so re-track old FicTrac data instead of pooling it with spintrack's. See [docs/fictrac.md](docs/fictrac.md).

## Develop

```bash
uv sync                                    # builds the extension with maturin
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
uv run --group bench python benchmarks/bench.py --help   # see docs/benchmark.md
```

[docs/algorithm.md](docs/algorithm.md) describes the tracker for maintainers.

## License and citation

Apache-2.0. spintrack shares no code with FicTrac. If you use it, cite it with the metadata in [CITATION.cff](CITATION.cff), and cite FicTrac: Moore, R. J. D. et al. (2014), FicTrac: a visual method for tracking spherical motion and generating fictive animal paths, *J. Neurosci. Methods* 225, 106-119.
