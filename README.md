# spintrack

spintrack measures the 3D rotation of a trackball (spherical treadmill) under a tethered animal from a single camera and integrates the animal's fictive path. It is a successor to [FicTrac](https://github.com/rjdmoore/fictrac) (Moore et al. 2014): it reads FicTrac config files and writes FicTrac's 25-column `.dat`, so existing rigs and analysis code keep working. Instead of matching a thresholded window against a binary map, it aligns every frame photometrically against a floating-point map of the ball's surface, with sub-pixel Gauss-Newton steps in a Rust core. On synthetic scenes with exact ground truth its median per-frame error is 0.014-0.053 deg where FicTrac's is 0.17-0.50 deg, and at the lab's window size it needs about a third of FicTrac's time per frame.

## Install

spintrack is not on PyPI yet; wheels will come with the first release. Install from a clone, which builds the Rust extension and so needs a [Rust toolchain](https://rustup.rs) and Python 3.11 or newer:

```bash
git clone https://github.com/NeLy-EPFL/spintrack
cd spintrack
uv sync            # or, in an environment of your own: pip install .
```

Streaming over a serial port needs the `serial` extra (`uv sync --extra serial` or `pip install ".[serial]"`).

## Quick start

The repository includes a 10.5 s example: frames 850-1899 of trial ANXXX049_251125_Fly1_003, a fly filmed by a camera behind it, at half resolution. The fly walks and turns enough in it to map the whole ball.

```console
$ uv run spintrack run examples/sample/config.txt --debug-video
spintrack 0.1.0: examples/sample/sample.mp4 (800x504, 100 fps, 1050 frames)
done: 1050 frames, 0 dropped, 5.8 s (182 fps)
run quality: 1050 frames, 1050 tracked, 0 dropped
ball moved: no
radius: silhouette 258.4 px, config 259.8 px (-0.6%)
hard tracking: none
wrote sample.dat, sample.parquet, sample-summary.json, sample-debug.mp4 in examples/sample
```

For a rig of your own, `spintrack calibrate config.txt --src clip.mp4 --auto --c2a-angles 0 180 0` writes a config from a recording; the [user guide](docs/guide.md) explains each step.

## What you get

Each run writes, next to the config (or into `--out`), files named after the video or the config's `output_fn`:

- `NAME.dat`: FicTrac's 25 columns, one row per tracked frame.
- `NAME.parquet`: the same records with named columns and units.
- `NAME-summary.json`: run quality and the provenance of every geometric input.
- `NAME-debug.mp4` with `--debug-video`: the frame, the tracking window, the surface map and the path, for checking a run by eye.
- `NAME-map.npz` with `--save-map`: the ball's surface map, to start a later run from.

[docs/output.md](docs/output.md) lists every column with its unit, frame and sign.

## Python

```python
import spintrack

track = spintrack.track("examples/sample/config.txt")  # the run, without the files
df = track.to_pandas()  # needs pandas; columns by name: forward_total, heading, ...
print(track.quality.n_dropped, df[["forward_total", "side_total", "heading"]].tail())
```

For a live camera, feed frames from your own capture code to a `Tracker`. The config must already describe the ball, so calibrate on a recorded clip first.

```python
from spintrack import Config, Tracker

cfg = Config.load("config.txt")
tracker = Tracker(cfg, width, height)
for gray, ts_ms in frames():  # your camera SDK: 2-D uint8 images, timestamps in ms
    res = tracker.process_frame(gray, ts_ms)
    if res is not None:  # None: the frame could not be tracked
        print(res.forward, res.side, res.turn, res.heading)  # rad, this frame
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

- FicTrac `config.txt` files work as they are, as long as they set `c2a_r` (the camera-to-animal rotation), which spintrack requires. YAML and TOML files with the same keys work too.
- The `.dat` file and the UDP, TCP and serial streams have FicTrac's 25 columns and line format, and the path columns are integrated exactly as FicTrac does.
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
