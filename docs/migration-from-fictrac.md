# Migrating from FicTrac

spintrack reads FicTrac configuration files and writes FicTrac's output format, so most
pipelines only need the executable swapped.

## Install

```bash
uv add spintrack      # or: pip install spintrack
```

Prebuilt wheels for Linux, macOS and Windows; no CMake, OpenCV, NLopt or Boost to build.

## Commands

| FicTrac                     | spintrack                                   |
|-----------------------------|---------------------------------------------|
| `fictrac config.txt`        | `spintrack run config.txt`                  |
| `fictrac config.txt -s vid` | `spintrack run config.txt --src vid`        |
| socket output (`sock_port`) | honoured; also `--udp host:port`, `--tcp`   |
| serial output (`com_port`)  | honoured; also `--serial PORT[:BAUD]`       |
| `configGui config.txt`      | `spintrack calibrate config.txt`            |
| `save_debug: y`             | honoured; also `--debug-video [PATH]`       |
| `sphere_map_fn`             | honoured (.png template or spintrack .npz)  |
| (no equivalent)             | `--refine N` offline re-estimation          |
| (no equivalent)             | `--save-map`, `--load-map`, `--frozen-map`  |

## Configuration

All documented FicTrac keys are accepted (`vfov`, `fisheye`, `q_factor`, `roi_circ`, `roi_c`,
`roi_r`, `roi_ignr`, `c2a_*`, `sock_*`, `com_*`, `src_fps`, `max_bad_frames`, `opt_do_global`,
`thr_win_pc`, ...), plus the lab fork's `accumulate_map`. YAML and TOML files with the same
keys work too. Keys spintrack does not need (`thr_ratio`, `opt_max_evals`, `opt_tol`,
`opt_max_err`, `q_factor` beyond the window size) are accepted and ignored. spintrack adds
`map_frozen: y` (never update a map loaded through `sphere_map_fn`).

How the keys map onto spintrack's solver:

- `q_factor`: tracking window side is `10 * q_factor` pixels, as in FicTrac. The solve runs
  on at most ~4000 window pixels by default (`--all-pixels` to use every pixel).
- `thr_win_pc`: size of the local photometric normalization window (FicTrac used it for
  adaptive thresholding).
- `opt_bound`: maximum accepted rotation per frame (rad).
- `opt_do_global`: enables relocalisation against the map after tracking is lost.
- `accumulate_map: n`: keep only the currently visible surface (plus a one-cell margin),
  reproducing the lab fork's behaviour; the default accumulates the whole surface.
- `max_bad_frames`: reset tracking after this many consecutive untrackable frames.
- `sphere_map_fn`: a FicTrac sphere-map PNG (converted on import) or a spintrack `.npz`
  written by `--save-map`; the first frame is localised globally against it.
- `save_debug`: write the annotated debug video next to the `.dat`.

## Output

The `.dat` file has FicTrac's 25 columns, `", "` separated, 14 significant digits, and the
same semantics: frame counter (0-based, like FicTrac 2.1.2), delta rotation vectors in camera
and lab frames, absolute rotation vectors, integrated position and heading, movement direction
and speed, integrated forward/side motion, timestamps, sequence counter. Socket and serial
messages are the same line prefixed with `FT, `.

Two deliberate differences:

- Column 5 (error score) is spintrack's weighted mean squared residual in normalized
  intensity units, not FicTrac's binary-match error. Use it relatively (frame to frame).
- Columns 2-4 and 9-11 are true camera-frame vectors. FicTrac reports these in its
  tracking-window frame (z toward the ball centre) and only labels them "camera"; the two
  coincide for a ball near the image centre. Lab-frame columns are unaffected.

## Python API

```python
from spintrack import Config, Tracker
from spintrack.io.sources import VideoSource

cfg = Config.load("config.txt")
src = VideoSource("trial.mp4")
tracker = Tracker(cfg, src.width, src.height)
for frame in src:
    result = tracker.process_frame(frame.image, frame.ts_ms)
    if result is not None:
        print(result.heading, result.w_lab)
```

For live cameras, feed grayscale frames from your own capture code (pypylon, PySpin,
OpenCV, ...) to `Tracker.process_frame`; spintrack does not bundle camera SDKs.
