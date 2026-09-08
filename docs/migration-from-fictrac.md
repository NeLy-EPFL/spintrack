# Migrating from FicTrac

spintrack reads FicTrac configuration files and writes FicTrac's output format, so most
pipelines only need the executable swapped.

## What will not match

The two trackers do not report the same amount of turning. On six 60 s trials of one lab
rig, spintrack reads 1.5 to 4% less turning than FicTrac on the five the animal walked
through, and 9% less on the sixth, where it barely moved; forward motion agrees to about a
percent, and sideslip to within the several percent the comparison itself can resolve.
Neither tracker is known to be the right one - that needs ground truth the recordings do
not have - and the difference does not reproduce on synthetic video: at this rig's geometry
and its animals' own speeds, both trackers land within 1.2% of exact truth and the small
residual difference runs the other way. `docs/verification.md` has the numbers and
what has been ruled out.

The practical consequence: do not pool turning across the switch. Re-run the earlier
recordings through spintrack rather than comparing its output against FicTrac numbers from
before, and keep one tracker per dataset that will be analyzed together. Integrated
heading inherits the difference, so the two fictive paths separate over a trial: on those
six, heading ends 0.2 to 19 degrees apart and the endpoint 0.1 to 6% apart after 60 s.

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
| socket output (`sock_port`) | honored; also `--udp host:port`, `--tcp`   |
| serial output (`com_port`)  | honored; also `--serial PORT[:BAUD]`       |
| `configGui config.txt`      | `spintrack calibrate config.txt`            |
| `save_debug: y`             | honored; also `--debug-video [PATH]`       |
| `sphere_map_fn`             | honored (.png template or spintrack .npz)  |
| (no equivalent)             | `--refine N` offline re-estimation          |
| (no equivalent)             | `--save-map`, `--load-map`, `--frozen-map`  |
| (no equivalent)             | `--load-illumination` the rig's lighting    |
| (no equivalent)             | `--two-pass` map-then-track in one command  |
| equal-area sphere-map grid  | equi-angular cubemap (`--map-projection`)   |

## Configuration

All documented FicTrac keys are accepted (`vfov`, `fisheye`, `q_factor`, `roi_circ`, `roi_c`,
`roi_r`, `roi_ignr`, `c2a_*`, `sock_*`, `com_*`, `src_fps`, `max_bad_frames`, `opt_do_global`,
`thr_win_pc`, ...), plus the lab fork's `accumulate_map`. YAML and TOML files with the same
keys work too. Keys spintrack does not need (`thr_ratio`, `opt_max_evals`, `opt_tol`,
`opt_max_err`, `q_factor` beyond the window size) are accepted and ignored. spintrack adds
`map_frozen: y` (never update a map loaded through `sphere_map_fn`) and `illumination: n`
(stop separating the rig's static lighting from the ball's texture).

Three differences worth knowing before you copy a FicTrac config across:

- **`vfov` may be `auto`**, or left out, and is then fitted from the recording. The fit is
  only used when the photometric cost has a real minimum over 1-120 degrees; when it does
  not, either the ball is small enough in the frame that the choice does not change the
  rotation scale (the run reports the flat range it picked from) or the fit refuses and asks
  for a number. FicTrac always required one.
- **`roi_circ` / `roi_c` / `roi_r` may be left out**, and the ball is then found in the first
  few hundred frames. A config that does describe a ball is never overridden; the detection
  still runs as a check and the difference goes into the summary.
- **`c2a_r` is required.** FicTrac silently used the identity when no camera-to-animal
  transform was given, which makes the lab-frame columns look meaningful when they are just
  the camera-frame ones. `spintrack run` refuses instead. `c2a_r : { 0, 0, 0 }` is a valid
  answer - it is the identity, stated on purpose - and `spintrack calibrate CONFIG
  --c2a-angles ELEV AZIM TWIST` writes one from where the camera sits.

Each run also writes `<out>-summary.json` next to the `.dat` (disable with `--no-summary`).
It holds the cost percentiles, the map coverage, the stretches where tracking was harder than
usual, and the provenance of every geometric number the run used. Nothing about the `.dat`
itself changes: it has no header and the 25 columns are unmoved.

How the keys map onto spintrack's solver:

- `q_factor`: tracking window side is `10 * q_factor` pixels, as in FicTrac. The solve runs
  on at most ~4000 window pixels by default (`--all-pixels` to use every pixel).
- `thr_win_pc`: size of the local photometric normalization window (FicTrac used it for
  adaptive thresholding).
- `opt_bound`: maximum accepted rotation per frame (rad).
- `opt_do_global`: enables relocalization against the map after tracking is lost.
- `accumulate_map: n`: keep only the currently visible surface (plus a one-cell margin),
  reproducing the lab fork's behavior; the default accumulates the whole surface.
- `max_bad_frames`: reset tracking after this many consecutive untrackable frames.
- `sphere_map_fn`: a FicTrac sphere-map PNG (converted on import) or a spintrack `.npz`
  written by `--save-map`; the first frame is localized globally against it. A spintrack
  `.npz` also carries the static illumination field, which describes the rig rather than
  the ball, so a map saved from one trial gives the next one a head start.
- `illumination` (spintrack-only, on by default): estimate the camera-fixed illumination -
  the holder's shadow above all - and keep it out of the ball's surface map. FicTrac has
  no equivalent; `illumination: n` or `--no-illumination` turns it off.
- `illumination_fn` (spintrack-only), set by `--load-illumination PATH`: a map `.npz` to
  take illumination fields from, without its map, so the rig's lighting can be carried to
  a new ball. The field describes the rig and takes a few hundred frames to converge, so a
  short recording is better off borrowing one; `sphere_map_fn` also brings the fields, but
  only with the surface map of whatever ball they were saved beside.
- `--map-projection` (spintrack-only, `cube` by default): how the surface map tiles the
  sphere. `equal_area` is FicTrac's Lambert cylindrical grid, and is what a like-for-like
  comparison against FicTrac wants; `cube` is an equi-angular cubemap with the same number
  of cells, which removes the 8.6-degree cells the cylindrical grid puts at its poles. See
  `docs/algorithm.md`. Maps convert between the two grids on load, so a FicTrac template
  and a map saved by either projection all still load.
- `save_debug`: write the annotated debug video next to the `.dat`. `vid_codec` picks
  the encoder; `h264` (the default), `hevc` and `vp9` go through PyAV, whose wheels bundle
  an FFmpeg with those encoders. The `opencv-python` one does not, so `cv2.VideoWriter`
  would quietly write MPEG-4 Part 2 instead, which browsers and most editors cannot play.
  The fourcc codecs OpenCV does handle (`mp4v`, `xvid`, `mjpg`, `raw`) still use it.

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
  tracking-window frame (z toward the ball center) and only labels them "camera"; the two
  coincide for a ball near the image center. Lab-frame columns are unaffected.

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
