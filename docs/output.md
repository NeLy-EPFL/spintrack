# Output files

A run writes one folder, `NAME_spintrack` next to the video or `--out DIR` (the [guide](guide.md#run) has the details):

| file | content |
|---|---|
| `tracks.parquet` | the records, one row per tracked frame; a dropped frame has no row |
| `summary.json` | run quality and provenance |
| `log.txt` | the lines the run printed, warnings included |
| `config.toml` | the config as run, with the command line's changes and the ball if the run detected it; `spintrack run` on it, with `--out` another folder and the run's `--two-pass` and `--max-frames`, repeats the run |
| `debug.mp4` | with `--debug-video`: the annotated video ([guide](guide.md#the-debug-video)) |
| `map.npz` | with `--save-map`: the final surface map, for `--load-map` or `spintrack map` |

## Frames and units

The camera frame has x to the right of the image, y down and z along the optical axis. The lab frame is the animal's: x forward, y to its right, z down. The config's camera position (`camera.position_deg` or `camera.rotation`) takes one to the other.

Rotations are right-handed rotation vectors (axis times angle) in radians. Distances are in ball radii, which equal radians of ball rotation. Multiply by the ball's radius for a distance, and per-frame values by the frame rate for a rate:

- `pos_x` in mm = `pos_x` * radius in mm; likewise `pos_y`, `forward_total`, `side_total`.
- speed in mm/s = `speed` * radius in mm * frames per second.
- turning in deg/s = -`dr_lab_z` * frames per second * 180 / pi.

The row after dropped frames spans the gap, so divide by its `delta_ts` (ms) rather than multiply by the frame rate.

## The 25 columns

In FicTrac's order, which is also that of the streamed line; columns are numbered from 1, as in FicTrac's documentation, and named as in `tracks.parquet` (`spintrack.io.records.COLUMNS`).

| col | name | content | unit | frame, sign |
|---|---|---|---|---|
| 1 | `frame` | source frame index, from 0 | | counts dropped frames too |
| 2-4 | `dr_cam_x`, `_y`, `_z` | rotation since the last tracked frame | rad | camera |
| 5 | `err` | weighted mean squared photometric residual | | lower is better |
| 6-8 | `dr_lab_x`, `_y`, `_z` | rotation since the last tracked frame | rad | lab: forward walking is +y, a step to the right is -x, a right turn is -z |
| 9-11 | `r_cam_x`, `_y`, `_z` | absolute orientation | rad | camera, 0 at the first frame |
| 12-14 | `r_lab_x`, `_y`, `_z` | absolute orientation, `C R_cam C^T` with `C` the camera-to-lab rotation | rad | lab, 0 at the first frame |
| 15-16 | `pos_x`, `pos_y` | integrated fictive position | ball radii | x along the initial heading, y to its right |
| 17 | `heading` | integrated heading, [0, 2 pi) | rad | grows as the animal turns right |
| 18 | `direction` | direction of motion relative to the heading, [0, 2 pi) | rad | 0 forward, pi/2 to the right |
| 19 | `speed` | forward and sideways motion combined | rad/frame | |
| 20-21 | `forward_total`, `side_total` | integrated forward and sideways motion, ignoring heading | ball radii | side positive to the right |
| 22 | `timestamp` | video position, or capture time for a camera | ms | |
| 23 | `seq` | frames since tracking last (re)started | | |
| 24 | `delta_ts` | time since the last tracked frame, 0 on the first | ms | |
| 25 | `wall_ms` | wall-clock time the frame was read | ms since midnight | |

A tracking reset (after `tracking.max_bad_frames` losses, or when tracking starts) restarts `seq`, the heading and the position; `forward_total` and `side_total` carry on, as in FicTrac. A frame relocalized by the global search reports zero motion.

In Python, `FrameResult.forward`, `side` and `turn` are `dr_lab_y`, `-dr_lab_x` and `-dr_lab_z`.

## Parquet

`tracks.parquet` has the 25 columns under the names above, `frame` and `seq` as int64 and the rest as float64 at full precision. Its key-value metadata holds the version (`spintrack`), the columns' units as JSON (`units`) and the run's provenance as JSON (`provenance`):

```python
import json
import polars as pl

df = pl.read_parquet("sample_spintrack/tracks.parquet")
units = json.loads(pl.read_parquet_metadata("sample_spintrack/tracks.parquet")["units"])
```

The file is written when the run ends, also when it ends in an error or is stopped by Ctrl-C, SIGTERM or SIGHUP; a run killed with SIGKILL leaves none.

## The streamed line

`--udp`, `--tcp`, `--serial` and `--print` send each record as FicTrac's 25-field line, described in [fictrac.md](fictrac.md#the-streamed-line).

## The summary

`summary.json` holds the version (`spintrack`) and two sections:

- `provenance`: the config and source, where the field of view (`vfov`), the ball and the camera position came from, and `geometry`: the ball's image circle (`center_px`, `radius_px`), the window size, and the follower's record (`radius_measured_px`, the moves as `[start, end, px]` in `episodes`, `stopped_at`).
- `quality`: `n_frames`, `n_tracked`, `n_dropped`; cost percentiles (`cost_median`, `cost_p90`, `cost_p99`) and solver iterations; `sources`, how many frames were solved against the map, the previous frame, by global search or by a reset; `map_coverage`, the fraction of the ball mapped; the hard-tracking `episodes`; and `checks`, the terminal summary's lines.

## Differences from FicTrac's output

Path integration is a line-for-line port of FicTrac's (given FicTrac's columns 6-8, it reproduces its columns 15-21), and the frame, sequence and delta-timestamp columns mean the same. What differs:

- **Column 5** is a weighted mean squared photometric residual, not FicTrac's matching error; compare it only within a run.
- **Columns 2-4 and 9-11** are in the true camera frame. FicTrac writes its tracking-window frame (z toward the ball's center) under the camera label.
- **Columns 6-8**, and everything integrated from them, therefore differ from FicTrac's by a rotation through the ball's off-axis angle; FicTrac's are the wrong ones ([fictrac.md](fictrac.md#the-frame-bug)).
- **Columns 12-14** are the absolute orientation in the lab frame, 0 at the first frame. FicTrac composes `c2a_r` into it, so its columns start at `c2a_r`.
- **Column 22** is the video position on every row, where FicTrac writes the epoch time on the first. For a live camera it is a monotonic clock in ms, not epoch time.
