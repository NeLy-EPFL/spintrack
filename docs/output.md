# Output files

A run writes one folder, `NAME_spintrack` next to the video or `--out DIR` (the [guide](guide.md#run) has the details):

| file | content |
|---|---|
| `tracks.parquet` | the records, one row per tracked frame; a dropped frame has no row |
| `summary.json` | run quality and provenance |
| `log.txt` | the lines the run printed, warnings included |
| `config.toml` | the config as run, with the command line's changes and the ball if the run detected it; `spintrack run` on it, with `--out` another folder and the run's `--two-pass` and `--max-frames`, repeats the run |
| `debug.mp4` | with `--debug-video`: the annotated video ([guide](guide.md#the-debug-video)) |
| `map.npz` | with `--save-map`: the final surface map, for `tracking.initial_map`, or `spintrack.maps.render_map` to draw it |

## Frames and units

The camera frame has x to the right of the image, y down and z along the optical axis. The lab frame is the animal's: x forward, y to its left, z up, so positive turning is to the left (counterclockwise seen from above), as for any right-handed frame with z up. The config's camera position (`camera.position_deg` or `camera.rotation`) takes one to the other. FicTrac's lab frame is x forward, y right, z down: the streamed records are converted to it ([below](#differences-from-fictracs-output)).

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
| 6-8 | `dr_lab_x`, `_y`, `_z` | rotation since the last tracked frame | rad | lab: the ball turns under the animal, so forward walking is -y, a step to the left is +x, a left turn is -z |
| 9-11 | `r_cam_x`, `_y`, `_z` | absolute orientation | rad | camera, 0 at the first frame |
| 12-14 | `r_lab_x`, `_y`, `_z` | absolute orientation, `C R_cam C^T` with `C` the camera-to-lab rotation | rad | lab, 0 at the first frame |
| 15-16 | `pos_x`, `pos_y` | integrated fictive position | ball radii | x along the initial heading, y to its left |
| 17 | `heading` | integrated heading, [0, 2 pi) | rad | grows as the animal turns left |
| 18 | `direction` | direction of motion relative to the heading, [0, 2 pi) | rad | 0 forward, pi/2 to the left |
| 19 | `speed` | forward and sideways motion combined | rad/frame | |
| 20-21 | `forward_total`, `side_total` | integrated forward and sideways motion, ignoring heading | ball radii | side positive to the left |
| 22 | `timestamp` | video position, or capture time for a camera | ms | |
| 23 | `seq` | frames since tracking last (re)started | | |
| 24 | `delta_ts` | time since the last tracked frame, 0 on the first | ms | |
| 25 | `wall_ms` | wall-clock time the frame was read | ms since midnight | |

A tracking reset (after `tracking.max_bad_frames` losses, or when tracking starts) restarts `seq`, the heading and the position; `forward_total` and `side_total` carry on, as in FicTrac. A frame relocalized by the global search reports zero motion.

In Python, `FrameResult.forward`, `side` and `turn` are `dr_lab_y`, `-dr_lab_x` and `-dr_lab_z`.

## The ball's position

FicTrac assumes the ball's center never moves. Real balls sink, jerk and drift in their holders, so spintrack's ball follower measures the ball's outline on every frame and moves the tracking window with it ([algorithm.md](algorithm.md#ball-follower)). `tracks.parquet` saves that measurement after the 25 columns. The streams don't carry these columns, because they are only final once the run ends.

| col | name | content | unit |
|---|---|---|---|
| 26-27 | `ball_x_px`, `ball_y_px` | center of the ball's outline in the image | px |
| 28 | `ball_seen` | whether the outline was measured on this frame, rather than filled in | |
| 29 | `window_offset` | distance from the tracking window to the ball: about the error it puts in the orientation | ball radii (= rad) |

**`ball_x_px`, `ball_y_px`** are in the coordinates of `ball.rim`: x right, y down, with the top-left pixel's center at (0.5, 0.5), so subtract 0.5 to draw them with OpenCV. They are the best estimate over the whole run, not the raw per-frame fits. The fits are interpolated across frames without one, cleaned by a 5-frame median and smoothed by a Gaussian whose width shrinks through jerks. They are measured as displacements from where the ball sat over its first fits and added to the config's ball circle, so they start at the config's center. Keep in mind:

- They are NaN before the first fit and after the last. The follower spends at least the first 30 frames measuring the ball's radius, so those are always NaN. Every row is NaN if the follower never locked onto the outline.
- Legs crossing the outline pull the fit by a few pixels for tens of frames, and the smoothing keeps such slow pulls. A small excursion while the animal is moving is more likely its legs than the ball.
- Only motion across the image is measured. Motion toward the camera would change only the ball's apparent size, which is not tracked.
- For a distance, multiply by the ball's radius in mm over `summary.json`'s `geometry.radius_px`. This holds for motion across the image only.

**`window_offset`** is a quality column for a failure that `err` cannot see. A window `d` pixels off the ball's center sees the ball's texture shifted by `d`. The fit explains the shift as a rotation of about `d / r`, and the rotated ball matches the image well, so `err` stays low. `window_offset` is `d / r` on every frame: about how many radians the orientation columns (9-14) are off on that frame. In a synthetic test the true error was 10-25% larger. The error goes away once the window is back on the ball, so the per-frame rotations (2-4, 6-8) carry it in while the offset grows and back out as it shrinks.

How to read it:

- At rest it is a few thousandths.
- The follower leaves the window where it is until the ball is 10 px or 2% of the radius away, whichever is larger, because legs pull the outline fit by about that much. Below that size, which is 0.04 for a 259 px ball, the column may overstate the error: the ball may have moved, or the legs may have pulled the fit.
- Every move the online follower confirms starts with a peak of about that size, since the window only leaves once the ball is that far. On `examples/ball_drop`, a 259 px ball that sinks 100 px in three jerks, it peaks at 0.05 at the first jerk.
- `--two-pass` plans the window from the whole first pass and has no such lag. On `ball_drop` it stays under 0.007. Rerun with it, or drop the frames above the error you can accept (0.05 rad is about 3 deg).
- It is NaN wherever the ball's position is.

## Parquet

`tracks.parquet` has the 25 columns under the names above, then the four for the ball's position. `frame` and `seq` are int64, `ball_seen` boolean and the rest float64 at full precision. Its key-value metadata holds the version (`spintrack`), the columns' units as JSON (`units`) and the run's provenance as JSON (`provenance`):

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

- `provenance`: the config, the command line's `overrides` and the source; where the field of view (`vfov`), the ball and the camera position came from (`config`, `command line`, `detected`, `estimated` or `assumed`, with what the detection or the animal's silhouette measured); and `geometry`: the ball's image circle (`center_px`, `radius_px`), the window size, and the follower's record (`radius_measured_px`, the moves as `[start, end, px]` in `episodes`, `stopped_at`).
- `quality`: `n_frames`, `n_tracked`, `n_dropped`; cost percentiles (`cost_median`, `cost_p90`, `cost_p99`) and solver iterations; `sources`, how many frames were solved against the map, the previous frame, by global search or by a reset; `map_coverage`, the fraction of the ball mapped; the hard-tracking `episodes`; and `checks`, the terminal summary's lines.

## Differences from FicTrac's output

Path integration is a line-for-line port of FicTrac's (given FicTrac's columns 6-8, it reproduces its columns 15-21, mirrored), and the frame, sequence and delta-timestamp columns mean the same. What differs:

- **The lab frame** is z up, FicTrac's z down: columns 7-8 and 13-14 have the opposite sign, and the path is FicTrac's mirrored (columns 16, 17, 18 and 21: y, heading and direction to the left, sideways motion positive to the left). The streams (`--udp`, `--tcp`, `--serial`, `--print`) send FicTrac's signs, so FicTrac's clients work unchanged; `spintrack.io.records.to_fictrac` converts a record either way.

- **Column 5** is a weighted mean squared photometric residual, not FicTrac's matching error; compare it only within a run.
- **Columns 2-4 and 9-11** are in the true camera frame. FicTrac writes its tracking-window frame (z toward the ball's center) under the camera label.
- **Columns 6-8**, and everything integrated from them, therefore differ from FicTrac's by a rotation through the ball's off-axis angle; FicTrac's are the wrong ones ([fictrac.md](fictrac.md#the-frame-bug)).
- **Columns 12-14** are the absolute orientation in the lab frame, 0 at the first frame. FicTrac composes `c2a_r` into it, so its columns start at `c2a_r`.
- **Column 22** is the video position on every row, where FicTrac writes the epoch time on the first. For a live camera it is a monotonic clock in ms, not epoch time.
