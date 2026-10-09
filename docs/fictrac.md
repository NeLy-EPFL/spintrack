# Coming from FicTrac

spintrack streams FicTrac's records in FicTrac's line format, so a closed-loop rig moves over by changing the command and translating its config. Files are another matter: the config is TOML, and the 25 columns of the `.dat` go to `tracks.parquet` under names. This page lists what maps to what and what differs.

## Commands

| FicTrac | spintrack |
|---|---|
| `fictrac config.txt` | `spintrack run config.toml`, or `spintrack run VIDEO -c config.toml`, or `spintrack run VIDEO` alone |
| `configGui config.txt` | `spintrack gui VIDEO` (a page in the browser, tracking while you set it up) |
| | `--debug-video`, `--udp`, `--tcp`, `--serial`, `--two-pass`, `--save-map`, `--out`, `--force`, and `--set KEY=VALUE` for any config key |

`spintrack run --help` lists every option.

## Config keys

spintrack does not read `config.txt`. Its config is a TOML file with the keys grouped in tables ([configuration](configuration.md)):

| FicTrac | spintrack | |
|---|---|---|
| `src_fn` | `video` | |
| `vfov` | `camera.vfov_deg` | |
| `fisheye` | `camera.fisheye` | |
| `src_fps` | `camera.fps` | used only when the video has no frame rate |
| `c2a_r` | `camera.rotation` | converted, as FicTrac's animal frame has y right and z down: `spintrack.calibrate.sliders.rotation_from_fictrac(c2a_r)`; or give `camera.position_deg` instead, or neither and let the run place the camera |
| `roi_circ` | `ball.rim` | `{ x1, y1, x2, y2, ... }` becomes `[[x1, y1], [x2, y2], ...]` |
| `roi_ignr` | `mask.ignore` | each polygon becomes a list of `[x, y]` the same way |
| `q_factor` | `tracking.window_px` | `10 * q_factor` |
| `opt_do_global` | `tracking.global_search` | |
| `max_bad_frames` | `tracking.max_bad_frames` | left out for -1 |
| `opt_bound` | `tracking.max_step_rad` | the largest rotation accepted in one frame |
| `thr_win_pc` | `tracking.norm_window` | sizes the local brightness normalization, not a thresholding window |
| `accumulate_map` (lab fork) | `tracking.forget_outside_view` | inverted: `accumulate_map : n` is `forget_outside_view = true` |
| `sphere_map_fn` | `tracking.initial_map` | a FicTrac sphere-map PNG loads and is converted |
| `output_fn` | `output.name` | names the output folder, `NAME_spintrack` |
| `save_debug` | `output.debug_video` | |
| `vid_codec` | `output.debug_codec` | |
| `sock_host`, `sock_port` | `stream.udp` | `"HOST:PORT"` |
| `com_port`, `com_baud` | `stream.serial` | `"PORT:BAUD"` |

`roi_c` and `roi_r` have no counterpart: give the rim points they were fitted from, or leave the ball out for the run to detect it. The calibration square's corners (`c2a_src`, `c2a_cnrs_xy`, ...) are replaced by the rotation they give, which FicTrac writes as `c2a_r` anyway. The other FicTrac keys (`do_display`, `save_raw`, `thr_ratio`, `thr_rgb_tfrm`, `opt_max_err`, `opt_max_evals`, `opt_tol`, `c2a_t`, `enh_cfg_disp`, `reconfig`) mean nothing to spintrack and are errors in its config, as is any unknown key. For example,

```
src_fn         : camera_F.mp4
vfov           : 2
q_factor       : 12
roi_circ       : { 1375, 993, 1246, 515, 722, 383, 410, 623 }
c2a_r          : { 0.895359, -0.895359, -1.378731 }
accumulate_map : n
```

becomes

```toml
video = "camera_F.mp4"

[camera]
vfov_deg = 2
rotation = [-1.510689, -1.510689, 0.981054]  # rotation_from_fictrac; position_deg [24, 0, 0]

[ball]
rim = [[1375, 993], [1246, 515], [722, 383], [410, 623]]

[tracking]
window_px = 120
forget_outside_view = true
```

Three things are handled differently:

- Without a camera position, the camera is placed from where the animal stands on the ball, behind it unless told otherwise, and every run checks the azimuth against the animal's net walking ([guide](configuration.md#the-camera-position)). `rotation = [0, 0, 0]` asks for the identity explicitly.
- Without a field of view, the run fits it from the recording.
- Without a ball, the run detects it in the recording.

## Records

### The `.dat` columns

spintrack writes no `.dat` file: `tracks.parquet` holds FicTrac's 25 columns in FicTrac's order, under these names ([output.md](output.md#the-25-columns) has their units, frames and signs), then four FicTrac has no equivalent for: the ball's position in the image and the tracking window's offset from it ([output.md](output.md#the-balls-position)):

| `.dat` column | `tracks.parquet` column |
|---|---|
| 1 | `frame` |
| 2-4 | `dr_cam_x`, `dr_cam_y`, `dr_cam_z` |
| 5 | `err` |
| 6-8 | `dr_lab_x`, `dr_lab_y`, `dr_lab_z` |
| 9-11 | `r_cam_x`, `r_cam_y`, `r_cam_z` |
| 12-14 | `r_lab_x`, `r_lab_y`, `r_lab_z` |
| 15-16 | `pos_x`, `pos_y` |
| 17 | `heading` |
| 18 | `direction` |
| 19 | `speed` |
| 20-21 | `forward_total`, `side_total` |
| 22 | `timestamp` |
| 23 | `seq` |
| 24 | `delta_ts` |
| 25 | `wall_ms` |

`spintrack.io.records.read_dat` loads a `.dat` FicTrac wrote as an array with these columns (`COLUMNS`).

### The streamed line

`--udp`, `--tcp`, `--serial` and `--print`, or the config's `stream.udp` and `stream.serial`, send each record as FicTrac does: the 25 fields separated by `", "`, floats with 14 significant digits, `frame` and `seq` as integers. Sockets and serial prefix it with `FT, `; every line ends with a newline. The record is converted to FicTrac's lab frame (y right, z down) on the way out (`spintrack.io.records.to_fictrac`), so FicTrac's clients read the signs they expect, and `spintrack.io.records.parse_row` reads one back.

## What differs and why

### The frame bug

FicTrac estimates the rotation in its tracking-window frame, whose z axis points at the ball's center, and writes it as the camera frame: upstream `src/Trackball.cpp:889` sets `dr_cam = dr_roi`, with the window-to-camera transform commented out. It then applies `c2a_r` to these window-frame vectors, so its lab columns, heading and path are rotated by the ball's off-axis angle (0.76 deg on the rig measured). spintrack applies the camera rotation to true camera-frame vectors.

The rotation leaks sideslip into turning wherever the two are correlated (0.83-0.97 on the trials measured). On six 60 s trials of one rig, scored against an independent referee, FicTrac read turning about 2% high, mostly from this bug, and forward walking 0.6-1.4% high; spintrack was within about 1% on all three components.

FicTrac's tracking window also stays where the config put it. When the ball moves in its holder, FicTrac reads the movement as rotation; spintrack's window follows the ball, and `tracks.parquet` records where the ball was and how far the window was from it on every frame.

**Re-track rather than pool.** FicTrac's error depends on the rig's off-axis angle and on how correlated sideslip and turning are in each recording, so no constant corrects it. Re-track old FicTrac recordings with spintrack, using their configs translated as above, instead of combining FicTrac and spintrack results in one analysis.

### Other differences

- The lab frame is x forward, y left, z up rather than FicTrac's y right, z down, so in `tracks.parquet` the lab rotations' y and z, and the path's y, heading, direction and sideways motion, have the opposite sign. The streams keep FicTrac's.
- Column 5 is a photometric residual, not FicTrac's matching error; columns 12-14 start at 0 rather than at `c2a_r`; column 22 holds the video position on the first row too. See [output.md](output.md#differences-from-fictracs-output).
- The surface map is an equi-angular cubemap rather than FicTrac's equal-area grid. FicTrac sphere-map PNGs load and are converted.
- Each run writes one folder: the records as Parquet with named columns, a summary of the run's quality, the run's log and the config as run. It refuses to overwrite an earlier run's files without `--overwrite`.
