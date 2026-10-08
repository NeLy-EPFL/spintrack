# Coming from FicTrac

spintrack reads FicTrac configs and writes FicTrac's `.dat`, so a rig moves over by changing the command. This page lists what maps to what and what differs. [output.md](output.md) has the columns in full.

## Commands

| FicTrac | spintrack |
|---|---|
| `fictrac config.txt` | `spintrack run config.txt` |
| `configGui config.txt` | `spintrack calibrate config.txt` (a window), or `--auto` and `--c2a-angles` (no window) |
| `save_debug : y` | the same, or `--debug-video` |
| `sphere_map_fn : map.png` | the same, or `--load-map map.png` |
| `sock_port`, `com_port` | the same, or `--udp`, `--tcp`, `--serial` |
| | `--two-pass`, `--save-map`, `--out`, `--overwrite` |

`spintrack run --help` lists every option.

## Config keys

Honored as in FicTrac: `src_fn`, `vfov`, `fisheye`, `q_factor`, `src_fps`, `max_bad_frames`, `opt_do_global`, `roi_circ`, `roi_c`, `roi_r`, `roi_ignr`, `c2a_r`, `c2a_src` with `c2a_cnrs_xy`/`_yz`/`_xz`, `sphere_map_fn`, `sock_host`, `sock_port`, `com_port`, `com_baud`, `save_debug`, `vid_codec`, `output_fn`, and the lab fork's `accumulate_map`. A `q_factor` of 0 or less falls back to 6, as in FicTrac.

Honored with a meaning that fits the new solver: `opt_bound` is the largest rotation accepted in one frame (rad), and `thr_win_pc` sets the size of the local brightness normalization instead of the thresholding window.

Accepted and ignored: `do_display`, `save_raw`, `thr_ratio`, `thr_rgb_tfrm`, `opt_max_evals`, `opt_tol`, `c2a_t`, `enh_cfg_disp`, `reconfig`, and `opt_max_err`, which warns because spintrack does not drop frames on the matching error. Unknown keys warn once. Inline `#` comments are stripped.

spintrack's own keys: `map_frozen : y` never updates a map loaded through `sphere_map_fn`; `illumination : n` turns off the static lighting correction; `illumination_fn` starts that correction from a map `.npz` saved on the same rig, which helps short recordings.

Three things are handled differently:

- A camera-to-animal transform (`c2a_r`, or the square's corners FicTrac's calibration writes) is required, because without one the lab columns would be camera columns under another name. `spintrack calibrate CONFIG --c2a-angles ELEV AZIM TWIST` writes it ([guide](guide.md#the-camera-position)); `c2a_r : { 0, 0, 0 }` asks for the identity explicitly.
- `vfov` must be a number when tracking. `vfov : auto` is fitted once by `spintrack calibrate CONFIG --auto`.
- A config without a ball (`roi_circ` or `roi_c`/`roi_r`) is not an error: the ball is detected in the recording.

## What differs and why

### The frame bug

FicTrac estimates the rotation in its tracking-window frame, whose z axis points at the ball's center, and writes it as the camera frame: upstream `src/Trackball.cpp:889` sets `dr_cam = dr_roi`, with the window-to-camera transform commented out. It then applies `c2a_r` to these window-frame vectors, so its lab columns, heading and path are rotated by the ball's off-axis angle (0.76 deg on the rig measured). spintrack applies `c2a_r` to true camera-frame vectors.

The rotation leaks sideslip into turning wherever the two are correlated (0.83-0.97 on the trials measured). On six 60 s trials of one rig, scored against an independent referee, FicTrac read turning about 2% high, mostly from this bug, and forward walking 0.6-1.4% high; spintrack was within about 1% on all three components.

FicTrac's tracking window also stays where the config put it. When the ball moves in its holder, FicTrac reads the movement as rotation; spintrack's window follows the ball.

**Re-track rather than pool.** FicTrac's error depends on the rig's off-axis angle and on how correlated sideslip and turning are in each recording, so no constant corrects it. Re-track old FicTrac recordings with spintrack, using the same configs, instead of combining FicTrac and spintrack results in one analysis.

### Other differences

- Column 5 is a photometric residual, not FicTrac's matching error; columns 12-14 start at 0 rather than at `c2a_r`; column 22 holds the video position on the first row too. See [output.md](output.md#differences-from-fictracs-output).
- The surface map is an equi-angular cubemap rather than FicTrac's equal-area grid. FicTrac sphere-map PNGs load and are converted.
- Each run also writes a Parquet copy of the records and a summary of the run's quality, and refuses to overwrite an earlier run's files without `--overwrite`.
