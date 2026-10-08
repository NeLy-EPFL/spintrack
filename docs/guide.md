# User guide

From a recorded clip to a tracked file, and how to tell whether to trust it. A FicTrac config works as it is ([fictrac.md](fictrac.md)), but the sections on the camera position and on masking still apply.

## Set up a rig

### The config

spintrack reads FicTrac's `key : value` format, or YAML or TOML with the same keys. The keys that matter:

| key | meaning |
|---|---|
| `src_fn` | the video, relative to the config, or a camera index |
| `vfov` | the camera's vertical field of view in degrees; `auto` asks `calibrate --auto` to fit it |
| `roi_circ` | points on the ball's rim in the image; `roi_c` and `roi_r` (the ball's direction and angular radius) are the alternative |
| `c2a_r` | the camera-to-animal rotation, as a rotation vector; required |
| `roi_ignr` | polygons in the image to ignore: the animal, the tether, the holder's edge |
| `q_factor` | the tracking window is `10 * q_factor` pixels square (default 6) |
| `opt_do_global` | `y` relocalizes against the whole map after the local solve fails |
| `output_fn` | the base name of the outputs; default the video's name |

Calibration now limits accuracy more than the code does. A 1% error in the ball's radius changes the rotation reported about axes in the image plane (forward walking, for a camera behind the animal) by about 2%. Where sideslip and turning are correlated (0.83-0.97 on the trials measured), 1 deg of error in `c2a_r` changes turning by about 2%.

### Find the ball

`calibrate --auto` creates the config if needed, detects the ball, fits `vfov` if it is missing or `auto`, and with `--c2a-angles` writes `c2a_r` (next section), all without a window:

```console
$ spintrack calibrate config.txt --src clip.mp4 --auto --c2a-angles 0 180 0
...
ball: detected at (390.4, 387.9) r 261.5 px, confidence 1.00
vfov: 2.454 deg (fitted; not identifiable, the cost is flat over 1-6.02 deg, and the rotation scale does not depend on it)
wrote config.txt
```

Detection uses the first 100 frames (`--frames`) and exits with status 2 rather than write a ball it is unsure of. `vfov` is fitted only here (`run` refuses `vfov : auto`), so all runs of a rig share it. On a long lens the fit is flat, as above, and any value in the flat range gives the same rotations; if you know the field of view, write it. `run` detects the ball itself when the config has none, but writing it once keeps a rig's runs consistent.

Without `--auto`, a window opens: `c` marks rim points, `i` draws an ignore polygon, `s` marks a calibration square's corners, `a` sets the camera position with sliders, `w` writes the config.

### The camera position

`--c2a-angles ELEV AZIM TWIST` writes `c2a_r` from where the camera sits, in degrees. The azimuth runs around the animal's vertical axis: 0 in front, 90 at its right, 180 behind, -90 at its left. The elevation is the angle above the horizontal, and the twist rolls the camera about its optical axis. The camera is assumed to look at the ball's center, with image-down as close to animal-down as the geometry allows.

| camera | `--c2a-angles` | `c2a_r` | image right is the animal's |
|---|---|---|---|
| behind the animal, level with the ball | `0 180 0` | `{ 1.2092, 1.2092, 1.2092 }` | right |
| behind, 30 deg above | `30 180 0` | `{ 0.8155, 0.8155, 1.4125 }` | right |
| in front, level | `0 0 0` | `{ 1.2092, -1.2092, -1.2092 }` | left |
| at the animal's right, level | `0 90 0` | `{ 1.5708, 0, 0 }` | front |
| at the animal's left, level | `0 -90 0` | `{ 0, -2.2214, -2.2214 }` | back |
| above, head toward the top of the image | `90 0 180` | `{ 0, 0, 1.5708 }` | right |
| above, head toward the bottom of the image | `90 0 0` | `{ 0, 0, -1.5708 }` | left |

`c2a_r : { 0, 0, 0 }` makes the camera frame the animal frame on purpose. A `c2a_r` from FicTrac's calibration square works unchanged.

### Mask the animal

The solver assumes that everything in the tracking window turns with the ball. The animal does not, so cover it with `roi_ignr` polygons in image pixels, `{ { x1, y1, x2, y2, ... }, { ... } }`, from the calibrator's `i` step or by hand. Cover the legs' whole reach, not just the body: the robust weights reject legs only at the cost of the texture behind them, and legs at the rim also pull the ball follower's silhouette look. The debug video outlines the ignored regions in red.

## Run

```bash
spintrack run config.txt                    # NAME.dat, NAME.parquet, NAME-summary.json
spintrack run config.txt --src other.mp4    # another recording of the same rig
spintrack run config.txt --out results/     # write into a directory
spintrack run config.txt --out trial3.dat   # or name every output after a path
```

NAME is the config's `output_fn` or the video's name. A run will not replace an earlier run's files without `--overwrite`. `--max-frames N` stops early. `--save-map` adds `NAME-map.npz`, which `--load-map` starts a later run from (as it does a FicTrac sphere-map PNG) and `spintrack map` renders as a picture.

### Two passes

`--two-pass` maps the ball in a throwaway first pass, then tracks again from that map. The second pass starts with the whole ball mapped and the lighting field converged, and if the ball moved in its holder, its window follows a trajectory planned from the whole first pass, without the online follower's delay. It doubles the run time and changes the per-frame error on synthetic truth by under 1%, so use it when the ball moves in its holder or the first seconds matter.

### The debug video

`--debug-video` writes `NAME-debug.mp4` (`--debug-axes` adds the ball's axes). On the left is the frame with the ball's outline in green, the `roi_ignr` regions in red and the animal's trail over the ball; on the right, the tracking window, the fictive path (scale bar in ball radii), the map unfolded as a dice around the face the camera sees, the lighting gain (how much texture contrast each part of the window gets; dark is shadow) and the frame's numbers. Check that:

- the outline stays on the ball's rim, including when the ball moves in its holder;
- the trail rides with the texture rather than sliding across it;
- the map's faces show sharp texture, not smears;
- the lighting gain shows the holder's shadow and smooth shading, never texture;
- the path moves only when the animal walks.

### Live cameras and streaming

`--src 0` opens camera 0 through OpenCV; outputs are named `camera.*` unless `output_fn` is set. The config must already describe the ball and `vfov`, since automatic geometry needs a recording, and `--two-pass` is refused. For cameras OpenCV cannot open (Basler, FLIR, ...), feed the vendor SDK's frames to the `Tracker` loop in the README.

Records stream in FicTrac's line format (`FT, ` and the 25 fields) with `--udp HOST:PORT`, `--tcp HOST:PORT`, `--serial PORT[:BAUD]` (install the `serial` extra) and `--print`. The config's `sock_host`/`sock_port` and `com_port`/`com_baud` work as in FicTrac.

## Read the summary

Every run ends with a short block; `NAME-summary.json` holds the same and more (see [output.md](output.md#the-summary)). On the example:

```
run quality: 1050 frames, 1050 tracked, 0 dropped
ball moved: no
radius: silhouette 258.4 px, config 259.8 px (-0.6%)
hard tracking: none
```

- `run quality`: frames read, tracked and dropped. A dropped frame has no row in the `.dat`.
- `ball moved`: `no`, or the frames over which the ball moved in its holder and how far, in pixels. The window followed; check those frames in the debug video.
- `radius`: the silhouette's radius over the first frames against the config's, with a warning beyond 3%.
- `hard tracking`: stretches where both the cost and the solver's iterations rose above the run's baseline. A few short ones in fast turns are normal; a long one means the model stopped fitting the ball.
- A `ball` line comes first when the run detected the ball itself.

## Troubleshooting

**"the ball is only N px in radius".** Below 15 px of radius in the image, the reported rotation shrinks without any other sign, starting with the rotation about the optical axis (sideslip, for a camera behind the animal). Put more pixels on the ball.

**The ball moved.** The window follows a ball that sinks or jerks in its holder, so the movement is not read as rotation. If the follower stopped, the window would have left the ball, usually because the look climbed onto the animal: mask the legs, check the debug video, and consider `--two-pass`.

**The radius warning.** The config's ball is more than 3% off the silhouette. The follower uses the measured radius, but the rotation scale comes from the config's: re-run `calibrate --auto` or correct `roi_circ`. `too little rim in view to compare` means the silhouette is too hidden to measure.

**Dropped frames.** No solve passed the gates: the rotation exceeded `opt_bound` (rad per frame), or too little of the window matched the map. Look at those frames in the debug video for an unmasked animal, blur or a ball that left the window. `opt_do_global : y` recovers by relocalizing, and `max_bad_frames` restarts tracking after that many losses in a row. The row after a gap spans the gap, as its `delta_ts` says.

**`run` refuses the config.** "no camera-to-animal transform": write `c2a_r`, for instance with `--c2a-angles`. "vfov is auto": run `calibrate --auto` once.
