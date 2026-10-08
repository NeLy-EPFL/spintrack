# User guide

From a recorded clip to a tracked file, and how to tell whether to trust it. Coming from FicTrac, [fictrac.md](fictrac.md) translates a `config.txt`; the sections on the camera position and on masking still apply.

## Set up a rig

### The config

A config is a TOML file that describes a rig and, through `video`, a recording of it. `spintrack calibrate` writes the measured parts; the example's is:

```toml
video = "sample.mp4"

[camera]
vfov_deg = 2.3893
position_deg = [0, 180, 0]  # elevation, azimuth, twist

[ball]
rim = [[656.5, 411.0], [142.5, 411.5], [145.0, 321.0], ...]

[tracking]
window_px = 120
global_search = true
max_bad_frames = 100
```

Every key, with its default where it has one:

| key | default | meaning |
|---|---|---|
| `video` | | the video, relative to the config, or a camera index (a number) |
| `camera.vfov_deg` | | the vertical field of view in degrees; `calibrate --auto` fits it when missing |
| `camera.fisheye` | `false` | an equidistant (f-theta) lens rather than a pinhole |
| `camera.fps` | | the frame rate, for a source that does not report one |
| `camera.position_deg` | | where the camera sits around the animal: elevation, azimuth, twist ([below](#the-camera-position)); required to run |
| `camera.rotation` | | the camera-to-animal rotation vector instead, as a calibration square gives it |
| `ball.rim` | | points on the ball's rim, `[x, y]` in image pixels; detected when missing |
| `mask.ignore` | `[]` | polygons to ignore, each a list of `[x, y]` pixels ([below](#mask-the-animal)) |
| `tracking.window_px` | `60` | the side of the square tracking window |
| `tracking.global_search` | `false` | relocalize against the whole map when the local solve fails |
| `tracking.max_bad_frames` | | restart tracking after this many lost frames in a row; never when missing |
| `tracking.max_step_rad` | `0.35` | the largest rotation accepted in one frame |
| `tracking.norm_window` | `0.25` | the side of the local brightness normalization, as a fraction of the window |
| `tracking.forget_outside_view` | `false` | forget map cells the window no longer sees |
| `tracking.illumination` | `true` | separate the rig's static lighting from the ball's texture |
| `tracking.initial_map` | | start from a saved `map.npz`, or a FicTrac sphere-map PNG |
| `tracking.freeze_map` | `false` | never update `initial_map` |
| `tracking.initial_illumination` | | start the lighting correction from a `map.npz` saved on the same rig, which helps short recordings |
| `output.name` | the video's name | names the output folder, `NAME_spintrack` |
| `output.debug_video` | `false` | also write `debug.mp4`, as `--debug-video` does |
| `output.debug_codec` | `"h264"` | its codec: `h264`, `hevc` or `vp9`, or OpenCV's `mp4v`, `xvid`, `mjpg` |
| `stream.udp` | | stream the records to `HOST:PORT` over UDP |
| `stream.serial` | | stream the records to `PORT[:BAUD]` |

Paths are relative to the config. An unknown key is an error that names the key it most resembles, so a typo cannot pass for a default.

Calibration now limits accuracy more than the code does. A 1% error in the ball's radius changes the rotation reported about axes in the image plane (forward walking, for a camera behind the animal) by about 2%. Where sideslip and turning are correlated (0.83-0.97 on the trials measured), 1 deg of error in the camera position changes turning by about 2%.

### Find the ball

`calibrate --auto` creates the config if needed, detects the ball, fits `camera.vfov_deg` if it is missing, and with `--camera-position` writes the camera position (next section), all without a window:

```console
$ spintrack calibrate config.toml --src clip.mp4 --auto --camera-position 0 180 0
...
ball: detected at (390.4, 387.9) r 261.5 px, confidence 1.00
vfov: 2.454 deg (fitted; not identifiable, the cost is flat over 1-6.02 deg, and the rotation scale does not depend on it)
wrote config.toml
```

Detection uses the first 100 frames (`--frames`) and exits with status 2 rather than write a ball it is unsure of. The field of view is fitted only here (`run` refuses a config without one), so all runs of a rig share it. On a long lens the fit is flat, as above, and any value in the flat range gives the same rotations; if you know the field of view, write it. `run` detects the ball itself when the config has none, but writing it once keeps a rig's runs consistent.

Detection looks at the per-pixel 90th percentile of those frames: as the ball turns, its dark markings pass over every pixel of it, so the high percentile shows the ball's plain surface while the static background stays as it is. Two detectors work on that image:

- With the `sam` extra (`uv sync --extra sam` or `pip install ".[sam]"`), SAM 3 is asked for a "ball" and a "sphere". Its masks only say where the ball is: the rim is measured on the image in a narrow band around each mask, and a candidate is kept only if the rim confirms enough of the outline the mask shows, fits a circle tightly, stands out of the noise and agrees with the mask; among those, SAM's score decides. On 266 recordings from six rigs this found all 225 balls without a wrong one and refused all 41 recordings with no usable ball in view. It takes 0.3 s on a GPU or about 13 s on a CPU, once per recording. The checkpoint (3.4 GB, downloaded on first use) is gated by Meta: request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and log in once with `hf auth login`. Its [license](https://github.com/facebookresearch/sam3/blob/main/LICENSE) asks publications that use it to acknowledge SAM.
- Without the extra, or when the checkpoint cannot be loaded, a classical detector thresholds the image. It is as precise when it answers, but on the same recordings it found only 108 of the 225 balls, refusing the rest.

Both assume a ball lighter than its markings, as on every rig tested. A ball that moves in its holder during the first frames reads slightly large; the summary's radius line shows it.

Without `--auto`, a window opens: `c` marks rim points, `i` draws an ignore polygon, `s` marks a calibration square's corners (written as `camera.rotation`), `a` sets the camera position with sliders, `w` writes the config. `calibrate` rewrites the whole file: the comments at its top are kept, those further down are not.

### The camera position

`--camera-position ELEV AZIM TWIST` writes where the camera sits, in degrees, as `camera.position_deg`. The azimuth runs around the animal's vertical axis: 0 in front, 90 at its right, 180 behind, -90 at its left. The elevation is the angle above the horizontal, and the twist rolls the camera about its optical axis. The camera is assumed to look at the ball's center, with image-down as close to animal-down as the geometry allows.

| camera | `position_deg` | the same as `rotation` | image right is the animal's |
|---|---|---|---|
| behind the animal, level with the ball | `[0, 180, 0]` | `[1.2092, 1.2092, 1.2092]` | right |
| behind, 30 deg above | `[30, 180, 0]` | `[0.8155, 0.8155, 1.4125]` | right |
| in front, level | `[0, 0, 0]` | `[1.2092, -1.2092, -1.2092]` | left |
| at the animal's right, level | `[0, 90, 0]` | `[1.5708, 0, 0]` | front |
| at the animal's left, level | `[0, -90, 0]` | `[0, -2.2214, -2.2214]` | back |
| above, head toward the top of the image | `[90, 0, 180]` | `[0, 0, 1.5708]` | right |
| above, head toward the bottom of the image | `[90, 0, 0]` | `[0, 0, -1.5708]` | left |

A config gives one of `position_deg` and `rotation`. `rotation = [0, 0, 0]` makes the camera frame the animal frame on purpose, and FicTrac's `c2a_r` works unchanged as `rotation`. `run --camera-position` overrides the config's for one run.

### Mask the animal

The solver assumes that everything in the tracking window turns with the ball. The animal does not, so cover it with `mask.ignore` polygons in image pixels, `ignore = [[[611, 348], [912, 337], [880, 450]], ...]`, from the calibrator's `i` step or by hand. Cover the legs' whole reach, not just the body: the robust weights reject legs only at the cost of the texture behind them, and legs at the rim also pull the ball follower's silhouette look. The debug video outlines the ignored regions in red.

## Run

```bash
spintrack run config.toml                       # the video the config names
spintrack other.mp4 --config config.toml        # another recording of the same rig
spintrack run config.toml --out results/trial3  # into a folder of your choice
```

`spintrack VIDEO` is short for `spintrack run VIDEO`; a `run` argument ending in `.toml` is a config, anything else a video (or a camera index). Each run writes one folder, `NAME_spintrack` next to the video unless `--out` names another, where NAME is the config's `output.name` or the video's name. It holds `tracks.parquet`, `summary.json`, `log.txt` and `config.toml` ([output.md](output.md)). That `config.toml` is the config as run, with the command line's changes (`--camera-position`, `--load-map`, `--debug-video`) and the ball if the run detected it, so `spintrack run NAME_spintrack/config.toml --out OTHER`, with the run's `--two-pass` and `--max-frames` if it had them, tracks the same video the same way. A run will not replace an earlier run's files without `--overwrite`. `--max-frames N` stops early. `--save-map` adds `map.npz`, which `--load-map` starts a later run from (as it does a FicTrac sphere-map PNG) and `spintrack map` renders as a picture.

### Two passes

`--two-pass` maps the ball in a throwaway first pass, then tracks again from that map. The second pass starts with the whole ball mapped and the lighting field converged, and if the ball moved in its holder, its window follows a trajectory planned from the whole first pass, without the online follower's delay. It doubles the run time and changes the per-frame error on synthetic truth by under 1%, so use it when the ball moves in its holder or the first seconds matter.

### The debug video

`--debug-video` writes `debug.mp4` (`--debug-axes` adds the ball's axes). On the left is the frame with the ball's outline in green, the `mask.ignore` regions in red and the animal's trail over the ball; on the right, the tracking window, the fictive path (scale bar in ball radii), the map unfolded as a dice around the face the camera sees, the lighting gain (how much texture contrast each part of the window gets; dark is shadow) and the frame's numbers. Check that:

- the outline stays on the ball's rim, including when the ball moves in its holder;
- the trail rides with the texture rather than sliding across it;
- the map's faces show sharp texture, not smears;
- the lighting gain shows the holder's shadow and smooth shading, never texture;
- the path moves only when the animal walks.

### Live cameras and streaming

`spintrack run 0 --config config.toml` opens camera 0 through OpenCV; outputs go to `camera_spintrack` in the current directory, or to `NAME_spintrack` for an `output.name`. The config must already describe the ball and the field of view, since automatic geometry needs a recording, and `--two-pass` is refused. For cameras OpenCV cannot open (Basler, FLIR, ...), feed the vendor SDK's frames to the `Tracker` loop in the README.

Records stream in FicTrac's line format (`FT, ` and the 25 fields) with `--udp HOST:PORT`, `--tcp HOST:PORT`, `--serial PORT[:BAUD]` (install the `serial` extra) and `--print`. The config's `stream.udp` and `stream.serial` do what the flags do.

## Read the summary

Every run ends with a short block; `summary.json` holds the same and more (see [output.md](output.md#the-summary)). On the example:

```
run quality: 1050 frames, 1050 tracked, 0 dropped
ball moved: no
radius: silhouette 258.3 px, config 259.8 px (-0.6%)
hard tracking: none
```

- `run quality`: frames read, tracked and dropped. A dropped frame has no row in `tracks.parquet`.
- `ball moved`: `no`, or the frames over which the ball moved in its holder and how far, in pixels. The window followed; check those frames in the debug video.
- `radius`: the silhouette's radius over the first frames against the config's, with a warning beyond 3%.
- `hard tracking`: stretches where both the cost and the solver's iterations rose above the run's baseline. A few short ones in fast turns are normal; a long one means the model stopped fitting the ball.
- A `ball` line comes first when the run detected the ball itself.

## Troubleshooting

**"the ball is only N px in radius".** Below 15 px of radius in the image, the reported rotation shrinks without any other sign, starting with the rotation about the optical axis (sideslip, for a camera behind the animal). Put more pixels on the ball.

**The ball moved.** The window follows a ball that sinks or jerks in its holder, so the movement is not read as rotation. If the follower stopped, the window would have left the ball, usually because the look climbed onto the animal: mask the legs, check the debug video, and consider `--two-pass`.

**The radius warning.** The config's ball is more than 3% off the silhouette. The follower uses the measured radius, but the rotation scale comes from the config's: re-run `calibrate --auto` or correct `ball.rim`. `too little rim in view to compare` means the silhouette is too hidden to measure.

**Dropped frames.** No solve passed the gates: the rotation exceeded `tracking.max_step_rad`, or too little of the window matched the map. Look at those frames in the debug video for an unmasked animal, blur or a ball that left the window. `global_search = true` recovers by relocalizing, and `max_bad_frames` restarts tracking after that many losses in a row. The row after a gap spans the gap, as its `delta_ts` says.

**`run` refuses the config.** "no camera position": write one with `calibrate CONFIG --camera-position`, or pass `--camera-position` to the run. "no field of view": run `calibrate --auto` once. "invalid config": the message lists each wrong key and what it expected.
