# User guide

From a recording to a tracked file, and how to tell whether to trust it. `spintrack run VIDEO` finds what it needs in the recording; when it gets something wrong, `spintrack gui VIDEO` fixes it while you watch, and saves a config that later runs start from. Coming from FicTrac, [fictrac.md](fictrac.md) translates a `config.txt`; the sections on the camera position and on masking still apply.

## Run

```bash
spintrack run trial3.mp4 camera.azimuth_deg=180            # a camera behind the fly
spintrack run session/*.mp4 -c rig.toml                     # a rig's config, many videos
spintrack run trial3.mp4 -c rig.toml tracking.window_px=80  # one key changed for this run
spintrack run trial3_spintrack/config.toml --out redo       # repeat a run
```

`spintrack VIDEO` is short for `spintrack run VIDEO`. Where the config (`-c`) leaves them out, a run finds in the recording:

- **the ball**, with SAM 3 ([below](#how-the-ball-is-found));
- **the field of view and the camera position** from a deeperfly calibration, when the video belongs to a deeperfly project ([below](#a-deeperfly-calibration));
- otherwise **the field of view**, by tracking 1000 frames under a range of candidates and keeping the one that fits best, which takes 10-20 s;
- and **the camera's elevation and twist**, from where the animal stands on the ball ([below](#the-camera-position)).

The one thing a run cannot find is which side the camera films the animal from, its front or its back, so without a deeperfly project it asks for the azimuth: `camera.azimuth_deg=180` for a camera behind the animal, `0` in front, `90` at its right, `-90` at its left. FicTrac requires the same, as `c2a_r`. Put it in the rig's config to run many videos.

The summary at the end says what was found and how, and the run's `config.toml` records it.

Any video FFmpeg reads will do: color or gray, 8 or more bits, any container. Frames are tracked as luma (gray). An AVI is timed and sought by frame count, as it stores no presentation times: FFmpeg's guesses are out of order under B-frames, and its seeks by them can land hundreds of frames off.

Each video gets one folder, `NAME_spintrack` next to it unless `--out` names another (for one video), where NAME is the config's `output.name` or the video's name. It holds `tracks.parquet`, `summary.json`, `log.txt` and `config.toml` ([output.md](output.md)). That `config.toml` is the config as run, with the command line's changes and what the run found, so `spintrack run NAME_spintrack/config.toml --out OTHER`, with the run's `--two-pass` and `--max-frames` if it had them, tracks the same video the same way. A run will not replace an earlier run's files without `--overwrite`. When one of several videos fails, the others still run and the exit status is 2. `--max-frames N` stops early. `--save-map` adds `map.npz`, which `tracking.initial_map=PATH` starts a later run from (as it does a FicTrac sphere-map PNG).

### The preview

A run prints a link, `preview: http://127.0.0.1:8300/?token=...`. Open it to watch the run: the frame with the ball's outline, the animal's trail over the ball and the ignored regions (and the axes, on request), the tracking window, the fictive path, the surface map, the last 10 s of forward, side and turning speed and of the solver's cost, and the summary at the end. Its Stop ends the run as Ctrl-C does, with the records so far written. The page costs the run nothing until it is opened, and little after: it captures at most 12 frames a second, and only while a page asks. On a remote machine, forward the port first, as the line says (`ssh -L 8300:localhost:8300 HOST`). `--no-preview` serves nothing; `--port` picks the port.

What to check, on the page or in the debug video:

- the outline stays on the ball's rim, including when the ball moves in its holder;
- the trail starts under the animal and rides with the texture rather than sliding across it;
- the map's faces show sharp texture, not smears;
- the lighting gain (the map's corner tile) shows the holder's shadow and smooth shading, never texture;
- the path moves only when the animal walks.

### Change the config on the command line

`KEY=VALUE` arguments set config keys for one run, over the config's: `tracking.window_px=80`, `camera.position_deg=0,180,0`, `output.debug_video=true`, `tracking.initial_map=map.npz`. A value is read as TOML, a list may drop its brackets, and `none` restores the default (`ball.rim=none` finds the ball again even when the config has one). Setting `camera.position_deg` replaces a config's `camera.rotation`, and the other way around. A misspelled key is an error that names the key it most resembles. In zsh, quote a value with brackets: `'ball.rim=[[1, 2], [3, 4], [5, 6]]'`.

### Two passes

`--two-pass` maps the ball in a throwaway first pass, then tracks again from that map. The second pass starts with the whole ball mapped and the lighting field converged, and if the ball moved in its holder, its window follows a trajectory planned from the whole first pass, without the online follower's delay. It doubles the run time and changes the per-frame error on synthetic truth by under 1%, so use it when the ball moves in its holder or the first seconds matter.

### The debug video

`--debug-video` (or `output.debug_video=true`) writes `debug.mp4`: on the left the frame as the preview shows it, on the right the tracking window, the fictive path (scale bar in ball radii), the map unfolded as a dice around the face the camera sees, the lighting gain and the frame's numbers. Arrows show the ball's axes from its center, turning with it, and the animal's from where it stands on the ball: x forward, y left, z the ball's normal there. An axis along the line of sight is a ring, dotted when it points toward the camera and crossed when away. `output.debug_axes=false` leaves them out. When SAM 3 found the ball or placed the camera, an empty tile of the map's net ("segmentation") shows its masks over the first frame, with their scores: the ball's in yellow, as its outline, the animal's in magenta. They are found once, on the opening frames, so the tile does not change.

### Live cameras and streaming

`spintrack run 0 -c config.toml` opens camera 0 through OpenCV; outputs go to `camera_spintrack` in the current directory, or to `NAME_spintrack` for an `output.name`. A live camera cannot be looked at before tracking starts, so the config must describe the ball, the field of view and the camera position: run or open a short recording of the rig first and pass the config it writes. `--two-pass` is refused. For cameras OpenCV cannot open (Basler, FLIR, ...), feed the vendor SDK's frames to the `Tracker` loop in the README.

Records stream in FicTrac's line format (`FT, ` and the 25 fields) with `--udp HOST:PORT`, `--tcp HOST:PORT`, `--serial PORT[:BAUD]` (install the `serial` extra) and `--print`. The config's `stream.udp` and `stream.serial` do what the flags do.

## Fix it in the gui

```bash
spintrack gui trial3.mp4                # saves spintrack.toml next to the video
spintrack gui trial3.mp4 -c rig.toml    # starts from rig.toml and saves to it
```

The gui opens the preview's page in a browser (with `--no-browser`, or without a display, it prints the link), finds what the config leaves out as a run does (without an azimuth, it asks which side the camera films from, and tracks as if from behind until told), and then tracks a clip of the video, 10 s from its start, over and over at its own speed. Every change restarts the clip, so its effect shows within seconds. The panel on the left changes:

- **Ball**: drag the circle's center or edge, click points on the ball's edge, or have SAM 3 find it again.
- **Camera position**: presets (behind, in front, at either side, above), sliders for the elevation, azimuth and twist, or the four corners of a calibration square, clicked in FicTrac's order, which names them in the animal's terms and so holds from any side (for the xy plane: front-left, front-right, back-right, back-left; the gui gives each plane's). Tick *axes* under the frame to check: from where the animal stands, which is where the trail starts, x (red) points forward, y (green) to the animal's left and z (blue) up.
- **Field of view**: type it, or fit it again.
- **Ignored regions**: click a polygon's corners over what moves but is not the ball ([below](#mask-the-animal)).
- **Tracking**: the parameters of the config's `[tracking]`, each with its default a click away.

Under the frame, pick the clip's start and length and the playback speed; Space pauses and `.` steps a paused clip by a frame. **Save** writes the config, to `-c`'s file or to `spintrack.toml` next to the video, keeping the comments at the file's top, and shows the command that uses it: `spintrack run VIDEO... -c spintrack.toml`. **Track the whole video** saves, then runs as `spintrack run` would, on the same page.

## The config

A config is a TOML file that describes a rig and, through `video`, a recording of it. Every key is optional. The example's is:

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
| `camera.vfov_deg` | fitted | the vertical field of view in degrees |
| `camera.fisheye` | `false` | an equidistant (f-theta) lens rather than a pinhole |
| `camera.fps` | | the frame rate, for a source that does not report one |
| `camera.position_deg` | found | where the camera sits around the animal: elevation, azimuth, twist ([below](#the-camera-position)) |
| `camera.rotation` | | the camera-to-animal rotation vector instead, as a calibration square gives it |
| `camera.calibration` | found next to the video | a deeperfly calibration, manifest or project folder, for the field of view and the camera position ([below](#a-deeperfly-calibration)) |
| `camera.view` | from the manifest | the calibration's view that filmed the video |
| `camera.azimuth_deg` | | which side the camera films from, when nothing above says: 180 behind the animal, 0 in front, 90 at its right, -90 at its left; required then |
| `ball.rim` | detected | points on the ball's rim, `[x, y]` in image pixels |
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
| `output.debug_axes` | `true` | draw the ball's and the animal's axes in `debug.mp4` |
| `output.debug_codec` | `"h264"` | its codec: `h264`, `hevc` or `vp9`, or OpenCV's `mp4v`, `xvid`, `mjpg` |
| `stream.udp` | | stream the records to `HOST:PORT` over UDP |
| `stream.serial` | | stream the records to `PORT[:BAUD]` |

Paths are relative to the config. An unknown key is an error that names the key it most resembles, so a typo cannot pass for a default. A config that describes the ball fixes it for every video it is used with; leave `ball.rim` out of a rig's config to find the ball in each video instead.

Calibration now limits accuracy more than the code does. A 1% error in the ball's radius changes the rotation reported about axes in the image plane (forward walking, for a camera behind the animal) by about 2%. Where sideslip and turning are correlated (0.83-0.97 on the trials measured), 1 deg of error in the camera position changes turning by about 2%.

### How the ball is found

Detection looks at the per-pixel 90th percentile of the first 100 frames: as the ball turns, its dark markings pass over every pixel of it, so the high percentile shows the ball's plain surface while the static background stays as it is. Two detectors work on that image:

- By default, SAM 3 is asked for a "ball" and a "sphere". Its masks only say where the ball is: the rim is measured on the image in a narrow band around each mask, and a candidate is kept only if the rim confirms enough of the outline the mask shows, fits a circle tightly, stands out of the noise and agrees with the mask; among those, SAM's score decides. On 266 recordings from six rigs this found all 225 balls without a wrong one and refused all 41 recordings with no usable ball in view. It takes 0.3 s on a GPU or about 13 s on a CPU, once per recording. The checkpoint (3.4 GB) downloads on first use, without a Hugging Face login, from [an ungated copy](https://huggingface.co/tkclam/sam3) of Meta's [gated original](https://huggingface.co/facebook/sam3); its files are byte-identical to Meta's. Its [license](https://github.com/facebookresearch/sam3/blob/main/LICENSE) asks publications that use it to acknowledge SAM.
- When the checkpoint cannot be loaded (offline before its first download), a classical detector thresholds the image instead. It is as precise when it answers, but on the same recordings it found only 108 of the 225 balls, refusing the rest.

Both assume a ball lighter than its markings, as on every rig tested, and a run refuses a recording where neither is sure rather than guess; place the ball in the gui then. A ball that moves in its holder during the first frames reads slightly large; the summary's radius line shows it. The field of view is fitted with the ball's outline held fixed. On a long lens the fit is flat, and the summary says how much the rotation scale changes over the flat range (a few percent at most on the lab's rigs). A wider lens shows a clear minimum once the ball has turned a few hundred degrees. A lean toward a wide lens without one is not taken for a wide lens: lens distortion causes it too (a lab octacam with a 2 deg lens leaned to 26 deg and would have read rotations 18% low). A narrow lens is assumed then, and the summary says how much smaller the rotations would read at the vfov the cost leaned to. When the narrow end clearly loses, or the ball turned too little to tell, a run refuses. Rotations read up to tens of percent smaller through a wide lens than through a narrow one at the same image motion, so if you know the field of view, write it, above all for webcams and phones.

### The camera position

`camera.position_deg` is where the camera sits, in degrees: elevation, azimuth, twist. The azimuth runs around the animal's vertical axis: 0 in front, 90 at its right, 180 behind, -90 at its left. The elevation is the angle above the horizontal, and the twist rolls the camera about its optical axis. The camera is assumed to look at the ball's center, with image-down as close to animal-down as the geometry allows.

| camera | `position_deg` | the same as `rotation` | image right is the animal's |
|---|---|---|---|
| behind the animal, level with the ball | `[0, 180, 0]` | `[-1.2092, 1.2092, -1.2092]` | right |
| behind, 30 deg above | `[30, 180, 0]` | `[-1.5835, 1.5835, -0.9142]` | right |
| in front, level | `[0, 0, 0]` | `[-1.2092, -1.2092, 1.2092]` | left |
| at the animal's right, level | `[0, 90, 0]` | `[-1.5708, 0, 0]` | front |
| at the animal's left, level | `[0, -90, 0]` | `[0, -2.2214, 2.2214]` | back |
| above, head toward the top of the image | `[90, 0, 180]` | `[2.2214, -2.2214, 0]` | right |
| above, head toward the bottom of the image | `[90, 0, 0]` | `[-2.2214, -2.2214, 0]` | left |

A config gives one of `position_deg` and `rotation`, the camera-to-animal rotation vector in the animal frame (x forward, y left, z up). `rotation = [0, 0, 0]` makes the camera frame the animal frame on purpose. FicTrac's `c2a_r` is in its own frame (y right, z down); `spintrack.calibrate.sliders.rotation_from_fictrac` converts it.

Without either, a deeperfly project the video belongs to places the camera ([below](#a-deeperfly-calibration)). Without one, `camera.azimuth_deg` says which side the camera films from, and SAM 3 is asked for the "insect" in three frames, whose silhouette gives the rest. A tethered animal stays put while the ball turns, so the animal is the mask found at the same place in two of the three (a spot of the ball's pattern can be the best "insect" in one frame), and a mask over most of the ball is the ball (a cow-patterned one is an "insect" too). "insect" finds crabs as well. The animal frame's up is the ball's normal where the animal stands, so the silhouette's direction from the ball's center gives the twist, and its distance the elevation: inside the outline, at the elevation's cosine, for a camera above. On the outline the elevation does not show, because the animal's height above the ball outweighs the elevation's cosine there, and level is assumed. On the lab's octacam, where the fly's leg tips put the hind camera 10.5-12 deg above the fly's horizon ([below](#a-deeperfly-calibration)), the silhouette looks as it would from a level camera, while the twist it gives agrees with the leg tips' to 0.5 deg. An elevation error mixes turning and sideways rotation by about its sine (17% at 10 deg), so on such a rig give the elevation (`camera.position_deg`) or use a deeperfly project. From nearly straight above (the animal within a quarter radius of the ball's center), the elevation still shows but the twist does not: none is assumed, and the azimuth says which way the animal faces in the image (180 up, 90 right, 0 down, -90 left). No animal found leaves the camera level, and the summary says so. Without any of these, a run refuses, before any slow work.

Every run then checks the azimuth against the animal's net walking, since animals walk mostly forward: the summary's `walking` line says where the net walking points, and when it points more than 45 deg from forward over at least 10 ball radii, it says which azimuth would make it forward. It is a check for gross errors, not a measurement: on 48 octacam trials with the azimuth measured against the fly's body axis, most flies walked under half a ball radius net in 20 s, and one that walked 5 radii did so 106 deg from its body axis. An experiment that makes the animal walk backward reads as 180 deg off; read the line knowing the experiment.

Neither the silhouette nor the walking can tell the azimuth: on octacam trials, SAM 3's "eye", "wing" and "abdomen" fired on flies seen from behind as often as on flies seen from the front. A nominal azimuth is good to a few degrees, though: on 48 octacam trials, the fly's body axis was 2.1 deg (median) and at most 7.6 deg from straight behind the camera that filmed it from behind. For the precise value, use a deeperfly project, or measure it in the gui: the sliders with the axes drawn on the ball, or the corners of a calibration square aligned with the animal's axes, as FicTrac's configGui does.

### A deeperfly calibration

A rig calibrated by [deeperfly](https://github.com/NeLy-EPFL/deeperfly) gives the field of view and the camera position, without a fit. When the video is listed in a deeperfly project next to it (`deeperfly/deeperfly.toml` in the video's folder, as octacam recordings are), a run uses the project's active calibration for the view that filmed the video, and says so. Elsewhere, name it: `camera.calibration=PATH`, a calibration file (`calibrations/NNNN-label.toml`), a manifest or a project folder, and `camera.view=NAME` when the manifest does not list the video. deeperfly's world (x forward, y left, z up) is the animal frame, so a solved view's rotation needs no conversion; a video downscaled from the calibrated size keeps its field of view, and one of another aspect ratio is refused. The config's own `vfov_deg`, `position_deg` or `rotation` win over the calibration's.

A calibration alone places the camera in the rig, not on the fly. Flies are never mounted exactly along the rig's axis (on 48 octacam trials, their bodies pointed 2.3 deg (median) and up to 11.7 deg off it), nor on the ball's top. So when the project also holds pose results (`deeperfly/results/*.h5`, the newest of one animal), the run places the camera relative to where the fly stands. It triangulates the fly's leg tips and thorax-coxa points in the opening frames, puts the ball's center on the ray through the center of the ball's image, at the distance that rests the most leg tips (the legs in stance) on its surface, and takes the ball's normal through the thorax-coxa points for up and the body's long axis, from the hind to the front coxae, laid on the ball there, for forward. The summary says where the fly stands and how many of its leg tips the ball holds. On four octacam trials the fly stood 9-13 deg from the ball's top, toward the hind camera, 73-84% of its leg tips rested on the fitted ball, and the camera behind it sat 10.5-12 deg above its horizon. Without leg tips in the results, or when the fit is not believable, the rig's up is up and the body gives only the heading.

### Mask the animal

The solver assumes that everything in the tracking window turns with the ball. The animal does not, so cover it with `mask.ignore` polygons in image pixels, `ignore = [[[611, 348], [912, 337], [880, 450]], ...]`, drawn in the gui or written by hand. Cover the legs' whole reach, not just the body: the robust weights reject legs only at the cost of the texture behind them, and legs at the rim also pull the ball follower's silhouette look. The page and the debug video outline the ignored regions in red.

## Read the summary

Every run ends with a short block; `summary.json` holds the same and more (see [output.md](output.md#the-summary)). On the example with no config:

```
run quality: 1050 frames, 1050 tracked, 0 dropped
ball: detected at (400.5, 371.5) r 260.8 px by SAM 3 (score 0.75), rim confirms 86% of its outline
vfov: 3.31 deg (fitted; not identifiable, the cost is flat over 1-11 deg, over which the rotation scale changes by 6%)
camera position: azimuth 180 from camera.azimuth_deg; elevation 0, twist -0.6 deg from where the animal stands (it stands on the ball's outline, where the elevation does not show and level is assumed; SAM 3 score 0.87)
walking: net 3.9 ball radii, 7 deg right of forward
ball moved: no
radius: silhouette 258.3 px, config 260.1 px (-0.7%)
hard tracking: none
```

- `run quality`: frames read, tracked and dropped. A dropped frame has no row in `tracks.parquet`.
- `ball`, `vfov`, `camera position`: what the run found in the recording, when the config did not say.
- `walking`: where the animal's net walking points ([above](#the-camera-position)).
- `ball moved`: `no`, or the frames over which the ball moved in its holder and how far, in pixels. The window followed; check those frames on the page or in the debug video.
- `radius`: the silhouette's radius over the first frames against the config's, with a warning beyond 3%.
- `hard tracking`: stretches where both the cost and the solver's iterations rose above the run's baseline. A few short ones in fast turns are normal; a long one means the model stopped fitting the ball.

## Troubleshooting

**"where the camera sits around the animal is unknown".** Add `camera.azimuth_deg=180` for a camera behind the animal (`0` in front, `90` at its right, `-90` at its left), or put it in the rig's config.

**"the ball is only N px in radius".** Below 15 px of radius in the image, the reported rotation shrinks without any other sign, starting with the rotation about the optical axis (sideslip, for a camera behind the animal). Put more pixels on the ball.

**No ball found.** SAM 3 and the rim check refused every candidate. Open the gui, drag the circle onto the ball and save, then run with that config.

**"its surface shows little texture to track".** The ball's surface varies by less than 4 gray levels at the tracking window's scale (patterned balls show 10 or more), as plain white polystyrene does: whatever rotation is reported comes from the animal, shadows or noise. Such balls need a pattern, or another sensor (optical mice).

**The field of view cannot be fitted.** The cost leaned to a wide lens without a clear minimum, or the ball turned too little. Write `camera.vfov_deg` from the lens (the sensor height and the focal length give it), or from the rig's calibration.

**The ball moved.** The window follows a ball that sinks or jerks in its holder, so the movement is not read as rotation. If the follower stopped, the window would have left the ball, usually because the look climbed onto the animal: mask the legs, check the page or the debug video, and consider `--two-pass`. `tracks.parquet` has the ball's position on every frame, and `window_offset` estimates the orientation error the window's lag caused on each one ([output.md](output.md#the-balls-position)).

**The radius warning.** The config's ball is more than 3% off the silhouette. The follower uses the measured radius, but the rotation scale comes from the config's: correct the ball in the gui, or run with `ball.rim=none` to detect it.

**The walking line says the camera position is wrong.** Unless the animal walked backward or sideways on purpose, set the azimuth the line suggests, `camera.position_deg=0,AZIMUTH,0`, or measure the rig in the gui.

**Dropped frames.** No solve passed the gates: the rotation exceeded `tracking.max_step_rad`, or too little of the window matched the map. Look at those frames for an unmasked animal, blur or a ball that left the window. `global_search = true` recovers by relocalizing, and `max_bad_frames` restarts tracking after that many losses in a row. The row after a gap spans the gap, as its `delta_ts` says.

**"invalid config".** The message lists each wrong key, from the file or from the command line, and what it expected.
