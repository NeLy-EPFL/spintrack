# User guide

From a recording to a tracked file, and how to tell whether to trust it. `spintrack run VIDEO` finds what it needs in the recording; when it gets something wrong, `spintrack gui VIDEO` fixes it while you watch, and saves a config that later runs start from. Coming from FicTrac, [fictrac.md](fictrac.md) translates a `config.txt`; the sections on the camera position and on masking still apply.

## Run

```bash
spintrack run trial3.mp4 --set camera.azimuth_deg=180             # a camera behind the fly
spintrack run session/*.mp4 -c rig.toml                           # a rig's config, many videos
spintrack run trial3.mp4 -c rig.toml --set tracking.window_px=80  # one key changed for this run
spintrack run trial3_spintrack/config.toml --out redo             # repeat a run
```

Where the config (`-c`) leaves them out, a run finds in the recording:

- **the ball**, with SAM 3 ([configuration](configuration.md#how-the-ball-is-found));
- **the field of view**, by tracking 1000 frames under a range of candidates and keeping the one that fits best, which takes 10-20 s;
- and **the camera's elevation and twist**, from where the animal stands on the ball ([configuration](configuration.md#the-camera-position)).

The one thing a run cannot find is which side the camera films the animal from, its front or its back, so it asks for the azimuth: `--set camera.azimuth_deg=180` for a camera behind the animal, `0` in front, `90` at its right, `-90` at its left. FicTrac requires the same, as `c2a_r`. Put it in the rig's config to run many videos.

The summary at the end says what was found and how, and the run's `config.toml` records it.

Any video FFmpeg reads will do: color or gray, 8 or more bits, any container. Frames are tracked as luma (gray). An AVI is timed and sought by frame count, as it stores no presentation times: FFmpeg's guesses are out of order under B-frames, and its seeks by them can land hundreds of frames off.

Each video gets one folder, `NAME_spintrack` next to it unless `--out` names another (for one video), where NAME is the config's `output.name` or the video's name. It holds `tracks.parquet`, `summary.json`, `log.txt` and `config.toml` ([output.md](output.md)). That `config.toml` is the config as run, with the command line's changes and what the run found, so `spintrack run NAME_spintrack/config.toml --out OTHER`, with the run's `--two-pass` and `--max-frames` if it had them, tracks the same video the same way. A run will not replace an earlier run's files without `--force`. When one of several videos fails, the others still run and the exit status is 1. `--max-frames N` stops early. `--save-map` adds `map.npz`, which `tracking.initial_map=PATH` starts a later run from (as it does a FicTrac sphere-map PNG).

### The live view

A run prints a link, `live view: http://127.0.0.1:8300/?token=...`. Open it to watch the run: the frame with the ball's outline, the animal's trail over the ball and the ignored regions (and the axes, on request), the tracking window, the fictive path, the surface map, the last 10 s of forward, side and turning speed and of the solver's cost, and the summary at the end. Its Stop ends the run as Ctrl-C does, with the records so far written. The page costs the run nothing until it is opened, and little after: it captures at most 12 frames a second, and only while a page asks. On a remote machine, forward the port first, as the line says (`ssh -L 8300:localhost:8300 HOST`). `--no-live` serves nothing; `--port` picks the port (or the next free one after it), and `--host 0.0.0.0` serves the page to other machines directly, still behind the link's token.

What to check, on the page or in the debug video:

- the outline stays on the ball's rim, including when the ball moves in its holder;
- the trail starts under the animal and rides with the texture rather than sliding across it;
- the map's faces show sharp texture, not smears;
- the lighting gain (the map's corner tile) shows the holder's shadow and smooth shading, never texture;
- the path moves only when the animal walks.

### Change the config on the command line

`--set KEY=VALUE`, as often as needed, sets a config key for one run, over the config's: `--set tracking.window_px=80`, `--set camera.position_deg=0,180,0`, `--set output.debug_video=true`, `--set tracking.initial_map=map.npz`. A value is read as TOML, a list may drop its brackets, and `none` restores the default (`--set ball.rim=none` finds the ball again even when the config has one). Setting `camera.position_deg` replaces a config's `camera.rotation`, and the other way around. A misspelled key is an error that names the key it most resembles. Quote a value with brackets or spaces: `--set 'ball.rim=[[1, 2], [3, 4], [5, 6]]'`.

### Two passes

`--two-pass` maps the ball in a throwaway first pass, then tracks again from that map. The second pass starts with the whole ball mapped and the lighting field converged, and if the ball moved in its holder, its window follows a trajectory planned from the whole first pass, without the online follower's delay. It doubles the run time and changes the per-frame error on synthetic truth by under 1%, so use it when the ball moves in its holder or the first seconds matter.

### The debug video

`--debug-video` (or `--set output.debug_video=true`) writes `debug.mp4`: on the left the frame as the live view shows it, on the right the tracking window, the fictive path (scale bar in ball radii), the map unfolded as a dice around the face the camera sees, the lighting gain and the frame's numbers. Arrows show the ball's axes from its center, turning with it, and the animal's from where it stands on the ball: x forward, y left, z the ball's normal there. An axis along the line of sight is a ring, dotted when it points toward the camera and crossed when away. `output.debug_axes=false` leaves them out. When SAM 3 found the ball or placed the camera, an empty tile of the map's net ("segmentation") shows its masks over the first frame, with their scores: the ball's in yellow, as its outline, the animal's in magenta. They are found once, on the opening frames, so the tile does not change.

### Live cameras and streaming

`spintrack run 0 -c config.toml` opens camera 0 through OpenCV; outputs go to `camera_spintrack` in the current directory, or to `NAME_spintrack` for an `output.name`. A live camera cannot be looked at before tracking starts, so the config must describe the ball, the field of view and the camera position: run or open a short recording of the rig first and pass the config it writes. `--two-pass` is refused. For cameras OpenCV cannot open (Basler, FLIR, ...), feed the vendor SDK's frames to the `Tracker` loop in the README.

Records stream in FicTrac's line format (`FT, ` and the 25 fields) with `--udp HOST:PORT`, `--tcp HOST:PORT`, `--serial PORT[:BAUD]` (install the `serial` extra) and `--print`. The config's `stream.udp` and `stream.serial` do what the flags do.

## Fix it in the gui

```bash
spintrack gui trial3.mp4                # saves spintrack.toml next to the video
spintrack gui trial3.mp4 -c rig.toml    # starts from rig.toml and saves to it
```

The gui opens the live view's page in a browser (with `--no-browser`, or without a display, it prints the link), finds what the config leaves out as a run does (without an azimuth, it asks which side the camera films from, and tracks as if from behind until told), and then tracks a clip of the video, 10 s from its start, over and over at its own speed. Every change restarts the clip, so its effect shows within seconds. The panel on the left changes:

- **Ball**: drag the circle's center or edge, click points on the ball's edge, or have SAM 3 find it again.
- **Camera position**: presets (behind, in front, at either side, above), sliders for the elevation, azimuth and twist, or the four corners of a calibration square, clicked in FicTrac's order, which names them in the animal's terms and so holds from any side (for the xy plane: front-left, front-right, back-right, back-left; the gui gives each plane's). Tick *axes* under the frame to check: from where the animal stands, which is where the trail starts, x (red) points forward, y (green) to the animal's left and z (blue) up.
- **Field of view**: type it, or fit it again.
- **Ignored regions**: click a polygon's corners over what moves but is not the ball ([configuration](configuration.md#mask-the-animal)).
- **Tracking**: the parameters of the config's `[tracking]`, each with its default a click away.

Under the frame, pick the clip's start and length and the playback speed; Space pauses and `.` steps a paused clip by a frame. **Save** writes the config, to `-c`'s file or to `spintrack.toml` next to the video, keeping the comments at the file's top, and shows the command that uses it: `spintrack run VIDEO... -c spintrack.toml`. **Track the whole video** saves, then runs as `spintrack run` would, on the same page.

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
- `walking`: where the animal's net walking points ([configuration](configuration.md#the-camera-position)).
- `ball moved`: `no`, or the frames over which the ball moved in its holder and how far, in pixels. The window followed; check those frames on the page or in the debug video.
- `radius`: the silhouette's radius over the first frames against the config's, with a warning beyond 3%.
- `hard tracking`: stretches where both the cost and the solver's iterations rose above the run's baseline. A few short ones in fast turns are normal; a long one means the model stopped fitting the ball.

## Troubleshooting

**"where the camera sits around the animal is unknown".** Add `--set camera.azimuth_deg=180` for a camera behind the animal (`0` in front, `90` at its right, `-90` at its left), or put it in the rig's config.

**"the ball is only N px in radius".** Below 15 px of radius in the image, the reported rotation shrinks without any other sign, starting with the rotation about the optical axis (sideslip, for a camera behind the animal). Put more pixels on the ball.

**No ball found.** SAM 3 and the rim check refused every candidate. Open the gui, drag the circle onto the ball and save, then run with that config.

**"its surface shows little texture to track".** The ball's surface varies by less than 4 gray levels at the tracking window's scale (patterned balls show 10 or more), as plain white polystyrene does: whatever rotation is reported comes from the animal, shadows or noise. Such balls need a pattern, or another sensor (optical mice).

**The field of view cannot be fitted.** The cost leaned to a wide lens without a clear minimum, or the ball turned too little. Write `camera.vfov_deg` from the lens (the sensor height and the focal length give it), or from the rig's calibration.

**The ball moved.** The window follows a ball that sinks or jerks in its holder, so the movement is not read as rotation. If the follower stopped, the window would have left the ball, usually because the look climbed onto the animal: mask the legs, check the page or the debug video, and consider `--two-pass`. `tracks.parquet` has the ball's position on every frame, and `window_offset` estimates the orientation error the window's lag caused on each one ([output.md](output.md#the-balls-position)).

**The radius warning.** The config's ball is more than 3% off the silhouette. The follower uses the measured radius, but the rotation scale comes from the config's: correct the ball in the gui, or run with `ball.rim=none` to detect it.

**The walking line says the camera position is wrong.** Unless the animal walked backward or sideways on purpose, set the azimuth the line suggests, `--set camera.position_deg=0,AZIMUTH,0`, or measure the rig in the gui.

**Dropped frames.** No solve passed the gates: the rotation exceeded `tracking.max_step_rad`, or too little of the window matched the map. Look at those frames for an unmasked animal, blur or a ball that left the window. `global_search = true` recovers by relocalizing, and `max_bad_frames` restarts tracking after that many losses in a row. The row after a gap spans the gap, as its `delta_ts` says.

**"invalid config".** The message lists each wrong key, from the file or from the command line, and what it expected.
