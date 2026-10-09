# Configuration

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

## How the ball is found

Detection looks at the per-pixel 90th percentile of the first 100 frames: as the ball turns, its dark markings pass over every pixel of it, so the high percentile shows the ball's plain surface while the static background stays as it is. Two detectors work on that image:

- By default, SAM 3 is asked for a "ball" and a "sphere". Its masks only say where the ball is: the rim is measured on the image in a narrow band around each mask, and a candidate is kept only if the rim confirms enough of the outline the mask shows, fits a circle tightly, stands out of the noise and agrees with the mask; among those, SAM's score decides. On 266 recordings from six rigs this found all 225 balls without a wrong one and refused all 41 recordings with no usable ball in view. It takes 0.3 s on a GPU or about 13 s on a CPU, once per recording. The checkpoint (3.4 GB) downloads on first use, without a Hugging Face login, from [an ungated copy](https://huggingface.co/tkclam/sam3) of Meta's [gated original](https://huggingface.co/facebook/sam3); its files are byte-identical to Meta's. Its [license](https://github.com/facebookresearch/sam3/blob/main/LICENSE) asks publications that use it to acknowledge SAM.
- When the checkpoint cannot be loaded (offline before its first download), a classical detector thresholds the image instead. It is as precise when it answers, but on the same recordings it found only 108 of the 225 balls, refusing the rest.

Both assume a ball lighter than its markings, as on every rig tested, and a run refuses a recording where neither is sure rather than guess; place the ball in the gui then. A ball that moves in its holder during the first frames reads slightly large; the summary's radius line shows it. The field of view is fitted with the ball's outline held fixed. On a long lens the fit is flat, and the summary says how much the rotation scale changes over the flat range (a few percent at most on the lab's rigs). A wider lens shows a clear minimum once the ball has turned a few hundred degrees. A lean toward a wide lens without one is not taken for a wide lens: lens distortion causes it too (a lab octacam with a 2 deg lens leaned to 26 deg and would have read rotations 18% low). A narrow lens is assumed then, and the summary says how much smaller the rotations would read at the vfov the cost leaned to. When the narrow end clearly loses, or the ball turned too little to tell, a run refuses. Rotations read up to tens of percent smaller through a wide lens than through a narrow one at the same image motion, so if you know the field of view, write it, above all for webcams and phones.

## The camera position

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

Without either, `camera.azimuth_deg` says which side the camera films from, and SAM 3 is asked for the "insect" in three frames, whose silhouette gives the rest. A tethered animal stays put while the ball turns, so the animal is the mask found at the same place in two of the three (a spot of the ball's pattern can be the best "insect" in one frame), and a mask over most of the ball is the ball (a cow-patterned one is an "insect" too). "insect" finds crabs as well. The animal frame's up is the ball's normal where the animal stands, so the silhouette's direction from the ball's center gives the twist, and its distance the elevation: inside the outline, at the elevation's cosine, for a camera above. On the outline the elevation does not show, because the animal's height above the ball outweighs the elevation's cosine there, and level is assumed. On the lab's octacam, where the fly's leg tips put the hind camera 10.5-12 deg above the fly's horizon ([below](#from-deeperfly)), the silhouette looks as it would from a level camera, while the twist it gives agrees with the leg tips' to 0.5 deg. An elevation error mixes turning and sideways rotation by about its sine (17% at 10 deg), so on such a rig give the elevation (`camera.position_deg`) or track the ball through deeperfly ([below](#from-deeperfly)). From nearly straight above (the animal within a quarter radius of the ball's center), the elevation still shows but the twist does not: none is assumed, and the azimuth says which way the animal faces in the image (180 up, 90 right, 0 down, -90 left). No animal found leaves the camera level, and the summary says so. Without any of these, a run refuses, before any slow work.

Every run then checks the azimuth against the animal's net walking, since animals walk mostly forward: the summary's `walking` line says where the net walking points, and when it points more than 45 deg from forward over at least 10 ball radii, it says which azimuth would make it forward. It is a check for gross errors, not a measurement: on 48 octacam trials with the azimuth measured against the fly's body axis, most flies walked under half a ball radius net in 20 s, and one that walked 5 radii did so 106 deg from its body axis. An experiment that makes the animal walk backward reads as 180 deg off; read the line knowing the experiment.

Neither the silhouette nor the walking can tell the azimuth: on octacam trials, SAM 3's "eye", "wing" and "abdomen" fired on flies seen from behind as often as on flies seen from the front. A nominal azimuth is good to a few degrees, though: on 48 octacam trials, the fly's body axis was 2.1 deg (median) and at most 7.6 deg from straight behind the camera that filmed it from behind. For the precise value, track the ball through deeperfly, or measure it in the gui: the sliders with the axes drawn on the ball, or the corners of a calibration square aligned with the animal's axes, as FicTrac's configGui does.

## From deeperfly

A rig calibrated by [deeperfly](https://github.com/NeLy-EPFL/deeperfly) tracks its ball through deeperfly, whose `[ball]` stage (the `deeperfly[ball]` extra) runs spintrack on one view's video and writes the ball's center, its rotation and the fly's axes on it in the project's world frame. It hands spintrack the view's field of view and a camera placed relative to where the fly stands, rather than to the rig: flies are never mounted exactly along the rig's axis (on 48 octacam trials, their bodies pointed 2.3 deg (median) and up to 11.7 deg off it), nor on the ball's top. The ball's center is on the ray through the center of the ball's image, at the distance that rests the most leg tips (the legs in stance) on its surface; the fly's up is the ball's normal through its thorax-coxa points, and forward its body's long axis, from the hind to the front coxae, laid on the ball there. On four octacam trials the fly stood 9-13 deg from the ball's top, toward the hind camera, 73-84% of its leg tips rested on the fitted ball, and the camera behind it sat 10.5-12 deg above its horizon. spintrack itself reads nothing of deeperfly's: deeperfly decodes the frames and calls `spintrack.track(config, frames=...)` with a complete config.

## Mask the animal

The solver assumes that everything in the tracking window turns with the ball. The animal does not, so cover it with `mask.ignore` polygons in image pixels, `ignore = [[[611, 348], [912, 337], [880, 450]], ...]`, drawn in the gui or written by hand. Cover the legs' whole reach, not just the body: the robust weights reject legs only at the cost of the texture behind them, and legs at the rim also pull the ball follower's silhouette look. The page and the debug video outline the ignored regions in red.
