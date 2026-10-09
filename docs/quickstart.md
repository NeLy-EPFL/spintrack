# Quickstart

The repository includes a 10.5 s example: frames 850-1899 of trial ANXXX049_251125_Fly1_003, a fly filmed by a camera behind it, at half resolution. The fly walks and turns enough in it to map the whole ball. A second example, `examples/ball_drop/`, shows the tracking window following a ball that sinks in its holder and comes back up. In a clone of the repository:

```console
$ spintrack run examples/sample/sample.mp4 --set camera.azimuth_deg=180
live view: http://127.0.0.1:8300/?token=... (from another machine: ssh -L 8300:localhost:8300 HOST)
loading SAM 3 on cuda (the first use downloads 3.4 GB)
fitting the field of view: tracking 1000 frames at each candidate
spintrack 0.1.0: examples/sample/sample.mp4 (800x504, 100 fps, 1050 frames)
done: 1050 frames, 0 dropped, 1.9 s (559 fps)
run quality: 1050 frames, 1050 tracked, 0 dropped
ball: detected at (400.5, 371.5) r 260.8 px by SAM 3 (score 0.75), rim confirms 86% of its outline
vfov: 3.31 deg (fitted; not identifiable, the cost is flat over 1-11 deg, over which the rotation scale changes by 6%)
camera position: azimuth 180 from camera.azimuth_deg; elevation 0, twist -0.6 deg from where the animal stands (it stands on the ball's outline, where the elevation does not show and level is assumed; SAM 3 score 0.87)
walking: net 3.9 ball radii, 7 deg right of forward
ball moved: no
radius: silhouette 258.3 px, config 260.1 px (-0.7%)
hard tracking: none
wrote tracks.parquet, summary.json, log.txt, config.toml in examples/sample/sample_spintrack
```

A run finds the ball (with SAM 3), fits the field of view and takes the camera's elevation and twist from where the fly stands; the summary says what it found. What a video cannot tell is the fly's front from its back, so say where the camera sits around the fly, as FicTrac also requires: `--set camera.azimuth_deg=180` behind it, `0` in front, `90` at its right, `-90` at its left. A rig's config (`-c`) can hold it. A [deeperfly](https://github.com/NeLy-EPFL/deeperfly) project tracks its ball with deeperfly's `[ball]` stage instead, which runs spintrack with the project's calibration and 3D pose and writes the ball in the project's coordinates; no azimuth is needed there.

The first line's link opens the live view of the run: the frame with the ball and the fly's trail over it, the tracking window, the surface map, the path and the speeds. It costs the run nothing until opened. The run's folder, `examples/sample/sample_spintrack/`, holds `tracks.parquet` (one row per frame, FicTrac's 25 columns by name and the ball's position), `summary.json`, `log.txt` and `config.toml`, the config as run; [output files](output.md) describes them.

## Fix it in the gui

When the run gets something wrong, fix it while watching the tracking:

```bash
spintrack gui examples/sample/sample.mp4
```

The gui tracks a 10 s clip over and over while you drag the ball's outline, set the camera position, mask the fly and change the tracking parameters, and saves them as a config (`spintrack.toml` next to the video). Use that config for every video of the rig, and change any key for one run with `--set`:

```bash
spintrack run session/*.mp4 -c spintrack.toml
spintrack run trial3.mp4 -c spintrack.toml --set tracking.window_px=80 --debug-video
```

The [user guide](guide.md) explains each step.
