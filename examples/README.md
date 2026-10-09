# Example recordings

`sample/sample.mp4` is 10.5 s (1050 frames, 100 fps) of a tethered fly walking on a ball, seen by the hind camera of the lab's eight-camera rig: frames 850-1899 of trial `ANXXX049_251125_Fly1_003`, `camera_H`, downscaled 2x to 800x504 (`INTER_AREA`) and re-encoded with x264 at CRF 20. It was picked among the rig's recordings as the shortest stretch whose walking maps the whole ball (99% of the map) while the ball stays put in its holder. At this size it tracks like the full-resolution original: per-frame increments correlate at 0.999 or better and agree in scale to 0.4% (sideslip, the component read from the rim, loses the most to the halved resolution).

```bash
spintrack run examples/sample/config.toml --debug-video
```

writes `tracks.parquet`, `summary.json`, `log.txt`, `config.toml` and `debug.mp4` into `examples/sample/sample_spintrack/` (git-ignored). Add `--force` to run it again.

## The config

- `vfov_deg = 2.3893` comes from the rig's multi-camera calibration (focal length 24168.6 px at 1008 rows: `2 atan(504 / f)`). Downscaling does not change it. The view is nearly orthographic, so the ball spans only 1.2 deg of half-angle.
- `rim` are points on the ball's rim detected on the full-resolution video (as `spintrack run` detects it), halved. The bottom of the ball is cut off by the image edge.
- `position_deg = [0, 180, 0]` places the camera directly behind the animal, level with the ball center. The camera-frame columns do not depend on it; the lab-frame columns, the path and the trail on the page and in the debug video do.

## A ball that moves in its holder

`ball_drop/ball_drop.mp4` is 11 s (1100 frames) of the same rig, prepared the same way: frames 950-2049 of trial `AN07B017_260414_Fly4_004`. A second into it the ball sinks in its holder, in three jerks and by about 100 px (206 px at full resolution), away from the fly's legs, then rises back to where it started.

```bash
spintrack run examples/ball_drop/config.toml --debug-video
```

The run summary reports the move (`ball moved: frames 149-866 (103 px)`), and in the debug video the green outline stays on the ball throughout: the tracking window follows it, so its movement is not read as rotation. `rim` are the trial's hand-clicked rim points, halved.
