# Example recording

`sample/sample.mp4` is 10.5 s (1050 frames, 100 fps) of a tethered fly walking on a ball, seen by the hind camera of the lab's eight-camera rig: frames 850-1899 of trial `ANXXX049_251125_Fly1_003`, `camera_H`, downscaled 2x to 800x504 (`INTER_AREA`) and re-encoded with x264 at CRF 20. It was picked among the rig's recordings as the shortest stretch whose walking maps the whole ball (99% of the map) while the ball stays put in its holder. At this size it tracks like the full-resolution original: per-frame increments correlate at 0.999 or better and agree in scale to 0.4% (sideslip, the component read from the rim, loses the most to the halved resolution).

```bash
spintrack run examples/sample/config.txt --debug-video
```

writes `sample.dat`, `sample.parquet`, `sample-summary.json` and `sample-debug.mp4` next to the config (they are git-ignored). Add `--overwrite` to run it again.

## The config

- `vfov : 2.3893` comes from the rig's multi-camera calibration (focal length 24168.6 px at 1008 rows: `2 atan(504 / f)`). Downscaling does not change it. The view is nearly orthographic, so the ball spans only 1.2 deg of half-angle.
- `roi_circ` are rim points of the ball detected on the full-resolution video (`spintrack calibrate --auto`), halved. The bottom of the ball is cut off by the image edge.
- `c2a_r` assumes a camera directly behind the animal, level with the ball center (`spintrack calibrate config.txt --c2a-angles 0 180 0`). The camera-frame columns do not depend on it; the lab-frame columns, the path and the trail in the debug video do.
