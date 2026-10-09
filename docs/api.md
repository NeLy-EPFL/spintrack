# Python API

`spintrack.track` runs what `spintrack run` does and returns the records instead of writing files:

```python
import spintrack

track = spintrack.track("examples/sample/config.toml")  # the run, without the files
df = track.to_polars()  # columns by name: forward_total, heading, ...
print(track.quality.n_dropped, df.select("forward_total", "heading").tail())
```

For a live camera, feed frames from your own capture code to a `Tracker`. The config must already describe the ball, the field of view and the camera position, so track a recorded clip of the rig first and use the `config.toml` the run writes.

```python
from spintrack import Config, Tracker

cfg = Config.load("config.toml")
tracker = Tracker(cfg, width, height)
for gray, ts_ms in frames():  # your camera SDK: 2-D uint8 images, timestamps in ms
    res = tracker.process_frame(gray, ts_ms)
    if res is not None:  # None: the frame could not be tracked
        print(res.forward, res.side, res.turn)  # rad this frame; side, turn: left +
```

## Reference

::: spintrack.track

::: spintrack.Track

::: spintrack.Config

::: spintrack.Tracker

::: spintrack.FrameResult

::: spintrack.TrackParams
