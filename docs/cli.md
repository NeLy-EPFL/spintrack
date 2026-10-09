# Command line

Track the rotation of a trackball from video.

**Usage**:

```console
$ spintrack [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`: Show the version and exit.
* `--help`: Show this message and exit.

**Commands**:

* `run`: Track the ball in each video.
* `gui`: Fix a video's config while it tracks.
* `doctor`: Report what this installation can do.

## `spintrack run`

Track the ball in each video.

The ball, the field of view and the camera position come from the config (`-c`) when it has them, and from the recording when not. `--set` changes the config's keys for this run.

Each video gets a folder, `NAME_spintrack` next to it (or `--out`), with `tracks.parquet`, `summary.json`, `log.txt` and `config.toml`, the config as run. The live view, whose link is printed first, shows the run as it goes.

**Usage**:

```console
$ spintrack run [OPTIONS] [videos]...
```

**Arguments**:

* `videos...`: Videos or camera indices, or configs (`.toml`) that name their video.

**Options**:

* `-c, --config CONFIG`: A config (`.toml`) to start from, such as one `spintrack gui` saved.
* `--set KEY=VALUE`: Set a config key, such as `tracking.window_px=80` or `camera.position_deg=0,180,0`; `none` restores its default. Repeatable.
* `-o, --out DIR`: The output folder, for one video (default: `NAME_spintrack` next to it).
* `--force`: Replace the outputs of an earlier run.
* `--two-pass`: Map the ball in a first pass, then re-track from that map.
* `--max-frames N`: Stop after N frames.  [x>=1]
* `--debug-video`: Also write an annotated video, `debug.mp4` (`--set output.debug_video=true`).
* `--save-map`: Also write the final map, `map.npz`.
* `--no-live`: Serve no live view (by default its link is printed).
* `--host HOST`: The interface to serve the page on; by default, this machine.  [default: 127.0.0.1]
* `--port PORT`: The page's port (default: 8300). A busy port is passed over for the next free one.  [1<=x<=65535]
* `--udp HOST:PORT`: Stream records over UDP.
* `--tcp HOST:PORT`: Stream records over TCP.
* `--serial PORT[:BAUD]`: Stream records over a serial port (the `serial` extra).
* `--print`: Print records to the terminal.
* `-v, --verbose`: Debug messages, and a traceback on errors.
* `--help`: Show this message and exit.

## `spintrack gui`

Fix a video's config while it tracks.

Opens a page that tracks the video live while you fix the ball, the camera position and the tracking parameters, and saves them as a config for `spintrack run -c CONFIG`.

**Usage**:

```console
$ spintrack gui [OPTIONS] [video]
```

**Arguments**:

* `video`: The video or camera index, or a config (`.toml`) that names its video.

**Options**:

* `-c, --config CONFIG`: A config (`.toml`) to start from, such as one `spintrack gui` saved.
* `--set KEY=VALUE`: Set a config key, such as `tracking.window_px=80` or `camera.position_deg=0,180,0`; `none` restores its default. Repeatable.
* `--host HOST`: The interface to serve the page on; by default, this machine.  [default: 127.0.0.1]
* `--port PORT`: The page's port (default: 8300). A busy port is passed over for the next free one.  [1<=x<=65535]
* `--no-browser`: Print the page's link without opening a browser.
* `-v, --verbose`: Debug messages, and a traceback on errors.
* `--help`: Show this message and exit.

## `spintrack doctor`

Report what this installation can do.

Checks Python, the compiled core, PyTorch and the device SAM 3 runs on, whether SAM 3's checkpoint is cached, PyAV's FFmpeg and the serial extra. It downloads nothing. Exits with status 1 on errors (and on warnings with `--check`).

**Usage**:

```console
$ spintrack doctor [OPTIONS]
```

**Options**:

* `--json`: Print the report as JSON, for scripts, instead of text.
* `--check`: Exit with status 1 on warnings too, not only on errors.
* `-v, --verbose`: Debug messages, and a traceback on errors.
* `--help`: Show this message and exit.
