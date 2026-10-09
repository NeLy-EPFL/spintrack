# spintrack

spintrack tracks the 3D rotation of a trackball (spherical treadmill) under a tethered animal from one camera's video, and integrates the animal's fictive path. It streams FicTrac's records, so closed-loop rigs keep working, writes them to Parquet with named columns, and on synthetic scenes its per-frame error is about a tenth of FicTrac's.

## Install

```bash
uv tool install --torch-backend auto git+https://github.com/NeLy-EPFL/spintrack
```

The [installation page](installation.md) covers GPUs, the Rust toolchain, the `serial` extra and updates.

## Track the example

```bash
git clone https://github.com/NeLy-EPFL/spintrack && cd spintrack
spintrack run examples/sample/sample.mp4 --set camera.azimuth_deg=180
spintrack gui examples/sample/sample.mp4
```

The run finds the ball, the field of view and the camera's elevation in the video; which side the camera films the fly from (`180`: behind it) is the one thing to say. It prints a link to a live view of the run, and writes `tracks.parquet`, a summary and the config as run into `examples/sample/sample_spintrack/`. The [quickstart](quickstart.md) goes through it step by step.

## Where next

- [User guide](guide.md): from a recording to a tracked file, and how to tell whether to trust it.
- [Configuration](configuration.md): every config key, how the ball is found, and the camera position.
- [Output files](output.md): every column, with its unit, frame and sign.
- [Coming from FicTrac](fictrac.md): config keys, records, and what differs.
- [Command line](cli.md) and [Python API](api.md) references.
