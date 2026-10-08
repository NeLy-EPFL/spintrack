# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (unreleased)

First release: a FicTrac-compatible trackball tracker with a Rust core.

### Tracking

- Per-frame photometric Gauss-Newton alignment of the tracking window against a floating-point surface map of the ball, with robust weights, a pyramid fallback, constant-velocity prediction and optional global relocalization.
- An equi-angular cubemap surface map; FicTrac sphere-map PNGs load and are converted.
- Static illumination fields that keep the rig's fixed lighting out of the map: an additive bias, and a gain for the texture contrast a shadow takes away. Map updates are weighted by that gain and by the viewing angle, so dim and foreshortened views do not wash out the map.
- A ball follower that moves the tracking window with a ball that sinks or jerks in its holder, so the movement is not read as rotation.
- `--two-pass`: map the ball in a first pass, then track from that map, with the window following a trajectory planned from the whole recording.

### Interfaces

- `spintrack run`, `calibrate` and `map`. `run` takes a config, or a video with `--config`, and `spintrack VIDEO` is short for `spintrack run VIDEO`. `calibrate --auto` detects the ball and fits the field of view from a recording, and `--camera-position` writes where the camera sits, both without a window; `run --camera-position` overrides the config's for one run.
- Ball detection with SAM 3 (the `sam` extra): the model proposes the ball's silhouette and the rim is measured and checked on the image. It found all 225 balls of 266 recordings from six rigs and refused the 41 without a usable ball; the classical detector, used without the extra, found 108.
- A TOML config with readable names in tables (`camera`, `ball`, `mask`, `tracking`, `output`, `stream`), validated on load with pydantic: an unknown key is an error that names the key it most resembles. Paths are relative to the config. The camera-to-animal transform is the camera's position (`camera.position_deg`) or a rotation vector (`camera.rotation`), and the ball is its rim points (`ball.rim`).
- Outputs: one folder per run, `NAME_spintrack` next to the video, holding the records as Parquet with named columns and units (`tracks.parquet`), a JSON summary of the run's quality and inputs, the run's log, the config as run (`config.toml`), and an optional annotated debug video. Existing outputs are never overwritten without `--overwrite`.
- UDP, TCP and serial streaming in FicTrac's line format.
- Python API: `spintrack.track` for a whole recording (`to_polars()` for a DataFrame) and `spintrack.Tracker` for one frame at a time.

### Compatibility with FicTrac

- Streams FicTrac's 25-field records in FicTrac's line format, with path integration ported line for line. It reads no FicTrac `config.txt` and writes no `.dat` file; `docs/fictrac.md` translates both, key by key and column by column. A camera-to-animal transform is required.
- Rotation columns are in the true camera and lab frames, which FicTrac's are not; column 5 is a photometric residual. See `docs/fictrac.md` and `docs/output.md`.

### Packaging

- Built with maturin; Python 3.14 or newer. Installed from source until wheels are published with this release.
