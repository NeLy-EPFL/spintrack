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

- `spintrack run`, `calibrate` and `map`. `calibrate --auto` detects the ball and fits `vfov` from a recording, and `--c2a-angles` writes `c2a_r` from the camera's position, both without a window.
- Outputs: FicTrac's 25-column `.dat`, a Parquet copy with named columns and units, a JSON summary of the run's quality and inputs, and an optional annotated debug video. Existing outputs are never overwritten without `--overwrite`.
- UDP, TCP and serial streaming in FicTrac's line format.
- Python API: `spintrack.track` for a whole recording and `spintrack.Tracker` for one frame at a time.

### Compatibility with FicTrac

- Reads FicTrac configs (and YAML or TOML) and writes FicTrac's columns, with path integration ported line for line. `c2a_r` is required.
- Rotation columns are in the true camera and lab frames, which FicTrac's are not; column 5 is a photometric residual. See `docs/fictrac.md` and `docs/output.md`.

### Packaging

- Built with maturin; Python 3.11 or newer. Installed from source until wheels are published with this release.
