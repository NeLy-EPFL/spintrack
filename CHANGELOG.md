# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (unreleased)

First release: a FicTrac-compatible trackball tracker with a Rust core.

### Tracking

- Per-frame photometric Gauss-Newton alignment of the tracking window against a floating-point surface map of the ball, with robust weights, a pyramid fallback, constant-velocity prediction and optional global relocalization.
- An equi-angular cubemap surface map; FicTrac sphere-map PNGs load and are converted.
- Static illumination fields that keep the rig's fixed lighting out of the map: an additive bias, and a gain for the texture contrast a shadow takes away. Map updates are weighted by that gain and by the viewing angle, so dim and foreshortened views do not wash out the map.
- A ball follower that moves the tracking window with a ball that sinks or jerks in its holder, so the movement is not read as rotation. `tracks.parquet` records the ball's position on every frame (`ball_x_px`, `ball_y_px`, `ball_seen`) and the window's offset from it (`window_offset`), about the orientation error it causes, which the photometric error cannot show.
- `--two-pass`: map the ball in a first pass, then track from that map, with the window following a trajectory planned from the whole recording.

### Interfaces

- Two commands. `spintrack run VIDEO...` tracks videos given only which side the camera films from (`camera.azimuth_deg`): it detects the ball, fits the field of view and takes the camera's elevation and twist from where the animal stands, when the config (`-c`) does not say. `KEY=VALUE` arguments override any config key for one run. `spintrack VIDEO` is short for `spintrack run VIDEO`.
- The field of view is fitted from the photometric cost. A lean toward a wide lens without a clear minimum, after the ball turned a few hundred degrees, is taken for lens distortion rather than a wide lens: a narrow lens is assumed and the summary says how much smaller the rotations would read at the lean. With too little rotation the fit tracks further into the video, and a run refuses rather than guess.
- A ball with too little texture to track (plain polystyrene) is said out loud.
- Any video FFmpeg reads, color or gray; AVI files are timed and sought by frame count, which B-frames make FFmpeg's timestamps fail at.
- `spintrack gui VIDEO`: a page that tracks a clip of the video live while the ball, the camera position, the field of view, the ignored regions and the tracking parameters are changed, saves them as a config, and tracks the whole video with it.
- A preview of every run, in the browser: the frame with the ball, the animal's trail and the ignored regions, the tracking window, the map, the path, the speeds and the cost, and a Stop. `run` prints its link; it costs nothing until opened.
- `track(config, frames=...)` tracks frames the caller decodes, from a complete config. deeperfly's `[ball]` stage drives spintrack this way: it places the camera from the project's calibration and the fly's leg tips on the ball, and writes the ball in the project's world frame. spintrack reads nothing of deeperfly's.
- The camera position from the animal: SAM 3 finds the animal on the ball, in three frames, as the mask that stays put while the ball turns (a spot of the pattern, or a patterned ball, can be the best "insect"). Its direction from the ball's center gives the twist, and its distance the elevation when it is inside the outline; on the outline, level is assumed, and from nearly straight above the twist is not read. The azimuth, which side the camera films from, is the one thing to give (`camera.azimuth_deg`), unless a deeperfly project says it; a run without it refuses, as FicTrac does without `c2a_r`. Every run checks the azimuth against the animal's net walking.
- Ball detection with SAM 3: the model proposes the ball's silhouette and the rim is measured and checked on the image. It found all 225 balls of 266 recordings from six rigs and refused the 41 without a usable ball; the classical detector, used when the model cannot be loaded, found 108.
- A TOML config with readable names in tables (`camera`, `ball`, `mask`, `tracking`, `output`, `stream`), validated on load with pydantic: an unknown key is an error that names the key it most resembles. Paths are relative to the config. The camera-to-animal transform is the camera's position (`camera.position_deg`) or a rotation vector (`camera.rotation`), and the ball is its rim points (`ball.rim`).
- Outputs: one folder per run, `NAME_spintrack` next to the video, holding the records as Parquet with named columns and units (`tracks.parquet`), a JSON summary of the run's quality and inputs, the run's log, the config as run (`config.toml`), and an optional annotated debug video. Existing outputs are never overwritten without `--overwrite`.
- UDP, TCP and serial streaming in FicTrac's line format.
- Python API: `spintrack.track` for a whole recording (`to_polars()` for a DataFrame) and `spintrack.Tracker` for one frame at a time.

### Compatibility with FicTrac

- Streams FicTrac's 25-field records in FicTrac's line format, with path integration ported line for line. It reads no FicTrac `config.txt` and writes no `.dat` file; `docs/fictrac.md` translates both, key by key and column by column.
- The animal frame is x forward, y left, z up, as in deeperfly and flygym, where FicTrac's is y right, z down: in `tracks.parquet` and the Python API, sideways motion, turning, the heading and the path's y are positive to the left. The streams are converted to FicTrac's signs. `spintrack.calibrate.sliders.rotation_from_fictrac` converts a `c2a_r`.
- Rotation columns are in the true camera and lab frames, which FicTrac's are not; column 5 is a photometric residual. See `docs/fictrac.md` and `docs/output.md`.

### Packaging

- Built with maturin; Python 3.14 or newer. Installed from source until wheels are published with this release; `uv tool install --torch-backend auto` is the recommended way, picking the PyTorch build for the machine's GPU.
