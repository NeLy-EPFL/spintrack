# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Output

- Each run writes its records as `<out>.parquet` beside the `.dat`: the same 25 columns, named and typed, with units in the field metadata and the version and provenance in the file metadata. `--no-parquet` turns it off. `pyarrow` is now a dependency.

### Tracking

- The tracking window keeps up with a ball that drops in its holder. Online, a move is detected on the rim looks themselves rather than on a slow filter of them, and the window follows an adaptive line fit of the recent looks instead of an alpha-beta filter. With `--two-pass`, the planned window is smoothed with a width that shrinks through a jerk instead of a fixed six-frame Gaussian. Each frame is now tracked in the window placed for it, not in the previous frame's. On trial 004's jerky drop the window's p95 distance from the ball goes from 42 to 5 px online and from 15 to 3 px with `--two-pass`. Recordings whose ball does not move track exactly as before.
- A resting ball that jumps farther in one frame than its rim look reaches is found again by the seed-independent detection, which used to run only while the window was already following.
- The benchmark has three new scenes, `jerky_drop`, `jerky_diag` and `lab_jerky_drop`, in which the ball drops in jerks of about a tenth of its radius over ten frames, as on trial 004; `ball_drop` moves too slowly for any follower to lag. The scene generator takes a `steps` ball path for them.

## 0.1.0

First release.

### Tracking

- Photometric estimation of the ball's 3D rotation against an accumulated surface
  map of its texture, solved per frame on an image pyramid. See `docs/algorithm.md`.
- An equi-angular cubemap surface map by default, which removes the wide cells
  FicTrac's equal-area cylindrical grid puts at its poles. `--map-projection
  equal_area` selects FicTrac's grid; maps convert between the two on load, so a
  FicTrac template and a map saved by either projection all still load.
- A static camera-frame illumination correction, so fixed lighting does not enter
  the ball's texture. `--no-illumination` turns it off; `--load-illumination`
  carries a rig's field between recordings.
- `--two-pass` builds the surface map and then tracks against it in one command;
  `--refine N` re-estimates offline. `--save-map`, `--load-map` and `--frozen-map`
  manage the map directly.
- Per-frame quality reporting, and a warning when the detected ball disagrees with
  the configured radius or is smaller than 15 pixels in the source image, below
  which the reported rotation shrinks silently.

### Compatibility with FicTrac

- Reads FicTrac configuration files and writes its 25-column `.dat`, socket and
  serial output. `docs/migration-from-fictrac.md` has the mapping and the
  differences, of which two are deliberate: the error score in column 5 is a
  weighted mean squared residual rather than a binary-match error, and columns
  2-4 and 9-11 are true camera-frame vectors rather than tracking-window ones.
- **Turning does not come out identical to FicTrac's.** On six 60 s trials of one
  rig spintrack reads 1.5 to 4% less turning on the five the animal walked through
  and 9% less on the sixth. Neither tracker is known to be the right one. Do not
  pool turning across the switch; see `docs/migration-from-fictrac.md`.

### Verification

- `docs/verification.md` states what is checked against what, what the synthetic
  benchmark structurally cannot see, and what is left for a caliper rather than
  for code.
- `docs/benchmark.md` reports precision and per-component gain against exact
  synthetic truth. Its gains come from one activity level, 5 to 12 times more
  active than the recordings the tool was built for.

### Packaging

- Prebuilt wheels for Linux, macOS and Windows; no CMake, OpenCV, NLopt or Boost
  to build. Rust extension built with maturin, Python 3.11 and newer.
