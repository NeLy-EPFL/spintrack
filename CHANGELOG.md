# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Nothing yet.

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
