# First run: a rig with no FicTrac config

You have a camera, a ball under a tethered animal, and a recorded clip. This page gets you
from there to a tracked `.dat`, and tells you which numbers in the run summary to trust.
It assumes no FicTrac config; if you have one, `spintrack run config.txt` already works and
[migration-from-fictrac.md](migration-from-fictrac.md) is the page you want.

## 1. A minimal config

spintrack reads FicTrac `config.txt` files. The smallest one that tracks names the clip and
the camera's vertical field of view:

```
src_fn  : clip.mp4
vfov    : auto      # degrees; `auto` fits it from the recording
src_fps : 100
```

`vfov : auto` is fitted from the photometric cost, but only when the view has enough
perspective to constrain it; a near-orthographic rig (the ball small on the sky, a long
lens) leaves the cost flat, and the run says so instead of inventing a number. If you know
the field of view, write it.

## 2. Find the ball and set the animal frame

```
spintrack calibrate config.txt --auto --c2a-angles 0 180 0
```

`--auto` detects the ball's circle in the first frames (and fits `vfov` if it was `auto`)
and writes them into the config, with no window. `--c2a-angles ELEV AZIM TWIST` writes
`c2a_r`, the rotation from the camera frame to the animal's, from where the camera sits in
degrees: a camera directly behind the animal and level with the ball is `0 180 0`. `c2a_r`
is required. Without it the forward/side and heading columns would be camera-frame numbers
wearing the animal's labels; `c2a_r : { 0, 0, 0 }` says "the camera frame *is* the animal
frame" on purpose.

Detection refuses rather than guesses when it is unsure, so a written config is one it was
confident about. If `--auto` cannot find the ball, or you want to set the ignore regions
and animal axes by eye, `spintrack calibrate config.txt` (no `--auto`) opens a window to
click the rim, the regions to ignore, and the forward and side axes.

## 3. Track

```
spintrack run config.txt --debug-video run.mp4
```

Watch `run.mp4`: the orientation axes should stay planted on the ball, and the fictive path
should move only when the animal walks. Once that looks right, `--two-pass` maps the ball in
a throwaway first pass so the opening frames see a whole ball, and `--refine 2` re-estimates
each frame offline; neither removes drift over the recording.

## 4. Read the summary, and when to distrust it

Every run prints a quality block and writes `<out>-summary.json`. The fields that decide
whether a recording is trustworthy:

- **Dropped frames and cost.** `n_dropped` should be 0, and `cost_p90`/`cost_p99` should sit
  near `cost_baseline`. Stretches where both the cost and the solver's iteration count rise
  together are listed as episodes with their frame ranges: a few during fast turns are
  normal, a long one means the model stopped fitting the ball there.
- **Map coverage.** The fraction of the ball's surface the run has mapped. Low early is
  expected; `--two-pass` starts a run from a full map.
- **Rotation scale** (`radius_err_pct`, verdict `ok`/`warn`/`fail`). Whether the assumed ball
  radius agrees with what the inner and outer parts of the window each imply about the
  motion. `warn` or `fail` means the radius is probably wrong, and a radius error costs about
  twice as much of every speed reported about an axis in the image plane (so a camera behind
  the animal sees it mostly on forward walking). **Its absolute zero is not independently
  established:** the holder shadow biases the reading by about half a point of radius, so the
  check cannot confirm a radius to better than roughly 1% on its own. A caliper and the
  camera distance are the real measurement. `insufficient motion` means the clip did not turn
  enough to say anything.
- **Whether the ball moved.** `stable`, or `followed a moving ball over N frames` if it
  shifted in its holder. When it moved, the tracking window follows it rather than turning
  the shift into rotation.
- **A small ball warns.** Below about 15 px of ball radius in the source image the reported
  rotation shrinks and little else in the run says so, so the run prints a warning; the one
  about the optical axis (sideslip, for a camera behind the animal) goes first.

## 5. Mask the animal

The rotation-scale check and the solver both assume everything in the tracking window turns
with the ball. The animal's body and legs do not. Cover them with an ignore region
(`roi_ignr`, from the calibrator's ignore step or as a polygon in the config). It has to
cover the **legs**, not just the body: on the `occluded` benchmark scene a mask that stopped
at the body read the radius up to about two points too large, and widening it over the legs
cut that by roughly two-thirds. Legs left in the window also hold the solve back where they
cross the rim.

See [algorithm.md](algorithm.md) for how the tracking works, and
[verification.md](verification.md) for what has and has not been checked against ground
truth.
