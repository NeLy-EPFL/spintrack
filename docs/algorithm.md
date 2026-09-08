# How spintrack tracks the ball

spintrack estimates the 3D rotation of a patterned sphere from a single fixed camera. It
shares FicTrac's problem statement and interfaces, but not its estimator: instead of
matching a thresholded window against a binary map with a derivative-free search, it aligns a
photometrically normalized window against a floating-point surface map with Gauss-Newton
iterations and analytic derivatives.

## Geometry (once per configuration)

- **Camera.** Pinhole from the vertical field of view (`vfov`, degrees), or an equidistant
  fisheye (`fisheye: y`). Camera axes: x right, y down, z forward.
- **Ball.** A unit direction to the ball centre and its angular radius (`roi_c`, `roi_r`), or a
  least-squares cone fit through the rim points (`roi_circ`). Pixels inside `roi_ignr`
  polygons are ignored.
- **Tracking window.** A virtual fisheye camera aimed at the ball centre samples the source
  image into an `n x n` window (`n = 10 * q_factor`) with `cv2.remap`. A window pixel covers
  6-9 source pixels on these rigs, and bilinear sampling at that decimation aliases the
  texture, so the source is pre-filtered first: one `pyrDown` and a Gaussian on the half-size
  frame, a total blur of half the decimation (`sphere.PREFILTER_SIGMA`), which halves the
  per-frame error on the synthetic scenes. Each window pixel that sees the ball gets a unit
  vector `v_k` from the ball centre to the surface point it observes (window frame: z toward
  the ball). These vectors are fixed; rotating the ball rotates them.
- **Surface map.** A float32 grid over the ball surface storing a running weighted mean of
  normalized intensity plus a weight per cell (0 = never seen). Two tessellations, both
  addressed by a continuous `(u, v)` into one `(h, w)` array. `cube` (the default) is an
  equi-angular cubemap, six square faces stacked into `(6 face, face)`. `equal_area` is
  FicTrac's Lambert cylindrical grid: `u` from the longitude about the window y axis, `v`
  from `sin(latitude)`. See "The shape of the map's cells" below.

## Per frame

1. **Normalize.** Local zero-mean, unit-variance normalization of the window over the valid
   pixels (box window of `thr_win_pc * n` pixels), then subtraction of the static
   illumination field (below). The first removes smooth illumination gradients and flicker
   and makes residuals comparable across balls; the second removes what it cannot.
2. **Predict.** Start from the constant-velocity prediction `exp(w0) R_prev`, where `w0` is the
   low-passed previous increment.
3. **Align.** Iteratively reweighted Gauss-Newton on the residuals
   `r_k = I_k - M(project(R^T v_k))` with a left-multiplicative update `R <- exp([d]x) R`.
   The Jacobian of each residual is `-((R g_k) x v_k)` with `g_k` the map gradient pulled
   back through the projection. Robust weights (Huber, then Tukey) times the map confidence
   drop occluders and unseen cells. The full-resolution solve runs first; if it does not
   converge or fails the quality gates, a three-level pyramid (blurred map and window,
   strided pixels) widens the basin to cover saccades of up to ~0.35 rad/frame.
4. **Gate.** Overlap with seen cells, inlier fraction and step size decide whether the
   frame is accepted. On failure the window is aligned against the previous frame's map
   (drift-prone but robust to occlusion bursts); if that fails too the frame is dropped, and
   after `max_bad_frames` failures tracking resets. A gate on the residual cost exists
   (`cost_gate`) but is off by default: with the pre-filtered window a walking animal's
   cost is ten times a standing one's, and every cost level it was tried against rejected
   the frames where the animal starts to walk.
5. **Update.** The corrected window is splatted into the map (bilinear footprint, running
   mean with optional forgetting). `accumulate_map: n` keeps only a one-cell dilation of the
   current view, reproducing the lab fork's behaviour.
6. **Output.** The increment is expressed in camera coordinates (`R_wc w`), then lab
   coordinates (`c2a_r`), and integrated into the fictive path with FicTrac's column
   semantics. Unlike FicTrac, the camera-frame columns are in true camera coordinates rather
   than the window frame; the two coincide for a ball near the image centre.

## Static illumination

The lamps and the ball holder do not rotate with the ball, so whatever they do to a window
pixel is fixed in the camera frame while the texture it falls on turns past it. Anything
camera-fixed left in the observation is therefore not texture, and splatting it into the
ball-fixed map smears it over the surface. On the lab recordings the band just above the
holder sits about 0.9 low in normalized-intensity units, where map values span roughly
+/-3, and keeps a third of the texture contrast of the rest of the ball.

The local normalization in step 1 handles a *smooth* gradient by construction: a shading
field that varies slowly compared with its box cancels out of both the local mean and the
local standard deviation. A holder shadow is not smooth. Its edge is sharp compared with
the box, which straddles bright and dark and so reads high just above the shadow and low
inside it, and it divides by a standard deviation the same edge has inflated. That is what
produces the `+0.3, +0.2, -0.9` signature the three example recordings all show.

So spintrack estimates a per-window-pixel field `bias` and subtracts it before the solve
and before the splat, leaving both sides consistent:

    residual = (obs - bias) - gain * M(project(R^T v_k))   weighted by rho * map_conf * wt

`bias` is an exponential mean of `obs - M(project(R^T v_k))` per window pixel with time
constant `illum_tau`, read off every `illum_update_every` frames and projected to zero mean
because the field and the map are separable only up to one additive constant. (An earlier
version stepped the field toward that mean instead, which made a damped oscillator of it:
16% overshoot and 450 frames to reach two thirds of the way.) It has to go through the map: averaging the *frames* would work only if the ball
turned enough for the texture to average away, and on these recordings a window pixel sees
about three effectively independent patches of surface in two thousand frames, so a plain
temporal mean is mostly texture and subtracting it would take real signal out of the map.
Against the map the texture is subtracted explicitly, and no spatial smoothing is needed.

The update is self-limiting rather than self-reinforcing: when the ball is still, the map
absorbs the whole observation, the residual goes to zero, and the field stops moving.

The field belongs to a ball in a given place. A window re-fit that only nudges the window
resamples it by direction, since it is the camera frame it is fixed in; but a ball that has
actually moved in its holder is lit differently, shading being a function of the surface
normal, so past `illum_reset_move` of a radius the field is discarded and re-learned. On
the trial where the ball sinks 242 px and comes back, carrying the old field instead costs
more than the correction saves over that stretch.

`gain` and `wt` are the multiplicative and weighting counterparts, both off by default;
`spintrack.photometry` records what measuring them showed. `illumination: n`, or
`--no-illumination`, restores the plain behaviour.

## Geometry from the data

Three of the numbers the solver needs are measured from the recording rather than asked for.

**The ball's image circle** (`spintrack.detect`). A high per-pixel quantile over ~100 frames
erases the rotating surface texture and leaves the shading envelope; the convex hull of the
foreground removes what bites into the disc and RANSAC removes what sticks out of it; then
along each of 360 rays the outermost local maximum of the radial gradient is taken sub-pixel,
which is what keeps interior blob edges from winning. The measured rim, not a circle
resampled from the fit, is what goes into `roi_circ`: a sphere's silhouette under a pinhole
camera is an ellipse, and off-axis that is worth several percent of radius. The detection
carries a confidence and refuses rather than returning a poor circle.

**The field of view** (`autofit.fit_vfov`). The pixel circle and `vfov` together fix the
ball's angular radius, so only `vfov` is searched, with the circle held fixed. Nine log-spaced
candidates over 1-120 degrees, each tracked for 1000 frames, and the median cost taken; the
value is believed only when the curve has a minimum inside the grid, deeper than its own
wobble. The 1000 frames matter: over 300 the run is self-consistent at any assumed geometry
and the curve is flat.

**The radius, checked against itself** (`autofit.ScaleCheck`). The same frame-to-frame
increment is solved again on an inner disc and an outer annulus of the tracking window. The
two regions see the surface at different depths, and the depth is exactly what the assumed
radius sets, so their ratio is fixed when the radius is right and moves monotonically when it
is not - independently of the cost, which cannot tell an over-large ball from an under-large
one. The fixed value is the measured 1.005 rather than exactly 1, and the inversion is
normalized by it so that a correct radius reads zero error.

## Following a ball that moves

A ball that sinks in its holder leaves the window looking at the wrong part of the image, and
the solver quietly absorbs the translation into the rotation: a window left `d` pixels behind
the ball reads the ball's own movement as a rotation of about `d / r` radians. The cost cannot
say where the ball is - by the time it has risen the map has been built through the displaced
window and the two agree with each other, and on the lab ball it is flat over +-6 px of window
shift anyway - so `spintrack.refit` measures the ball's silhouette on every frame and moves
the window when the silhouette has moved.

The map is not rebuilt. It is stored in the window frame at `R = I`, which is the ball's body
frame, so a new window is only a change of coordinates: with `Q = R_wc_new^T R_wc_old`, the
orientation and velocity become `Q R` and `Q v`, the residual model `I_k - M(R^T v_k)` is
unchanged, and the ball's rotation relative to the camera is exactly preserved. The reported
absolute orientation stays referred to the first window frame so that the columns do not step.

The measurement is `relocate_ball`: the rim of a circle of the known radius, fitted from the
current frame alone in a band about where the ball is predicted to be (2.4 ms on a 1600 x
1008 frame). Two details make it usable on every frame. The fit's outlier cut is fixed at 1%
of the radius rather than estimated from the residuals, because the animal's body stands past
the rim and a residual-based scale grows to accommodate it: seeded on its own previous answer,
that fit climbs the animal's back and walks off the ball. And the look is seeded on the last
accepted look carried by the estimated velocity, never on the smoothed position, whose lag on
a fast drop puts the seed outside the look's capture range (about half the band).

An alpha-beta filter carries the position and velocity, with gains scheduled on the size of
the innovation relative to the look's own scatter: slow while the looks scatter like noise
about the prediction, fast while they run away from it. Neither pair does on its own - the
fast pair puts the look's noise into the window on a smooth, slow excursion, the slow pair
falls 50 px behind a drop. The window stays put while the filtered position is within a few
pixels of it and the speed is low, so a still ball's window never moves; once either bound is
exceeded for three frames running, the window follows the filter on every frame until the
position has been still for thirty. `detect_ball`, the look that needs no seed, is kept only
to recover a rim look that has failed for several frames in a row.

Measured on trial 004, where the ball falls 240 px and comes back: the window's distance from
an independent detector trajectory over the episode is 5-6 px (median) and 25 px (worst),
against 6.2 and 110 for the cost-triggered follower this replaces, the episode's median
photometric cost falls from 0.204 to 0.171 (an oracle-placed window gives 0.158), and the
agreement of the reported rotation with an optical-flow cross-check over the episode rises
from a correlation of 0.89 to 0.97. On the synthetic `ball_drop` scene the per-frame error
over the episode is 0.065 deg against 0.070.

Everything is measured as a displacement from a reference taken over the first looks, never
as an absolute position, so a systematic difference between the look and whatever fitted the
config's circle (3.5 px on 004) does not move a still ball's window.

Against exact truth (`lab_small_ball`, `ball_drop`) a look seeded on the true center errs
0.4-0.6 px in the median and under 3 px at worst, so the look itself is not biased toward the
animal. What the confirmation guards against is what a look seeded on its own previous answer
does: iterated that way on `lab_small_ball` it leaves the ball on a quarter of the frames, by
up to 65 px, climbing the animal, and dropping the rays through the animal makes it worse,
because a circle of fixed radius held only by its sides and bottom is free to climb. A
per-ray weight learned from each ray's agreement over the last twenty looks cut the raw
runaway to 4% of frames, but the follower's gates already contain it (a still ball's window
never moved either way) and it measured neutral to slightly worse on the benchmarks, so it
was not kept.

What remains of the delay online is structural - the confirmation that keeps a still ball's
window still, and the filter's lag on a fast drop - and `--two-pass` removes it, see below.

## Two passes over the recording

`--two-pass` tracks the recording twice: a throwaway first pass to map the ball, then the
real run starting from that map and from the illumination fields the first pass converged
on. The hand-over is in memory. It needs no correction, because the map lives in the window
frame at `R = I` and a mid-run window re-fit carries it unchanged, so both passes share a
body frame and the second one starts at `R = I` like any other run rather than searching
SO(3) the way a map loaded with `--load-map` has to.

What it buys is the cold start: a map complete from frame 0 instead of the single cap that
happens to be visible, so a ball that turns far in its first frames still has something to
match against, and the static illumination field from frame 0 instead of after its
hundred-frame warm-up. A live camera cannot be read twice, so the flag is refused rather
than ignored.

The handed-over map is a prior, not the truth, and the second pass treats it as one: its
weights are capped at `map_prior_w_max`, just above `w_min`, so that its cells count as seen
and can be matched where nothing fresher exists, while the first frame that sees a cell
replaces its content almost entirely. The map is stale in two ways. Lighting, shading and
the animal's shadow change over a recording, which is why the offline refinement below
matches against a temporally local map; and the first pass's drift has displaced each region
of the map by however far the first pass had drifted when it last saw it, about a degree on
trial 003 (a block cross-correlation of the map the first pass handed over against the map
the second pass left behind finds the same texture shifted by 0.75 degrees in the median,
1.1 at the 90th percentile). With its weights kept, the second pass on 003 opened at a cost
100 times the one-pass run's, spent 150 frames overwriting the visible region before the
cost came down, and over its first 300 frames matched the optical-flow cross-check 47% worse
than one pass in y; a *frozen* copy of the same map lost the ball within 100 frames. Any cap
that lets stale and fresh content mix for a few frames makes the estimate wander while the
mixture changes: 1.8 degrees of wobble over the first ten frames' increments at a cap of 3,
1.3 at 1, 0.5 at 0.5. At 0.15 the second pass's opening increments are the one-pass run's
frame for frame, the rest of the run is unchanged, and the ball_drop episode error with exact
truth is unchanged to the third decimal; on the synthetic scenes this gives up the 10-20%
the stale map used to buy over the opening frames, which on a recording it never bought. The
first frame against a handed-over or loaded map fixes where the ball is: it gets five times
the usual iteration budget and is reported with a zero increment, because the rotation the
second pass snaps through onto the stale map (1.9 degrees on 003) is not a rotation of the
ball.

Drift, measured by loop closure on trial 003 (frame j aligned directly against frame i
alone, starting from the run's own relative rotation; the residual is the drift between
them): 0.06 degrees for frames a few apart, which is the noise floor of one alignment, then
0.10, 0.15 and 0.18 degrees (median) for gaps of 8-50, 50-200 and 200-600 frames in the
one-pass run, with a systematic component about the window's x axis, the axis a walking
animal turns the ball about. The second pass closes the same loops at 0.10, 0.12 and 0.13
degrees, so it inherits a little less than the first pass's drift rather than more, but the
random walk over a whole recording is still there and the refinement below does not remove
it either; that would take loop-closure constraints between distant frames, which these
measurements show to be available.

The second pass also knows where the ball went. The follower above measures the ball's
silhouette on every frame of the first pass, and the second pass places its window on a
trajectory planned from all of those looks at once (`refit.plan_window_trajectory`):
interpolated over the frames without a look, cleaned with a five-frame median, smoothed with
a zero-phase Gaussian, held at the ball's resting level by the same rule that keeps a still
ball's window still online, and following each move from the frame the ball leaves that
level rather than from the frame an online confirmation ends. The second pass measures
nothing, which is also why it tracks faster than the first. On `ball_drop`, with exact
truth, the window's distance from the ball over the episode goes from 7.8 px (p95) for the
online follower to 2.2, and the episode's per-frame error from 0.056 deg (median) and 0.205
(p95), for a second pass that follows the ball for itself, to 0.044 and 0.113. On trial
004's fall, judged by the optical-flow cross-check over the episode, the residual along the
fall goes from 4.31 px rms for the online run to 4.17 for a second pass that follows for
itself and 3.82 for the planned window. The smoothing is six frames wide, and wider was
better on both scenes up to the widest tried: a window offset that stays constant costs no
rotation, only its change does, so the bias of smoothing an accelerating ball matters less
than the look's noise.

## Offline refinement

Online tracking builds the map incrementally, so the first frames' errors are baked into
it, and the frame-to-frame mode (`accumulate_map: n`) integrates a random walk. With
`--refine N`, spintrack keeps every normalized window, and after the run re-aligns every
frame against a map rebuilt from the other frames at their estimated orientations; each
sweep repeats both steps. Frames that were dropped online are seeded from their neighbors
and solved too. The refined orientations are integrated into a second `.dat` file
(`<out>-refined.dat`). Memory: about `2 * n^2` bytes per frame (`n` the window size), i.e.
~170 MB per minute at 100 fps and `q_factor 12`.

Three details decide whether the refined output is better than the online one, and all
three were measured on the lab trials with the optical-flow cross-check rather than by
agreement with the online run:

- The map is temporally local, an exponential window of 50 frames either side of the frame
  being solved. Against a map of the whole recording the per-frame increments on trial 003
  came out *worse* than online (0.59 px rms against 0.43): lighting, shading and the
  animal's shadow change over a recording, and a mean over all of it blurs the texture any
  one frame sees. The local map beats online (0.41 px) and, with exact truth on `ball_drop`,
  takes the episode error from 0.053 to 0.049 deg.
- Each window stays paired with the orientation in the window frame it was tracked in. A
  window's pixels map to the same surface directions wherever the window sits, so bringing
  the orientations into the final frame first splats every pre-move frame rotated by the
  move; on `ball_drop` that cost 0.088 deg per frame against 0.070.
- A frame whose re-solved orientation jumps away from both neighbors while the neighbors
  agree with each other keeps its previous orientation. Without that, wrong minima that fit
  well enough put out-and-back spikes of 9-12 deg into the refined 003 that the online run
  never had, and its cross-check correlation fell from 0.998 to 0.59.

## The shape of the map's cells

An equal-area grid gives every cell the same solid angle but not the same shape. Cell
extents are `(2 pi / W) cos(lat)` east-west and `(2 / H) / cos(lat)` north-south, so at the
lab default of 180x360 an equatorial cell is 0.64 x 1.0 degrees and the polar row is
8.6 x 0.1 - a sliver, in a region the camera resolves to about a degree. The projection
Jacobian carries the same asymmetry: `du/dp` grows as `1 / cos^2(lat)`, so map noise near a
pole is amplified on its way into the normal equations.

`benchmarks/spintrack_bench/map_grid_sweep.py` measures what that costs, and the answer is
that it costs something. Accuracy improves monotonically with cell count until cells outrun
window pixels, so the grid is the binding constraint; and moving the poles over the ball,
which changes nothing else, moves the median per-frame error by 6-19%, with the default
placement at the worst end of that range. The poles sit at the window's +/-y axis, and the
pitch rotation of a walking fly carries that material point through the near point of the
ball, where the camera resolves best and the map resolves worst.

The cubemap replaces the grid with six equi-angular faces, each with `s' = tan(pi s / 4)`
so the cells are near-uniform rather than gnomonic. At the same cell
count it gives up a little resolution at the equator (0.87 degrees square everywhere,
against 0.64 x 1.0) to remove the 8.6-degree cells at the poles, and its Jacobian is bounded
because a direction always lies on the face it is closest to. Faces meet at seams instead of
wrapping; a bilinear tap that runs off a face is resolved by re-projecting the direction of
the cell it asked for, which cannot disagree with the forward projection and costs a `tan`
and an `atan2` on the few per cent of taps near a seam. Maps convert between the two grids
on load, so a map saved on one is usable on the other, as is a FicTrac template.

Over the 21 synthetic scenes at the same cell count it lowers the median per-frame error on
18 of them, by 7% in the median and up to 24%, and drops no frames anywhere. On the lab
recordings, where there is no ground truth, neither projection loses a frame and the two
agree to a median 0.014-0.030 degrees per frame - except on one trial, which diverges by 20
degrees of heading over 3000 frames. There the cubemap is the better run by every signal
available: lower median cost (0.058 against 0.066), half the p99 cost, and online estimates
twice as close to their own offline refinement (median 0.028 against 0.058 degrees). It is
the default on that evidence. It costs a quarter to a third more tracking time, and
`--map-projection equal_area` goes back to FicTrac's grid, which is what a comparison
against FicTrac wants.

## Why it is more precise

With `q_factor 12` one window pixel spans about one degree of ball rotation and a walking fly
turns the ball about one pixel per frame. FicTrac's nearest-tile binary matching resolves
motion at roughly that scale; sub-pixel photometric alignment with bilinear sampling resolves
it an order of magnitude finer, and analytic derivatives reach the optimum in a few
iterations instead of tens of cost evaluations. See `docs/benchmark.md` for measurements.
