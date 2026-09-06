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
  image into an `n x n` window (`n = 10 * q_factor`) with `cv2.remap`. Each window pixel that
  sees the ball gets a unit vector `v_k` from the ball centre to the surface point it observes
  (window frame: z toward the ball). These vectors are fixed; rotating the ball rotates them.
- **Surface map.** A float32 Lambert equal-area grid over the ball surface (`u` from the
  longitude about the window y axis, `v` from `sin(latitude)`), storing a running weighted
  mean of normalized intensity plus a weight per cell (0 = never seen).

## Per frame

1. **Normalize.** Local zero-mean, unit-variance normalization of the window over the valid
   pixels (box window of `thr_win_pc * n` pixels). This removes illumination gradients and
   flicker and makes residuals comparable across balls.
2. **Predict.** Start from the constant-velocity prediction `exp(w0) R_prev`, where `w0` is the
   low-passed previous increment.
3. **Align.** Iteratively reweighted Gauss-Newton on the residuals
   `r_k = I_k - M(project(R^T v_k))` with a left-multiplicative update `R <- exp([d]x) R`.
   The Jacobian of each residual is `-((R g_k) x v_k)` with `g_k` the map gradient pulled
   back through the projection. Robust weights (Huber, then Tukey) times the map confidence
   drop occluders and unseen cells. The full-resolution solve runs first; if it does not
   converge or fails the quality gates, a three-level pyramid (blurred map and window,
   strided pixels) widens the basin to cover saccades of up to ~0.35 rad/frame.
4. **Gate.** Overlap with seen cells, inlier fraction, residual cost and step size decide
   whether the frame is accepted. On failure the window is aligned against the previous
   frame's map (drift-prone but robust to occlusion bursts); if that fails too the frame is
   dropped, and after `max_bad_frames` failures tracking resets.
5. **Update.** The normalized window is splatted into the map (bilinear footprint, running
   mean with optional forgetting). `accumulate_map: n` keeps only a one-cell dilation of the
   current view, reproducing the lab fork's behaviour.
6. **Output.** The increment is expressed in camera coordinates (`R_wc w`), then lab
   coordinates (`c2a_r`), and integrated into the fictive path with FicTrac's column
   semantics. Unlike FicTrac, the camera-frame columns are in true camera coordinates rather
   than the window frame; the two coincide for a ball near the image centre.

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
radius sets, so their ratio is 1 when the radius is right and moves monotonically when it is
not - independently of the cost, which cannot tell an over-large ball from an under-large one.

## Following a ball that moves

A ball that sinks in its holder leaves the window looking at the wrong part of the image, and
the solver quietly absorbs the translation into the rotation. `spintrack.refit` watches the
cost against a long baseline; when it is sustainedly high it re-detects the ball and, if it
really has moved, follows it.

The map is not rebuilt. It is stored in the window frame at `R = I`, which is the ball's body
frame, so a new window is only a change of coordinates: with `Q = R_wc_new^T R_wc_old`, the
orientation and velocity become `Q R` and `Q v`, the residual model `I_k - M(R^T v_k)` is
unchanged, and the ball's rotation relative to the camera is exactly preserved. The reported
absolute orientation stays referred to the first window frame so that the columns do not step.

The window has to move on *every* frame while the ball is moving, not just when a detection
lands: a window left one frame behind turns the ball's own movement into a rotation of about
`d / r` radians. Detection is too slow for that, so it runs every tenth frame and an
alpha-beta filter carries the centre in between.

## Offline refinement

Online tracking builds the map incrementally, so the first frames' errors are baked into
it, and the frame-to-frame mode (`accumulate_map: n`) integrates a random walk. With
`--refine N`, spintrack keeps every normalized window, and after the run rebuilds the map
from all frames at their estimated orientations and re-aligns every frame against that
complete map; each sweep repeats both steps. Frames that were dropped online are seeded from
their neighbours and solved too. The refined orientations are integrated into a second
`.dat` file (`<out>-refined.dat`). Memory: about `2 * n^2` bytes per frame (`n` the window
size), i.e. ~170 MB per minute at 100 fps and `q_factor 12`.

## Why it is more precise

With `q_factor 12` one window pixel spans about one degree of ball rotation and a walking fly
turns the ball about one pixel per frame. FicTrac's nearest-tile binary matching resolves
motion at roughly that scale; sub-pixel photometric alignment with bilinear sampling resolves
it an order of magnitude finer, and analytic derivatives reach the optimum in a few
iterations instead of tens of cost evaluations. See `docs/benchmark.md` for measurements.
