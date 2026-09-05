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

## Why it is more precise

With `q_factor 12` one window pixel spans about one degree of ball rotation and a walking fly
turns the ball about one pixel per frame. FicTrac's nearest-tile binary matching resolves
motion at roughly that scale; sub-pixel photometric alignment with bilinear sampling resolves
it an order of magnitude finer, and analytic derivatives reach the optimum in a few
iterations instead of tens of cost evaluations. See `docs/benchmark.md` for measurements.
