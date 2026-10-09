# How spintrack works

For maintainers: the tracker as it stands, why each piece is the way it is, and what pins it.

## Pipeline

`spintrack run` (`cli.py`) loads the config, detects the ball if the config has none (`autofit.prepare_config`), and hands a frame source to `pipeline.run`, which decodes in a background thread while the main thread tracks. For each frame, `Tracker.process_frame` (`tracker.py`):

1. lets the ball follower move the tracking window if the ball has moved in its holder (`refit.py`);
2. remaps the source frame into the window (`sphere.py`);
3. runs one engine step (`engine.py`): normalize, correct for the illumination fields, solve for the rotation against the surface map, update the map and the field;
4. converts the increment to camera and lab coordinates, integrates the path (`path.py`) and returns the 25 columns.

Recorders in `io/` write files and streams; `quality.py` summarizes the run. The Rust core (`rust/src/`) holds what touches every pixel: the solver, the map and the global search.

## Window geometry

The camera is a pinhole defined by its vertical field of view, or an equidistant fisheye. The ball is a unit direction to its center plus an angular radius, from a cone fitted through the config's rim points; angles rather than pixels keep the geometry exact for an off-axis ball, whose silhouette is an ellipse.

The tracking window is the image of a virtual fisheye camera aimed at the ball's center, `n = window_px` pixels square, sampled with `cv2.remap`. Each window pixel that sees the ball gets a fixed unit vector `v_k` from the ball's center to the surface point it observes, so the rotation is the only unknown. The config's `mask.ignore` pixels are masked out.

A window pixel spans several source pixels, and bilinear sampling at that decimation aliases the texture, so the source is pre-filtered: one `cv2.pyrDown` and a Gaussian on the half-size frame, a total blur of half the decimation (`sphere.PREFILTER_SIGMA`). Filtering at half size costs a quarter of a full-size Gaussian; below a decimation of 2 the filter is skipped.

## Per-frame solve

**Normalize.** The window is normalized to zero mean and unit variance in a local box of `tracking.norm_window * n` pixels, and corrected by the illumination fields. This removes smooth gradients and flicker and makes residuals comparable between rigs.

**Predict.** The solve starts from `exp([w0]x) R_prev`, with `w0` the low-passed previous increment, because a walking animal's rotation changes little between frames.

**Align.** Iteratively reweighted Gauss-Newton minimizes `r_k = I_k - M(project(R^T v_k))` between the window `I` and the map `M`, with the update `R <- exp([d]x) R` and the analytic Jacobian `-((R g_k) x v_k)`, `g_k` being the map gradient pulled back through the projection (`rust/src/solve.rs`). Analytic derivatives and bilinear sampling resolve a fraction of a window pixel in a few iterations, where FicTrac's binary tile search resolves about one pixel. Each residual is weighted by a Huber weight, a Tukey biweight and the map cell's confidence, so occluders and unseen cells drop out without a hard mask. The solve runs on a subsample of about 4000 window pixels (`TrackParams.max_pixels`) to bound the time per frame; the map is updated from every pixel.

**Fall back.** If the full-resolution solve does not settle or is rejected, a three-level pyramid widens the basin for saccades. If that fails, the window is aligned against the previous frame's map, which drifts but survives a burst of occlusion. Then, with `tracking.global_search`, the global search relocalizes; otherwise the frame is dropped, and after `tracking.max_bad_frames` drops in a row tracking restarts from a fresh map.

**Accept.** A solve is accepted when the increment is within `tracking.max_step_rad`, at least a quarter of the window falls on seen map cells, and at least half of those are inliers. There is no gate on the cost, because a walking animal's cost is much higher than a standing one's and a cost gate rejects the frames where it starts to walk.

**Relocalize.** The global search scores quasi-uniform orientations over SO(3) at the coarsest level, refines the best few through the pyramid, and keeps the cheapest that overlaps enough. The frame reports zero motion, as in FicTrac, since the jump corrects the estimate rather than measuring the ball. A map loaded with `tracking.initial_map` is localized this way on the first frame.

**Report.** The increment goes to camera coordinates through the window's orientation, to lab coordinates through the camera-to-animal rotation, and into the path through a port of FicTrac's `Trackball::updatePath`. The absolute orientation stays referred to the first window, so a window move does not make it step.

## The map

The map is an equi-angular cubemap (`rust/src/map.rs`): six square faces in one array, with face coordinates `s' = tan(pi s / 4)` so that cells stay near-square, at FicTrac's density of `2 (1.5 n)^2` cells. FicTrac's equal-area grid has sliver cells near its poles, where the projection Jacobian grows as `1 / cos^2(lat)`; the cube's is bounded. A bilinear tap that runs off a face re-projects the direction of the cell it asked for.

Each cell holds a running weighted mean of normalized intensity and a weight. A cell counts as seen above `W_MIN`, is fully trusted at `W_SAT`, and is capped at `W_MAX` so that the map keeps adapting. Accepted windows are splatted in with a bilinear footprint, each pixel weighted by the lighting gain squared (next section) and by the squared cosine of its viewing angle (`WindowGeometry.facing`). Because the cap makes each cell follow its most recent views, unweighted dim and foreshortened views would otherwise wash out the texture a patch showed when first seen head-on and well lit. `tracking.forget_outside_view = true` keeps only the current view plus a one-cell margin, as the lab's FicTrac fork does.

The map lives in the ball's body frame (the window frame at `R = I`), so a window move is a change of coordinates, not a rebuild. A loaded map is a prior: its weights are capped at `MAP_PRIOR_W_MAX`, just above `W_MIN`, so that the first fresh observation of a cell replaces it, because a map from another time differs in lighting and carries that run's drift. `tracking.freeze_map` keeps a loaded map fixed.

## Static illumination field

The lamps and the holder do not turn with the ball, so their shading is fixed in the camera frame while the texture turns past it; left in, it would smear over the map. Local normalization removes smooth shading but not the sharp edge of a holder's shadow, nor the loss of texture contrast inside the shadow. `photometry.Photometry` therefore estimates two fields per window pixel. The bias is the exponential mean of `obs - gain * M(project(R^T v_k))` over 500 frames and is subtracted. The gain is how much texture contrast the pixel delivers: the RMS of the bias-corrected observation, smoothed over 6 pixels and divided by its 90th percentile over the window, which needs no map because the ball's texture averages out as it turns. The observation is divided by the gain before both the solve and the splat, and the splat weights each pixel by the gain squared, since the division amplifies a dim pixel's noise. It samples every fourth accepted frame and refreshes every 25 frames after a 100-frame warm-up, and is kept at zero mean because field and map are separable only up to a constant.

Measuring against the map removes the texture explicitly, which a temporal mean of the frames would not, since a pixel sees few independent patches of surface in a recording. On a still ball the map absorbs the observation and the field stops moving, so the estimate cannot run away. A window move resamples the field; a move beyond 5% of the radius discards it, because a moved ball is lit differently. `tracking.initial_illumination` loads a field saved on the same rig as a prior worth one time constant.

## Ball follower

A window left `d` pixels behind the ball reads the ball's movement as a rotation of about `d / r`, and the cost cannot tell, so `refit.CenterWatch` looks at the silhouette on every frame. The rim look (`detect.RimLook`) fits a circle of known radius in a band around the predicted position. Its outlier cut is fixed at 1% of the radius, because a cut scaled from the residuals grows to take in the animal past the rim and lets the fit climb onto it. The radius is measured from the first 30 frames, so a config radius a few percent off does not bias the look.

While the ball rests, the window stays put, because the animal at the rim pulls the look by a few pixels for tens of frames. A move is followed once three looks in a row lie beyond 10 px or 2% of the radius. The window then sits on the longest of a set of line fits over the last 2 to 64 looks that agrees with every shorter one (Lepski's rule), which averages steady motion and shrinks to the last few looks through a jerk. Following ends after 60 still frames. Positions are displacements from a reference taken over the first looks, so a fixed offset between the look and the config's circle never moves the window. Failed looks fall back to a seed-free detection. A window that would move more than one radius means the look has run away, so the follower stops and puts it back.

When the run ends, all the looks are smoothed at once, as for the second pass below (`refit.smooth_looks`), and `tracks.parquet` saves the result as the ball's position, with the window's distance from it in ball radii (`Tracker.ball_columns`).

A window move is a change of coordinates `Q` between window frames: the orientation becomes `Q R`, the velocity `Q v`, and the map is kept. The window is moved before the frame it was placed for is tracked.

## Two passes

`--two-pass` tracks the recording once to map it and again to report it, the second tracker starting from the first one's map and illumination field (`Tracker.prime_from`). Both share the ball's body frame, so the second pass starts at `R = I` without a global search. Its map is capped like any loaded map, since it carries the first pass's drift; the first frame gets five times the usual iterations and reports zero motion, since snapping onto that map is not a rotation of the ball.

The second pass replays a window trajectory planned from all of the first pass's looks (`refit.plan_window_trajectory`): gaps interpolated, a five-frame median and a zero-phase Gaussian whose width (up to 12 frames) shrinks through jerks by the same Lepski rule (`refit.smooth_looks`), then held while the ball rests, and each move traced back to where it began. This removes the online confirmation delay, and the second pass, which measures nothing, runs faster than the first.

## What is verified, and how

- **Scale and geometry**, `tests/test_scale.py`: a ball rendered with none of spintrack's camera, sphere or rotation code comes back at scale 1.0000 within 0.05%; a relative radius error `eps` scales the in-plane rotation by about `1 - 1.9 eps` and leaves the rotation about the optical axis unchanged; a hand-worked quarter turn pins the direction in which the camera rotation is applied.
- **Path integration**, `tests/test_path.py`: on 200 rows FicTrac wrote for a lab trial (`tests/data/fictrac-003-head.dat`), FicTrac's columns 6-8 reproduce its columns 15-21 to 1e-12, and `FrameResult.forward`, `side` and `turn` sum to its columns 17, 20 and 21.
- **Camera position**, `tests/test_calibrate.py`: `camera.position_deg` angles give the expected frame at three camera poses, the calibration square recovers a synthetic pose, and a square projected from six camera poses and clicked in FicTrac's order gives the rotation of their angles. `tests/test_autofit.py`: an animal on the ball's top, seen from a known position, places the camera there.
- **Precision**, the [benchmark](benchmark.md): per-frame error, drift and scale against exact synthetic truth. Its scenes share spintrack's camera code, which is why the scale test exists.
