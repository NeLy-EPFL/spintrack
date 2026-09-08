# Verification

[benchmark.md](benchmark.md) says how *precise* the tracker is: hundredths of a degree per frame, against exact synthetic truth. This page is about whether it is *right* - whether one radian of ball rotation comes back as one radian, in the direction the animal actually moved - because a precision number cannot tell you that. A tracker reporting every rotation 2% small scores the same per-frame error as one that is right on average and equally noisy.

Two things are open, and they are stated here rather than buried: spintrack reports 1.5 to 4% less **turning** than FicTrac on five real recordings of one rig and 9% less on a sixth - a gap that widens the more slowly the animal turns, and that no synthetic scene reproduces even when built with that rig's geometry and slowed to its animals' own pace - and about 2% less **sideslip** than truth on the synthetic scenes whose ball is small in the frame. Both are below the per-frame error in absolute terms and neither changes what the tool is for, but both are systematic. The sections below say what has been ruled out for each.

## The circularity, and what closes it

The benchmark renders its ball with `spintrack.camera` and `spintrack.sphere`, and turns its ground-truth motion into camera coordinates with the same `spintrack.calibrate.sliders.camera_to_lab_from_angles` the tracker inverts. That is the right way to test a solver and the wrong way to test a convention: an error in the camera model, in the depth an assumed radius implies, or in the direction the camera-to-animal rotation is applied would cancel exactly between the renderer and the tracker, and every table in benchmark.md would still read hundredths of a degree.

Four things close it, from the most to the least that they can see.

**`tests/test_scale.py` renders a ball with none of spintrack's geometry code.** The pinhole projection is written out from its definition, the ball is an analytic sphere intersected with the rays, and the rotations come from a Rodrigues formula in the test file. It then asks whether the reported rotation is the rotation it rendered, and what a wrong assumed radius does to that.

**The camera-to-animal frame is pinned from first principles, in both directions.** `tests/test_calibrate.py` fixes what `camera_to_lab_from_angles` must produce at three camera poses (a camera in front of the animal looks backwards along -x, its image-down axis is animal-down, its image-right is animal-left), and `tests/test_scale.py` fixes the direction it is *applied* end to end, with a quarter turn about the optical axis written out by hand and the expected forward, sideslip and turning worked out from it. A transposed transform passes every other check in the repository.

**FicTrac on the real recordings is an independent implementation.** It reads the same `config.txt`, so a calibration error is common to both, but the code is not shared: where the two agree on scale, the disagreement is not in either implementation. Over six 60 s trials of one rig they agree on forward amplitude to about a percent, and **disagree on turning by 1.5 to 4% on the five trials the animal walked through, and by 9% on the sixth**, with spintrack lower on every one. Nothing here says which tracker is right: that needs truth on that rig, which these recordings do not have. It is the largest open discrepancy in this page, and it is on the component most of these experiments report. The numbers are in the agreement table in benchmark.md.

What it is not. Each of these is ruled out by measurement rather than by argument:

- **The geometry.** No benchmark family had this rig's: every one of them is either an 11-degree ball or a 0.31-degree one. Two scenes built to match it - 1.2 degree half-angle, a 506-pixel ball, 1600x1008 HEVC at `q_factor` 12, legs and body - put spintrack's turning at **0.9999 and 1.0000** of exact truth (+- 0.0002) and FicTrac's at **0.9990 and 1.0041** (+- 0.0010 and 0.0020). Their ratio is 1.001 and 0.996, against 0.911 to 0.985 on the real trials. `notes/session_2026-09-08b/make_lab_big.py` builds them; `notes/session_2026-09-08d/scale_gain.py` scores them.
- **The illumination correction** (turning the static-illumination estimate off moves the real trials 0.1 to 0.2 percentage points), and **the lab fork's own changes**: upstream FicTrac and the fork agree to five decimals on those two scenes.
- **The exposure.** A long exposure does attenuate turning - on `motion_blur`, whose exposure is 0.8 of the frame period, both trackers read 0.954 - but it attenuates them equally and leaves their ratio at 1.0003. Motion blur cannot separate the two.
- **The holder shadow**, which moves the ratio the observed way and nowhere near far enough: at the strength matched to these recordings it takes the ratio from 1.001 to 0.996, half a point, against the 1.5 points of the mildest real trial.
- **The estimator.** `scale_ratio` reads a known 1.00 as 0.999 and 1.003 on those same scenes, with FicTrac's per-frame error at the 0.16 to 0.25 degrees it really has there against spintrack's 0.03. Planted series whose true ratio is 1.00 carry that to the real trials' operating point: independent per-tracker noise heavy enough to drive the two series' correlation down to 0.67, below the worst real trial's 0.78, still reads 0.995, and an error the two trackers *share* - up to three quarters of the total, correlated over as much as eight frames, which is what reading the same video, the same legs and the same compression artifacts produces - cancels between the statistic's numerator and its denominator and reads 1.000. What it does get wrong at this geometry is sideslip, 4 to 6 points low - measured, though, where sideslip is a tenth of the amplitude of turning, and in the real trials sideslip is the *largest* component. Read the side column as carrying an error bar this comparison has not pinned down, rather than as agreement to a percent.

And what it depends on. Split each trial into 6-second windows and bin them by how fast the animal was actually turning (`notes/session_2026-09-08f/turn_rate.py`, which bins on the mean of *both* trackers' amplitude so that neither selects its own bins):

| turning, deg/frame | windows | spintrack / FicTrac |
|---|---|---|
| 0.03 to 0.08 | 16 | 0.933 +- 0.011 |
| 0.08 to 0.12 | 15 | 0.961 +- 0.005 |
| 0.12 to 0.25 | 15 | 0.977 +- 0.003 |
| 0.25 to 0.55 | 16 | 0.971 +- 0.013 |

**The deficit is a slow-turning effect**, three times larger in the slowest windows than in the fastest, and the estimator is not the cause of the trend: at 600-frame windows and correlations down to 0.58 - below the worst bin's 0.76 - planted series with a true ratio of 1.00 read it within 0.7%, and the lagged autocovariance those bins are instrumented on carries 0.61 to 0.83 of the variance against the 0.05 the estimator refuses below.

**The synthetic scenes move nothing like these recordings**, which looked like the explanation and is not. `lab_big_ball` turns at 0.87 and walks at 0.80 degrees per frame; five of the six trials turn at 0.07 to 0.16 and walk at 0.04 to 0.09, ten to twenty times slower, and the sixth - the one that does walk like the scene - has the second-best turn ratio of the six. So the same geometry, texture, lighting, occluders and sensor were re-rendered with `fly_walk` scaled to 0.3 and 0.1 (`notes/session_2026-09-08f/make_lab_slow.py`), which brackets the real trials, and scored against exact truth:

| scene | turning, deg/frame | spintrack | FicTrac | spintrack / FicTrac |
|---|---|---|---|---|
| `lab_big_ball` | 0.871 | 0.9999 +- 0.0002 | 0.9990 +- 0.0010 | 1.001 |
| `lab_rate30` | 0.266 | 0.9997 +- 0.0003 | 0.9965 +- 0.0026 | 1.003 |
| `lab_rate10` | 0.093 | 0.9978 +- 0.0012 | 0.9884 +- 0.0078 | 1.010 |

**It does not reproduce.** At 0.093 degrees per frame - the middle of the real trials' range, and above the slowest bin of the table above - spintrack is still within 0.2% of truth and FicTrac within 1.2%, and the ratio moves the *wrong way*: FicTrac is the one that softens as the ball slows, so spintrack reads **higher** than it, where on the real recordings spintrack reads 1.5 to 9% lower. Rate alone is not the mechanism, and the real trials' rate dependence has some other cause. (FicTrac's sideslip does collapse at this rate, 0.60 of truth against spintrack's 0.97, but sideslip is barely excited in these scenes and its error bars are ten times the others'.)

The mismatch is worth recording on its own account, though, because it is not confined to this question: **the benchmark has only one activity level**. Fourteen of its seventeen scenes with an identified turn gain turn at 0.871 degrees per frame, the remaining three at 0.87 to 1.39: the families vary the optics, the lighting, the occluders, the noise and the codec, and all of them drive the ball with the same seeded `fly_walk`. So the tables cannot show a rate-dependent effect of any kind, in either tracker, and the gains in them should be read as measured at one operating point 5 to 12 times more active than these recordings - not as a range that covers them.

**The `spintrack` and `spintrack-full` columns are the same solver on different pixel counts**, which catches anything that depends on how much of the window is used.

What none of them can reach: whether the circle in the config is the ball's real silhouette, and whether the real lens is the pinhole or equidistant model the config claims. Those are measurements on the rig - a caliper, a camera distance, a calibration target - not properties of the software. Everything below is about the software's contribution, which is the part that can be closed from here.

## Absolute scale

The scale is the least-squares factor between the reported rotation and the true one, pooled over the three camera-frame components. On the from-scratch ball it is **1.0000 within 0.05%**, over window sizes from 40 to 160 pixels and three ball positions in the frame. On the benchmark's 20 scenes it is **0.9986 to 1.0001**, and on the four that turn at a constant rate - where nothing temporal can enter - **0.9999 to 1.0000**. FicTrac's own scale over the same scenes is 1.0003 median (0.9985 to 1.0031, `speckle` excluded, where it loses the ball), so the two implementations agree on the geometry to about a part in a thousand.

Per component, in the animal frame, over the 20 scenes: forward walking **0.9990 to 1.0003**, turning **0.9982 to 1.0002**, sideslip **0.977 to 1.004**, with `motion_blur` (0.9969, 0.9565 and 0.9420) held out as the one scene where the exposure rather than the geometry sets the answer. The full tables, with bootstrap error bars, are in benchmark.md. Sideslip is the loose one for a reason worth understanding rather than worrying about - see below.

## What moves the scale

### A wrong assumed ball radius

The assumed radius sets the depth of every surface point, and an image rotation carries no depth, so the two separate cleanly. A relative radius error `eps` scales the rotation about an axis *in the image plane* by about `1 - 1.9 eps`, and the rotation about the optical axis not at all:

| relative radius error | in-plane rotation | rotation about the optical axis |
|---|---|---|
| -5% | 1.086 to 1.100 | 1.000 to 1.002 |
| -2% | 1.035 to 1.041 | 1.000 to 1.001 |
| +2% | 0.957 to 0.969 | 0.999 to 1.001 |
| +5% | 0.900 to 0.920 | 0.998 to 1.001 |

Each cell is the reported rotation at that radius error over the reported rotation at the right radius, so the absolute scale (above) divides out and what is left is the response. Measured by `notes/session_2026-09-08d/radius_slope.py` at half-angles of 3.3, 10.3 and 25 degrees and `q_factor` 6 and 12; the ranges are the spread over those six geometries, and the fitted in-plane slope is 1.68 to 2.04.

Two consequences. A camera behind the animal sees forward walking as a purely in-plane rotation, so **forward speed carries about twice the radius error**; and because the two directions respond differently, the error is not a common factor that cancels between one reported component and another. This is what `autofit.ScaleCheck` exists to catch, and it is why the two warnings that used to say "the rotation scale follows the radius almost 1:1" were wrong by a factor of two.

### An exposure that is a large fraction of the frame period

On `motion_blur` (exposure 0.8 of the frame period) the scale reads 0.969 and turning 0.957, while `scale_path` - the same factor as it reaches the integrated path - reads 1.0000. That is not a scale error: with a long exposure the frame *is* the average of the motion over the exposure, so the reported per-frame rotation is the exposure-averaged one rather than the instantaneous one. Per-frame speeds are low-passed, and the fictive path is exact. Shorten the exposure if instantaneous speeds matter; nothing needs correcting if the path does.

### A ball that is too small in the source image

Nothing in a run says the ball is too small. The window is resampled to whatever `q_factor` asks for, and the photometric cost stays low because the model fits the pixels it has - it is the reported *rotation* that shrinks. Rotation about the optical axis goes first: it is read from an image rotation, whose displacement at the rim is the ball's pixel radius times the angle, so it is the first displacement to fall below a pixel.

Measured on a rendered ball at 3x supersampling (`notes/session_2026-09-08d/ball_pixels.py`), sweeping both the pixel radius at a fixed rate and the rate at a fixed radius:

| ball radius | rim displacement | in-plane | about the optical axis |
|---|---|---|---|
| 4 px | 0.05 px | 0.704 | 0.600 |
| 7 px | 0.10 px | 1.012 | 0.935 |
| 14 px | 0.06 px | 0.978 | 0.983 |
| 14 px | 0.11 px | 0.995 | 0.988 |
| 14 px | 0.20 px | 0.998 | 0.995 |
| 14 px | 0.39 px | 0.997 | 0.999 |
| 28 px and up | 0.39 px and up | 0.994 to 0.999 | 0.990 to 1.000 |

It takes both conditions. At a 14-pixel radius and 0.2 pixels of rim displacement or more, every component is within half a percent; at the same radius but 0.06 pixels of rim displacement - the same ball turning three times slower - both are 2% low; and at 7 pixels of radius and below the tracker keeps running and reports 6 to 40% low whatever the rate. `spintrack run` now warns below 15 pixels (`tracker.MIN_BALL_RADIUS_PX`); nothing warns about the rim displacement, because it is a property of the motion rather than the setup, and 2% of a rotation that slow is a very small number.

The lab's own geometry is far from either limit: a 79-pixel ball radius on the synthetic lab scenes, several hundred on the real recordings, with turning moving the rim by one to two pixels.

This is also the *partial* explanation of the sideslip column. A camera 30 degrees behind the animal sees sideslip as 87% a rotation about the optical axis, and sideslip is a tenth of the amplitude of turning, so it is the component whose rim displacement is smallest:

| scene | ball radius | sideslip rim displacement | sideslip gain |
|---|---|---|---|
| clean_fly | 175 px | 0.33 px | 0.9988 +- 0.0035 |
| occluded (same geometry and motion) | 175 px | 0.33 px | 0.9840 +- 0.0068 |
| offaxis | 136 px | 0.26 px | 0.9890 +- 0.0081 |
| lab_small_ball | 79 px | 0.15 px | 0.9776 +- 0.0070 |
| holder_shadow_lab | 79 px | 0.15 px | 0.9818 +- 0.0080 |

The trend runs the right way, but the table above puts the attenuation at a rim displacement of 0.15 px at about a percentage point, and 87% of it squared is 0.8 - so the sub-pixel limit accounts for under half of the 2 points the small-ball scenes show. **The rest is not explained.** What it is not: the animal. Rendering `occluded` with and without its body, its legs and its dust specks moves the sideslip gain by at most 0.4 percentage points averaged over three seeds, inside the seed-to-seed scatter of about a point (`notes/session_2026-09-08d/occl_attrib.py`), and widening `roi_ignr` up to 2.5x recovers half a point at best (`occl_gain_lab.py`). Nor is it the solver's stopping tolerance, the window prefilter, the local normalization window, the illumination correction, the pyramid depth or the image noise: every one of those was swept and the reported rotation did not move in the fourth decimal.

In absolute terms the open part is small - sideslip runs at 0.13 degrees per frame in these scenes, so 1 to 2% of it is 0.002 degrees per frame against a per-frame error of 0.05 - and it is at the level where a single scene's gain for a component this weakly excited cannot be measured. It is listed here rather than resolved.

## What the per-component numbers do not say

Sideslip in these scenes has a tenth of the amplitude of turning: 0.13 degrees per frame rms against 1.43. Its gain is therefore measured against a residual of a comparable size, and its bootstrap error is ten times wider than the others' - a few tenths of a percentage point at best, over a percentage point on the harder scenes. A single scene reading 0.98 is not evidence of a 2% deficit, which is why the section above rests on a trend across geometries and on repeated seeds rather than on one number. The same caution applies to any component a scene barely excites; the tables mark those it does not excite at all with a dash.

## Re-running it

```bash
uv run pytest tests/test_scale.py -q          # the independent geometry checks, ~9 s
uv run --group bench python benchmarks/bench.py synth --families all
uv run --group bench python benchmarks/bench.py run --systems spintrack --pin-cpu 2
uv run --group bench python benchmarks/bench.py rescore   # metrics only, no tracking
uv run --group bench python benchmarks/bench.py report --out docs/benchmark.md
```

`rescore` recomputes every metric from the per-frame estimates saved next to each row, which is how a new metric reaches the table without re-running FicTrac - the C++ builds are local and not reproducible from a clone. The scene generator is seeded, so `synth` reproduces the videos themselves from `benchmarks/spintrack_bench/families.py`.
