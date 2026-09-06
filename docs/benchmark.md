# Benchmark results

Synthetic scenes rendered by `benchmarks/bench.py synth` (1000 frames at 100 fps, exact
ground-truth rotations; see `benchmarks/spintrack_bench/families.py` for each family).
`fictrac` and `fictrac-fork` are the C++ FicTrac 2.1.2 upstream and NeLy-EPFL builds run as
black boxes on the same videos and configs (FicTrac's window-frame rotation vectors are
mapped to camera coordinates before scoring). `spintrack` is the default configuration
(solve on at most ~4000 window pixels); `spintrack-full` solves on every pixel. Errors
are per-frame rotation errors in degrees; drift is the slope of the accumulated
orientation error; endpoint error is the fictive-path endpoint discrepancy as a
percentage of the true path length; tracking time excludes video decoding and was
measured pinned to one core.

`ball_drop` was added after the other scenes: its ball sinks 0.4 radii in its holder and comes back over frames 300-840 while the tethered fly stays put, which is what `spintrack.refit` is for. Only the `spintrack` column was measured for it, on a different machine from the rest of the table, so compare its timing with nothing here.

## median err (deg)

| dataset          |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:-----------------|----------:|---------------:|------------:|-----------------:|
| ball_drop        |   nan     |        nan     |       0.069 |          nan     |
| clean_fly        |     0.243 |          0.243 |       0.041 |            0.041 |
| constant_forward |     0.225 |          0.225 |       0.043 |            0.043 |
| constant_side    |     0.264 |          0.264 |       0.041 |            0.041 |
| constant_turn    |     0.259 |          0.259 |       0.045 |            0.045 |
| fast_forward     |     0.230 |          0.230 |       0.048 |            0.048 |
| lab_small_ball   |     0.165 |          0.165 |       0.049 |            0.043 |
| lighting         |     0.253 |          0.253 |       0.040 |            0.040 |
| low_contrast     |     0.346 |          0.346 |       0.046 |            0.046 |
| motion_blur      |     0.304 |          0.304 |       0.098 |            0.098 |
| noisy            |     0.258 |          0.258 |       0.043 |            0.043 |
| occluded         |     0.275 |          0.275 |       0.046 |            0.046 |
| offaxis          |     0.385 |          0.385 |       0.047 |            0.047 |
| random_walk      |     0.314 |          0.314 |       0.045 |            0.045 |
| saccades         |     0.242 |          0.242 |       0.034 |            0.034 |
| sparse           |     0.341 |          0.341 |       0.062 |            0.062 |
| speckle          |    90.266 |         90.266 |       0.037 |            0.037 |
| static           |     0.035 |          0.035 |       0.002 |            0.002 |

## p95 err (deg)

| dataset          |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:-----------------|----------:|---------------:|------------:|-----------------:|
| ball_drop        |   nan     |        nan     |       0.245 |          nan     |
| clean_fly        |     0.671 |          0.671 |       0.091 |            0.091 |
| constant_forward |     0.577 |          0.577 |       0.090 |            0.090 |
| constant_side    |     0.596 |          0.596 |       0.088 |            0.088 |
| constant_turn    |     0.603 |          0.603 |       0.091 |            0.091 |
| fast_forward     |     0.541 |          0.541 |       0.098 |            0.098 |
| lab_small_ball   |     0.432 |          0.432 |       0.099 |            0.085 |
| lighting         |     0.686 |          0.686 |       0.092 |            0.092 |
| low_contrast     |     0.902 |          0.902 |       0.110 |            0.110 |
| motion_blur      |     0.845 |          0.845 |       0.459 |            0.459 |
| noisy            |     0.632 |          0.632 |       0.096 |            0.096 |
| occluded         |     0.735 |          0.735 |       0.107 |            0.107 |
| offaxis          |     1.090 |          1.090 |       0.126 |            0.126 |
| random_walk      |     0.641 |          0.641 |       0.091 |            0.091 |
| saccades         |     0.611 |          0.611 |       0.081 |            0.081 |
| sparse           |     0.835 |          0.835 |       0.159 |            0.159 |
| speckle          |   175.677 |        175.677 |       0.091 |            0.091 |
| static           |     0.085 |          0.085 |       0.005 |            0.005 |

## fail frac

| dataset          |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:-----------------|----------:|---------------:|------------:|-----------------:|
| ball_drop        |   nan     |        nan     |       0.001 |          nan     |
| clean_fly        |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_forward |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_side    |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_turn    |     0.000 |          0.000 |       0.000 |            0.000 |
| fast_forward     |     0.000 |          0.000 |       0.000 |            0.000 |
| lab_small_ball   |     0.000 |          0.000 |       0.000 |            0.000 |
| lighting         |     0.000 |          0.000 |       0.000 |            0.000 |
| low_contrast     |     0.000 |          0.000 |       0.000 |            0.000 |
| motion_blur      |     0.000 |          0.000 |       0.000 |            0.000 |
| noisy            |     0.000 |          0.000 |       0.000 |            0.000 |
| occluded         |     0.000 |          0.000 |       0.000 |            0.000 |
| offaxis          |     0.000 |          0.000 |       0.000 |            0.000 |
| random_walk      |     0.000 |          0.000 |       0.000 |            0.000 |
| saccades         |     0.000 |          0.000 |       0.000 |            0.000 |
| sparse           |     0.000 |          0.000 |       0.000 |            0.000 |
| speckle          |     0.920 |          0.920 |       0.000 |            0.000 |
| static           |     0.000 |          0.000 |       0.000 |            0.000 |

## drift (deg/min)

| dataset          |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:-----------------|----------:|---------------:|------------:|-----------------:|
| ball_drop        |   nan     |        nan     |       9.839 |          nan     |
| clean_fly        |    -0.227 |         -0.227 |      -0.041 |           -0.041 |
| constant_forward |    -0.017 |         -0.017 |       0.087 |            0.087 |
| constant_side    |    -0.109 |         -0.109 |      -0.041 |           -0.041 |
| constant_turn    |    -0.367 |         -0.367 |       0.050 |            0.050 |
| fast_forward     |    -0.069 |         -0.069 |       0.022 |            0.022 |
| lab_small_ball   |    -1.327 |         -1.327 |       0.104 |            0.073 |
| lighting         |    -0.969 |         -0.969 |       0.025 |            0.025 |
| low_contrast     |    -1.030 |         -1.030 |      -0.008 |           -0.008 |
| motion_blur      |    -1.351 |         -1.351 |      -0.705 |           -0.705 |
| noisy            |    -0.151 |         -0.151 |      -0.031 |           -0.031 |
| occluded         |     0.138 |          0.138 |       0.132 |            0.132 |
| offaxis          |     2.050 |          2.050 |      -0.338 |           -0.338 |
| random_walk      |    -0.531 |         -0.531 |      -0.016 |           -0.016 |
| saccades         |    -0.071 |         -0.071 |      -0.395 |           -0.395 |
| sparse           |    -0.837 |         -0.837 |      -0.491 |           -0.491 |
| speckle          |   160.536 |        160.536 |      -0.068 |           -0.068 |
| static           |     0.131 |          0.131 |       0.523 |            0.523 |

## endpoint err (%)

| dataset          |             fictrac |        fictrac-fork |           spintrack |      spintrack-full |
|:-----------------|--------------------:|--------------------:|--------------------:|--------------------:|
| ball_drop        |             nan     |             nan     |               1.002 |             nan     |
| clean_fly        |               0.085 |               0.085 |               0.040 |               0.040 |
| constant_forward |               0.903 |               0.903 |               0.316 |               0.316 |
| constant_side    |               0.906 |               0.906 |               0.131 |               0.131 |
| constant_turn    | 214141060928959.250 | 214141060928959.250 | 169634058865333.844 | 169634058865333.844 |
| fast_forward     |               6.656 |               6.656 |               0.349 |               0.349 |
| lab_small_ball   |               0.188 |               0.188 |               0.010 |               0.023 |
| lighting         |               0.024 |               0.024 |               0.035 |               0.035 |
| low_contrast     |               0.135 |               0.135 |               0.061 |               0.061 |
| motion_blur      |               0.835 |               0.835 |               0.853 |               0.853 |
| noisy            |               0.237 |               0.237 |               0.080 |               0.080 |
| occluded         |               0.084 |               0.084 |               0.057 |               0.057 |
| offaxis          |               0.438 |               0.438 |               0.166 |               0.166 |
| random_walk      |               0.065 |               0.065 |               0.016 |               0.016 |
| saccades         |               0.108 |               0.108 |               0.104 |               0.104 |
| sparse           |               0.210 |               0.210 |               0.880 |               0.880 |
| speckle          |            1017.106 |            1017.106 |               0.020 |               0.020 |
| static           |             nan     |             nan     |             nan     |             nan     |

## tracking ms/frame

| dataset          |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:-----------------|----------:|---------------:|------------:|-----------------:|
| ball_drop        |   nan     |        nan     |       3.800 |          nan     |
| clean_fly        |     2.100 |          2.100 |       0.667 |            0.753 |
| constant_forward |     2.000 |          2.000 |       0.587 |            0.666 |
| constant_side    |     2.000 |          2.000 |       0.575 |            0.650 |
| constant_turn    |     2.000 |          2.000 |       0.577 |            0.642 |
| fast_forward     |     2.100 |          2.100 |       0.602 |            0.682 |
| lab_small_ball   |     6.900 |          6.900 |       0.936 |            2.504 |
| lighting         |     2.100 |          2.100 |       0.691 |            0.863 |
| low_contrast     |     2.100 |          2.100 |       0.698 |            0.828 |
| motion_blur      |     2.000 |          2.000 |       0.631 |            0.733 |
| noisy            |     4.300 |          4.400 |       0.755 |            0.757 |
| occluded         |     2.000 |          2.000 |       0.727 |            0.735 |
| offaxis          |     1.200 |          1.200 |       0.564 |            0.555 |
| random_walk      |     2.200 |          2.200 |       0.770 |            0.738 |
| saccades         |     2.000 |          2.000 |       0.760 |            0.715 |
| sparse           |     2.100 |          2.100 |       0.865 |            0.816 |
| speckle          |     2.200 |          2.200 |       1.492 |            1.450 |
| static           |     2.000 |          2.000 |       0.376 |            0.389 |

## Agreement with FicTrac on real recordings

Six 60 s trials (1600x1008 HEVC, 100 fps, `q_factor 12`, `accumulate_map: n`) tracked
with spintrack and compared with the lab fork's FicTrac output for the same video.
There is no ground truth here; differences are per-frame angles between the two
trackers' lab-frame rotation increments.

|   trial |   frames |   dropped |   median diff (deg) |   p95 diff (deg) |   turn corr |   heading diff (deg) |   endpoint diff (%) |   tracking ms/frame |   fps incl. decode |
|--------:|---------:|----------:|--------------------:|-----------------:|------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|     005 |     6015 |         0 |               0.089 |            0.259 |       0.918 |               -1.771 |               0.108 |               1.604 |            426.725 |
|     006 |     6015 |         0 |               0.116 |            0.312 |       0.950 |                1.093 |               0.270 |               1.660 |            418.010 |
|     007 |     6012 |         0 |               0.084 |            0.284 |       0.941 |               -1.144 |               0.064 |               1.587 |            434.562 |
|     008 |     6010 |         0 |               0.100 |            0.439 |       0.796 |                5.627 |               4.845 |               1.896 |            381.553 |
|     009 |    12015 |         0 |               0.085 |            0.267 |       0.895 |                1.818 |               0.361 |               2.193 |            313.491 |
|     012 |     6010 |         0 |               0.224 |            0.516 |       0.959 |               18.674 |               2.272 |               1.997 |            347.442 |
