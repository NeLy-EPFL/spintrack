# Benchmark results

Synthetic scenes rendered by `benchmarks/bench.py synth` (1000 frames at 100 fps, exact
ground-truth rotations; see `benchmarks/spintrack_bench/families.py` for each family).
`fictrac` and `fictrac-fork` are the C++ FicTrac 2.1.2 upstream and NeLy-EPFL builds run as
black boxes on the same videos and configs (FicTrac's window-frame rotation vectors are
mapped to camera coordinates before scoring). `spintrack` is the default configuration
(solve on at most ~4000 window pixels, with the rig's static illumination separated
from the ball's texture); `spintrack-full` solves on every pixel. Errors are
per-frame rotation errors in degrees; drift is the slope of the accumulated
orientation error; endpoint error is the fictive-path endpoint discrepancy as a
percentage of the true path length; tracking time excludes video decoding and was
measured pinned to one core.

## median err (deg)

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       0.042 |            0.042 |
| clean_fly         |     0.243 |          0.243 |       0.019 |            0.019 |
| constant_forward  |     0.225 |          0.225 |       0.016 |            0.016 |
| constant_side     |     0.264 |          0.264 |       0.013 |            0.013 |
| constant_turn     |     0.259 |          0.259 |       0.015 |            0.015 |
| fast_forward      |     0.230 |          0.230 |       0.028 |            0.028 |
| holder_shadow     |   nan     |        nan     |       0.023 |            0.023 |
| holder_shadow_cut |   nan     |        nan     |       0.052 |            0.047 |
| holder_shadow_lab |   nan     |        nan     |       0.053 |            0.050 |
| lab_small_ball    |     0.165 |          0.165 |       0.046 |            0.042 |
| lighting          |     0.253 |          0.253 |       0.020 |            0.020 |
| low_contrast      |     0.346 |          0.346 |       0.029 |            0.029 |
| motion_blur       |     0.304 |          0.304 |       0.092 |            0.092 |
| noisy             |     0.258 |          0.258 |       0.018 |            0.018 |
| occluded          |     0.275 |          0.275 |       0.036 |            0.036 |
| offaxis           |     0.385 |          0.385 |       0.029 |            0.029 |
| random_walk       |     0.314 |          0.314 |       0.018 |            0.018 |
| saccades          |     0.242 |          0.242 |       0.017 |            0.017 |
| sparse            |     0.341 |          0.341 |       0.026 |            0.026 |
| speckle           |    90.266 |         90.266 |       0.020 |            0.020 |
| static            |     0.035 |          0.035 |       0.002 |            0.002 |

## p95 err (deg)

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       0.153 |            0.153 |
| clean_fly         |     0.671 |          0.671 |       0.039 |            0.039 |
| constant_forward  |     0.577 |          0.577 |       0.032 |            0.032 |
| constant_side     |     0.596 |          0.596 |       0.030 |            0.030 |
| constant_turn     |     0.603 |          0.603 |       0.029 |            0.029 |
| fast_forward      |     0.541 |          0.541 |       0.055 |            0.055 |
| holder_shadow     |   nan     |        nan     |       0.048 |            0.048 |
| holder_shadow_cut |   nan     |        nan     |       0.107 |            0.095 |
| holder_shadow_lab |   nan     |        nan     |       0.109 |            0.098 |
| lab_small_ball    |     0.432 |          0.432 |       0.091 |            0.081 |
| lighting          |     0.686 |          0.686 |       0.042 |            0.042 |
| low_contrast      |     0.902 |          0.902 |       0.061 |            0.061 |
| motion_blur       |     0.845 |          0.845 |       0.448 |            0.448 |
| noisy             |     0.632 |          0.632 |       0.039 |            0.039 |
| occluded          |     0.735 |          0.735 |       0.082 |            0.082 |
| offaxis           |     1.090 |          1.090 |       0.063 |            0.063 |
| random_walk       |     0.641 |          0.641 |       0.036 |            0.036 |
| saccades          |     0.611 |          0.611 |       0.036 |            0.036 |
| sparse            |     0.835 |          0.835 |       0.062 |            0.062 |
| speckle           |   175.677 |        175.677 |       0.041 |            0.041 |
| static            |     0.085 |          0.085 |       0.004 |            0.004 |

## fail frac

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       0.000 |            0.000 |
| clean_fly         |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_forward  |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_side     |     0.000 |          0.000 |       0.000 |            0.000 |
| constant_turn     |     0.000 |          0.000 |       0.000 |            0.000 |
| fast_forward      |     0.000 |          0.000 |       0.000 |            0.000 |
| holder_shadow     |   nan     |        nan     |       0.000 |            0.000 |
| holder_shadow_cut |   nan     |        nan     |       0.000 |            0.000 |
| holder_shadow_lab |   nan     |        nan     |       0.000 |            0.000 |
| lab_small_ball    |     0.000 |          0.000 |       0.000 |            0.000 |
| lighting          |     0.000 |          0.000 |       0.000 |            0.000 |
| low_contrast      |     0.000 |          0.000 |       0.000 |            0.000 |
| motion_blur       |     0.000 |          0.000 |       0.000 |            0.000 |
| noisy             |     0.000 |          0.000 |       0.000 |            0.000 |
| occluded          |     0.000 |          0.000 |       0.000 |            0.000 |
| offaxis           |     0.000 |          0.000 |       0.000 |            0.000 |
| random_walk       |     0.000 |          0.000 |       0.000 |            0.000 |
| saccades          |     0.000 |          0.000 |       0.000 |            0.000 |
| sparse            |     0.000 |          0.000 |       0.000 |            0.000 |
| speckle           |     0.920 |          0.920 |       0.000 |            0.000 |
| static            |     0.000 |          0.000 |       0.000 |            0.000 |

## drift (deg/min)

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       4.890 |            4.890 |
| clean_fly         |    -0.227 |         -0.227 |      -0.032 |           -0.032 |
| constant_forward  |    -0.017 |         -0.017 |       0.065 |            0.065 |
| constant_side     |    -0.109 |         -0.109 |       0.228 |            0.228 |
| constant_turn     |    -0.367 |         -0.367 |       0.311 |            0.311 |
| fast_forward      |    -0.069 |         -0.069 |       0.109 |            0.109 |
| holder_shadow     |   nan     |        nan     |       0.241 |            0.241 |
| holder_shadow_cut |   nan     |        nan     |       0.808 |            0.832 |
| holder_shadow_lab |   nan     |        nan     |       0.326 |            0.399 |
| lab_small_ball    |    -1.327 |         -1.327 |      -0.010 |            0.052 |
| lighting          |    -0.969 |         -0.969 |      -0.044 |           -0.044 |
| low_contrast      |    -1.030 |         -1.030 |       0.151 |            0.151 |
| motion_blur       |    -1.351 |         -1.351 |      -0.719 |           -0.719 |
| noisy             |    -0.151 |         -0.151 |      -0.045 |           -0.045 |
| occluded          |     0.138 |          0.138 |       0.090 |            0.090 |
| offaxis           |     2.050 |          2.050 |       0.439 |            0.439 |
| random_walk       |    -0.531 |         -0.531 |       0.043 |            0.043 |
| saccades          |    -0.071 |         -0.071 |      -0.380 |           -0.380 |
| sparse            |    -0.837 |         -0.837 |      -0.499 |           -0.499 |
| speckle           |   160.536 |        160.536 |       0.121 |            0.121 |
| static            |     0.131 |          0.131 |       0.275 |            0.275 |

## endpoint err (%)

| dataset           |             fictrac |        fictrac-fork |          spintrack |     spintrack-full |
|:------------------|--------------------:|--------------------:|-------------------:|-------------------:|
| ball_drop         |             nan     |             nan     |              0.442 |              0.442 |
| clean_fly         |               0.085 |               0.085 |              0.039 |              0.039 |
| constant_forward  |               0.903 |               0.903 |              0.078 |              0.078 |
| constant_side     |               0.906 |               0.906 |              0.170 |              0.170 |
| constant_turn     | 214141060928959.250 | 214141060928959.250 | 96518056638879.062 | 96518056638879.062 |
| fast_forward      |               6.656 |               6.656 |              1.216 |              1.216 |
| holder_shadow     |             nan     |             nan     |              0.107 |              0.107 |
| holder_shadow_cut |             nan     |             nan     |              0.055 |              0.060 |
| holder_shadow_lab |             nan     |             nan     |              0.090 |              0.105 |
| lab_small_ball    |               0.188 |               0.188 |              0.043 |              0.049 |
| lighting          |               0.024 |               0.024 |              0.027 |              0.027 |
| low_contrast      |               0.135 |               0.135 |              0.051 |              0.051 |
| motion_blur       |               0.835 |               0.835 |              0.851 |              0.851 |
| noisy             |               0.237 |               0.237 |              0.041 |              0.041 |
| occluded          |               0.084 |               0.084 |              0.026 |              0.026 |
| offaxis           |               0.438 |               0.438 |              0.096 |              0.096 |
| random_walk       |               0.065 |               0.065 |              0.086 |              0.086 |
| saccades          |               0.108 |               0.108 |              0.039 |              0.039 |
| sparse            |               0.210 |               0.210 |              1.255 |              1.255 |
| speckle           |            1017.106 |            1017.106 |              0.229 |              0.229 |
| static            |             nan     |             nan     |            nan     |            nan     |

## tracking ms/frame

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       3.005 |            2.824 |
| clean_fly         |     2.100 |          2.100 |       2.469 |            3.040 |
| constant_forward  |     2.000 |          2.000 |       2.452 |            2.711 |
| constant_side     |     2.000 |          2.000 |       2.244 |            2.654 |
| constant_turn     |     2.000 |          2.000 |       2.409 |            2.687 |
| fast_forward      |     2.100 |          2.100 |       2.397 |            2.149 |
| holder_shadow     |   nan     |        nan     |       2.450 |            2.329 |
| holder_shadow_cut |   nan     |        nan     |       2.475 |            4.228 |
| holder_shadow_lab |   nan     |        nan     |       3.034 |            5.297 |
| lab_small_ball    |     6.900 |          6.900 |       2.708 |            5.311 |
| lighting          |     2.100 |          2.100 |       2.589 |            2.292 |
| low_contrast      |     2.100 |          2.100 |       2.435 |            2.485 |
| motion_blur       |     2.000 |          2.000 |       2.538 |            2.335 |
| noisy             |     4.300 |          4.400 |       2.718 |            2.464 |
| occluded          |     2.000 |          2.000 |       2.367 |            2.314 |
| offaxis           |     1.200 |          1.200 |       1.886 |            1.980 |
| random_walk       |     2.200 |          2.200 |       2.571 |            2.388 |
| saccades          |     2.000 |          2.000 |       2.385 |            2.379 |
| sparse            |     2.100 |          2.100 |       2.327 |            2.378 |
| speckle           |     2.200 |          2.200 |       3.010 |            2.786 |
| static            |     2.000 |          2.000 |       1.614 |            1.438 |

## Agreement with FicTrac on real recordings

Six 60 s trials (1600x1008 HEVC, 100 fps, `q_factor 12`, `accumulate_map: n`) tracked
with spintrack and compared with the lab fork's FicTrac output for the same video.
There is no ground truth here; differences are per-frame angles between the two
trackers' lab-frame rotation increments.

|   trial |   frames |   dropped |   median diff (deg) |   p95 diff (deg) |   turn corr |   heading diff (deg) |   endpoint diff (%) |   tracking ms/frame |   fps incl. decode |
|--------:|---------:|----------:|--------------------:|-----------------:|------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|     005 |     6015 |         0 |               0.090 |            0.257 |       0.919 |               -1.925 |               0.125 |               3.353 |            244.190 |
|     006 |     6015 |         0 |               0.114 |            0.307 |       0.952 |                0.726 |               0.256 |               3.521 |            232.849 |
|     007 |     6012 |         0 |               0.084 |            0.282 |       0.942 |               -0.200 |               0.173 |               3.331 |            244.623 |
|     008 |     6010 |         0 |               0.106 |            0.504 |       0.784 |               -3.093 |               6.031 |               3.836 |            217.545 |
|     009 |    12015 |         0 |               0.087 |            0.266 |       0.898 |                3.129 |               0.375 |               3.339 |            245.179 |
|     012 |     6010 |         0 |               0.216 |            0.508 |       0.961 |               18.713 |               2.294 |               3.610 |            227.415 |
