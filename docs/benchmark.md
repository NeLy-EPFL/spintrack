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
percentage of the true path length, and reads `nan` on a scene whose animal does
not go anywhere (`constant_turn` turns in place, `static` does not move at all);
tracking time excludes video decoding and was measured pinned to one core.

The rotation scale is the one thing the error angles cannot show: a tracker that
reported every rotation 2% small would score the same median as one that is right on
average and noisy. It is the least-squares factor between the reported rotation and
the true one, pooled over the three camera-frame components, and it reads `nan` on
`static`, which does not turn. A wrong assumed ball radius is what moves it (about
twice the relative radius error, on the components about axes in the image plane -
see `tests/test_scale.py`), and `motion_blur` shows the other way it can move: at an
exposure of 0.8 of the frame period the per-frame rotation is the average over that
exposure rather than the instantaneous one, which is 3% small here while leaving the
integrated path exact (`scale_path` in the parquet reads 1.0000 there).

These scenes are rendered with spintrack's own camera and sphere code, so they
cannot see an error in either; `docs/verification.md` says what closes that gap and
what the scale numbers rest on.

Reproducing the synthetic tables from a clone: the scenes are not committed
(`benchmarks/data/` is gitignored) but regenerate from the seeds in `families.py`, and
every number is read back from `benchmarks/results/results.parquet` without rerunning a
tracker, so only the timing columns depend on the machine. The scenes were rendered and
scored, and this file written, with

    uv run --group bench python benchmarks/bench.py synth
    uv run --group bench python benchmarks/bench.py run \
        --systems fictrac fictrac-fork spintrack spintrack-full spintrack-noillum \
        --pin-cpu 2
    uv run --group bench python benchmarks/bench.py report --out docs/benchmark.md

FicTrac is compiled from its own sources - upstream 2.1.2 (`github.com/rjdmoore/fictrac`)
and the NeLy-EPFL fork, which agree to five decimals here - and
`benchmarks/spintrack_bench/runners/fictrac_cpp.py` points at the two binaries. Timings
were measured on an Intel Core i9-14900K with tracking pinned to one core and are
relative to that machine; the error and scale columns are not. The six real recordings
in the last table are lab data and are not distributed with the repository.

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
| jerky_diag        |     0.397 |          0.397 |       0.038 |            0.038 |
| jerky_drop        |     0.502 |          0.502 |       0.039 |            0.039 |
| lab_jerky_drop    |     0.273 |          0.273 |       0.022 |            0.014 |
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
| ball_drop         |   nan     |        nan     |       0.130 |            0.130 |
| clean_fly         |     0.671 |          0.671 |       0.039 |            0.039 |
| constant_forward  |     0.577 |          0.577 |       0.032 |            0.032 |
| constant_side     |     0.596 |          0.596 |       0.030 |            0.030 |
| constant_turn     |     0.603 |          0.603 |       0.029 |            0.029 |
| fast_forward      |     0.541 |          0.541 |       0.055 |            0.055 |
| holder_shadow     |   nan     |        nan     |       0.048 |            0.048 |
| holder_shadow_cut |   nan     |        nan     |       0.107 |            0.095 |
| holder_shadow_lab |   nan     |        nan     |       0.109 |            0.098 |
| jerky_diag        |     1.376 |          1.376 |       0.238 |            0.238 |
| jerky_drop        |     2.421 |          2.421 |       0.277 |            0.277 |
| lab_jerky_drop    |     1.848 |          1.848 |       0.146 |            0.141 |
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
| jerky_diag        |     0.000 |          0.000 |       0.001 |            0.001 |
| jerky_drop        |     0.009 |          0.009 |       0.001 |            0.001 |
| lab_jerky_drop    |     0.002 |          0.002 |       0.000 |            0.000 |
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
| ball_drop         |   nan     |        nan     |       4.223 |            4.223 |
| clean_fly         |    -0.227 |         -0.227 |      -0.032 |           -0.032 |
| constant_forward  |    -0.017 |         -0.017 |       0.065 |            0.065 |
| constant_side     |    -0.109 |         -0.109 |       0.228 |            0.228 |
| constant_turn     |    -0.367 |         -0.367 |       0.311 |            0.311 |
| fast_forward      |    -0.069 |         -0.069 |       0.109 |            0.109 |
| holder_shadow     |   nan     |        nan     |       0.241 |            0.241 |
| holder_shadow_cut |   nan     |        nan     |       0.808 |            0.832 |
| holder_shadow_lab |   nan     |        nan     |       0.326 |            0.399 |
| jerky_diag        |    21.076 |         21.076 |       3.359 |            3.359 |
| jerky_drop        |    42.758 |         42.758 |       2.948 |            2.948 |
| lab_jerky_drop    |    40.635 |         40.635 |       1.158 |            1.171 |
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

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       0.350 |            0.350 |
| clean_fly         |     0.085 |          0.085 |       0.039 |            0.039 |
| constant_forward  |     0.903 |          0.903 |       0.078 |            0.078 |
| constant_side     |     0.906 |          0.906 |       0.170 |            0.170 |
| constant_turn     |   nan     |        nan     |     nan     |          nan     |
| fast_forward      |     6.656 |          6.656 |       1.216 |            1.216 |
| holder_shadow     |   nan     |        nan     |       0.107 |            0.107 |
| holder_shadow_cut |   nan     |        nan     |       0.055 |            0.060 |
| holder_shadow_lab |   nan     |        nan     |       0.090 |            0.105 |
| jerky_diag        |     1.794 |          1.794 |       0.319 |            0.319 |
| jerky_drop        |     6.821 |          6.821 |       0.209 |            0.209 |
| lab_jerky_drop    |     2.786 |          2.786 |       0.047 |            0.044 |
| lab_small_ball    |     0.188 |          0.188 |       0.043 |            0.049 |
| lighting          |     0.024 |          0.024 |       0.027 |            0.027 |
| low_contrast      |     0.135 |          0.135 |       0.051 |            0.051 |
| motion_blur       |     0.835 |          0.835 |       0.851 |            0.851 |
| noisy             |     0.237 |          0.237 |       0.041 |            0.041 |
| occluded          |     0.084 |          0.084 |       0.026 |            0.026 |
| offaxis           |     0.438 |          0.438 |       0.096 |            0.096 |
| random_walk       |     0.065 |          0.065 |       0.086 |            0.086 |
| saccades          |     0.108 |          0.108 |       0.039 |            0.039 |
| sparse            |     0.210 |          0.210 |       1.255 |            1.255 |
| speckle           |  1017.106 |       1017.106 |       0.229 |            0.229 |
| static            |   nan     |        nan     |     nan     |          nan     |

## tracking ms/frame

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         |   nan     |        nan     |       2.794 |            2.798 |
| clean_fly         |     2.100 |          2.100 |       2.235 |            2.269 |
| constant_forward  |     2.000 |          2.000 |       2.106 |            2.112 |
| constant_side     |     2.000 |          2.000 |       2.003 |            2.008 |
| constant_turn     |     2.000 |          2.000 |       2.112 |            2.124 |
| fast_forward      |     2.100 |          2.100 |       2.130 |            2.131 |
| holder_shadow     |   nan     |        nan     |       2.247 |            2.276 |
| holder_shadow_cut |   nan     |        nan     |       2.196 |            4.031 |
| holder_shadow_lab |   nan     |        nan     |       2.716 |            5.174 |
| jerky_diag        |     2.000 |          2.000 |       2.660 |            2.673 |
| jerky_drop        |     2.000 |          2.000 |       2.597 |            2.588 |
| lab_jerky_drop    |     8.900 |          9.000 |       4.840 |            7.185 |
| lab_small_ball    |     6.900 |          6.900 |       2.704 |            5.072 |
| lighting          |     2.100 |          2.100 |       2.276 |            2.281 |
| low_contrast      |     2.100 |          2.100 |       2.210 |            2.208 |
| motion_blur       |     2.000 |          2.000 |       2.235 |            2.237 |
| noisy             |     4.300 |          4.400 |       2.596 |            2.591 |
| occluded          |     2.000 |          2.000 |       2.168 |            2.170 |
| offaxis           |     1.200 |          1.200 |       2.108 |            2.109 |
| random_walk       |     2.200 |          2.200 |       2.376 |            2.376 |
| saccades          |     2.000 |          2.000 |       2.361 |            2.352 |
| sparse            |     2.100 |          2.100 |       2.294 |            2.283 |
| speckle           |     2.200 |          2.200 |       2.930 |            2.928 |
| static            |     2.000 |          2.000 |       1.447 |            1.456 |

## rotation scale (reported / true)

| dataset           |   fictrac |   fictrac-fork |   spintrack |   spintrack-full |
|:------------------|----------:|---------------:|------------:|-----------------:|
| ball_drop         | nan       |      nan       |     1.00006 |          1.00006 |
| clean_fly         |   1.00134 |        1.00134 |     0.99950 |          0.99950 |
| constant_forward  |   1.00027 |        1.00027 |     0.99992 |          0.99992 |
| constant_side     |   0.99998 |        0.99998 |     1.00001 |          1.00001 |
| constant_turn     |   1.00034 |        1.00034 |     1.00000 |          1.00000 |
| fast_forward      |   0.99999 |        0.99999 |     0.99999 |          0.99999 |
| holder_shadow     | nan       |      nan       |     0.99980 |          0.99980 |
| holder_shadow_cut | nan       |      nan       |     0.99863 |          0.99855 |
| holder_shadow_lab | nan       |      nan       |     0.99872 |          0.99854 |
| jerky_diag        |   0.98336 |        0.98336 |     0.99577 |          0.99577 |
| jerky_drop        |   0.96704 |        0.96704 |     1.00068 |          1.00068 |
| lab_jerky_drop    |   0.97004 |        0.97004 |     1.00016 |          1.00020 |
| lab_small_ball    |   0.99920 |        0.99920 |     0.99890 |          0.99889 |
| lighting          |   1.00050 |        1.00050 |     0.99951 |          0.99951 |
| low_contrast      |   1.00178 |        1.00178 |     0.99857 |          0.99857 |
| motion_blur       |   0.96850 |        0.96850 |     0.96913 |          0.96913 |
| noisy             |   0.99845 |        0.99845 |     0.99956 |          0.99956 |
| occluded          |   1.00137 |        1.00137 |     0.99956 |          0.99956 |
| offaxis           |   1.00257 |        1.00257 |     0.99890 |          0.99890 |
| random_walk       |   1.00000 |        1.00000 |     0.99991 |          0.99991 |
| saccades          |   1.00080 |        1.00080 |     1.00004 |          1.00004 |
| sparse            |   1.00307 |        1.00307 |     0.99878 |          0.99878 |
| speckle           | -38.42291 |      -38.42291 |     1.00009 |          1.00009 |
| static            | nan       |      nan       |   nan       |        nan       |

## Scale of each reported component (animal frame)

The same factor fitted per component, on the forward, turning and sideslip rotations
a user reads (`spintrack.path`), with a moving-block bootstrap error that carries the
autocorrelation of the tracking residual. Sideslip is the slack one: these scenes give
it a tenth of the amplitude they give turning, so its error bars are ten times wider
and a reading near 0.98 there is not evidence of a 2% deficit. A dash means the scene
does not turn about that axis at all.

### forward walking

| dataset           | fictrac             | fictrac-fork        | spintrack        | spintrack-full   |
|:------------------|:--------------------|:--------------------|:-----------------|:-----------------|
| ball_drop         | nan                 | nan                 | 1.0019 +- 0.0052 | 1.0019 +- 0.0052 |
| clean_fly         | 0.9988 +- 0.0012    | 0.9988 +- 0.0012    | 0.9999 +- 0.0002 | 0.9999 +- 0.0002 |
| constant_forward  | 1.0003 +- 0.0009    | 1.0003 +- 0.0009    | 0.9999 +- 0.0001 | 0.9999 +- 0.0001 |
| constant_side     | -                   | -                   | -                | -                |
| constant_turn     | -                   | -                   | -                | -                |
| fast_forward      | 1.0000 +- 0.0001    | 1.0000 +- 0.0001    | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 |
| holder_shadow     | nan                 | nan                 | 0.9997 +- 0.0003 | 0.9997 +- 0.0003 |
| holder_shadow_cut | nan                 | nan                 | 0.9990 +- 0.0006 | 0.9989 +- 0.0006 |
| holder_shadow_lab | nan                 | nan                 | 0.9994 +- 0.0003 | 0.9994 +- 0.0003 |
| jerky_diag        | 0.9665 +- 0.0331    | 0.9665 +- 0.0331    | 0.9955 +- 0.0068 | 0.9955 +- 0.0068 |
| jerky_drop        | 0.9850 +- 0.0261    | 0.9850 +- 0.0261    | 1.0010 +- 0.0044 | 1.0010 +- 0.0044 |
| lab_jerky_drop    | 0.9845 +- 0.0296    | 0.9845 +- 0.0296    | 1.0007 +- 0.0017 | 1.0008 +- 0.0017 |
| lab_small_ball    | 0.9986 +- 0.0009    | 0.9986 +- 0.0009    | 0.9998 +- 0.0002 | 0.9997 +- 0.0002 |
| lighting          | 0.9978 +- 0.0014    | 0.9978 +- 0.0014    | 1.0001 +- 0.0002 | 1.0001 +- 0.0002 |
| low_contrast      | 0.9977 +- 0.0018    | 0.9977 +- 0.0018    | 0.9995 +- 0.0004 | 0.9995 +- 0.0004 |
| motion_blur       | 0.9969 +- 0.0020    | 0.9969 +- 0.0020    | 0.9969 +- 0.0013 | 0.9969 +- 0.0013 |
| noisy             | 0.9985 +- 0.0011    | 0.9985 +- 0.0011    | 1.0001 +- 0.0002 | 1.0001 +- 0.0002 |
| occluded          | 1.0002 +- 0.0011    | 1.0002 +- 0.0011    | 1.0000 +- 0.0002 | 1.0000 +- 0.0002 |
| offaxis           | 1.0039 +- 0.0029    | 1.0039 +- 0.0029    | 1.0001 +- 0.0004 | 1.0001 +- 0.0004 |
| random_walk       | 1.0009 +- 0.0017    | 1.0009 +- 0.0017    | 0.9994 +- 0.0003 | 0.9994 +- 0.0003 |
| saccades          | 0.9974 +- 0.0035    | 0.9974 +- 0.0035    | 0.9996 +- 0.0005 | 0.9996 +- 0.0005 |
| sparse            | 0.9995 +- 0.0010    | 0.9995 +- 0.0010    | 0.9993 +- 0.0014 | 0.9993 +- 0.0014 |
| speckle           | -69.7670 +- 12.2002 | -69.7670 +- 12.2002 | 1.0000 +- 0.0002 | 1.0000 +- 0.0002 |
| static            | -                   | -                   | -                | -                |

### turning

| dataset           | fictrac           | fictrac-fork      | spintrack        | spintrack-full   |
|:------------------|:------------------|:------------------|:-----------------|:-----------------|
| ball_drop         | nan               | nan               | 0.9993 +- 0.0008 | 0.9993 +- 0.0008 |
| clean_fly         | 1.0024 +- 0.0021  | 1.0024 +- 0.0021  | 0.9993 +- 0.0005 | 0.9993 +- 0.0005 |
| constant_forward  | -                 | -                 | -                | -                |
| constant_side     | -                 | -                 | -                | -                |
| constant_turn     | 1.0003 +- 0.0010  | 1.0003 +- 0.0010  | 1.0000 +- 0.0002 | 1.0000 +- 0.0002 |
| fast_forward      | -                 | -                 | -                | -                |
| holder_shadow     | nan               | nan               | 0.9999 +- 0.0006 | 0.9999 +- 0.0006 |
| holder_shadow_cut | nan               | nan               | 0.9986 +- 0.0011 | 0.9985 +- 0.0011 |
| holder_shadow_lab | nan               | nan               | 0.9985 +- 0.0007 | 0.9983 +- 0.0009 |
| jerky_diag        | 0.9877 +- 0.0055  | 0.9877 +- 0.0055  | 0.9958 +- 0.0033 | 0.9958 +- 0.0033 |
| jerky_drop        | 0.9580 +- 0.0217  | 0.9580 +- 0.0217  | 1.0007 +- 0.0007 | 1.0007 +- 0.0007 |
| lab_jerky_drop    | 0.9627 +- 0.0178  | 0.9627 +- 0.0178  | 0.9999 +- 0.0002 | 0.9999 +- 0.0002 |
| lab_small_ball    | 0.9997 +- 0.0012  | 0.9997 +- 0.0012  | 0.9987 +- 0.0005 | 0.9987 +- 0.0005 |
| lighting          | 1.0022 +- 0.0021  | 1.0022 +- 0.0021  | 0.9992 +- 0.0005 | 0.9992 +- 0.0005 |
| low_contrast      | 1.0039 +- 0.0040  | 1.0039 +- 0.0040  | 0.9982 +- 0.0009 | 0.9982 +- 0.0009 |
| motion_blur       | 0.9560 +- 0.0100  | 0.9560 +- 0.0100  | 0.9565 +- 0.0103 | 0.9565 +- 0.0103 |
| noisy             | 0.9987 +- 0.0028  | 0.9987 +- 0.0028  | 0.9993 +- 0.0005 | 0.9993 +- 0.0005 |
| occluded          | 1.0018 +- 0.0018  | 1.0018 +- 0.0018  | 0.9995 +- 0.0005 | 0.9995 +- 0.0005 |
| offaxis           | 1.0025 +- 0.0034  | 1.0025 +- 0.0034  | 0.9984 +- 0.0007 | 0.9984 +- 0.0007 |
| random_walk       | 1.0005 +- 0.0011  | 1.0005 +- 0.0011  | 0.9999 +- 0.0002 | 0.9999 +- 0.0002 |
| saccades          | 1.0014 +- 0.0013  | 1.0014 +- 0.0013  | 1.0001 +- 0.0002 | 1.0001 +- 0.0002 |
| sparse            | 1.0071 +- 0.0045  | 1.0071 +- 0.0045  | 0.9982 +- 0.0011 | 0.9982 +- 0.0011 |
| speckle           | -3.8058 +- 2.6384 | -3.8058 +- 2.6384 | 1.0002 +- 0.0002 | 1.0002 +- 0.0002 |
| static            | -                 | -                 | -                | -                |

### sideslip

| dataset           | fictrac            | fictrac-fork       | spintrack        | spintrack-full   |
|:------------------|:-------------------|:-------------------|:-----------------|:-----------------|
| ball_drop         | nan                | nan                | 0.9845 +- 0.0130 | 0.9845 +- 0.0130 |
| clean_fly         | 1.0201 +- 0.0182   | 1.0201 +- 0.0182   | 0.9988 +- 0.0035 | 0.9988 +- 0.0035 |
| constant_forward  | -                  | -                  | -                | -                |
| constant_side     | 1.0000 +- 0.0010   | 1.0000 +- 0.0010   | 1.0000 +- 0.0001 | 1.0000 +- 0.0001 |
| constant_turn     | -                  | -                  | -                | -                |
| fast_forward      | -                  | -                  | -                | -                |
| holder_shadow     | nan                | nan                | 0.9999 +- 0.0041 | 0.9999 +- 0.0041 |
| holder_shadow_cut | nan                | nan                | 0.9827 +- 0.0066 | 0.9801 +- 0.0065 |
| holder_shadow_lab | nan                | nan                | 0.9818 +- 0.0080 | 0.9773 +- 0.0072 |
| jerky_diag        | 0.6849 +- 0.3534   | 0.6849 +- 0.3534   | 0.9989 +- 0.0303 | 0.9989 +- 0.0303 |
| jerky_drop        | 1.0693 +- 0.1595   | 1.0693 +- 0.1595   | 0.9753 +- 0.0152 | 0.9753 +- 0.0152 |
| lab_jerky_drop    | 1.0489 +- 0.1967   | 1.0489 +- 0.1967   | 1.0021 +- 0.0056 | 1.0009 +- 0.0053 |
| lab_small_ball    | 0.9758 +- 0.0215   | 0.9758 +- 0.0215   | 0.9776 +- 0.0070 | 0.9768 +- 0.0065 |
| lighting          | 0.9415 +- 0.0305   | 0.9415 +- 0.0305   | 1.0010 +- 0.0029 | 1.0010 +- 0.0029 |
| low_contrast      | 0.9737 +- 0.0323   | 0.9737 +- 0.0323   | 0.9938 +- 0.0054 | 0.9938 +- 0.0054 |
| motion_blur       | 0.8808 +- 0.0251   | 0.8808 +- 0.0251   | 0.9420 +- 0.0103 | 0.9420 +- 0.0103 |
| noisy             | 0.9578 +- 0.0291   | 0.9578 +- 0.0291   | 1.0044 +- 0.0030 | 1.0044 +- 0.0030 |
| occluded          | 1.0131 +- 0.0282   | 1.0131 +- 0.0282   | 0.9840 +- 0.0068 | 0.9840 +- 0.0068 |
| offaxis           | 0.9227 +- 0.0524   | 0.9227 +- 0.0524   | 0.9890 +- 0.0081 | 0.9890 +- 0.0081 |
| random_walk       | 0.9988 +- 0.0016   | 0.9988 +- 0.0016   | 1.0003 +- 0.0002 | 1.0003 +- 0.0002 |
| saccades          | 0.9413 +- 0.0290   | 0.9413 +- 0.0290   | 0.9947 +- 0.0033 | 0.9947 +- 0.0033 |
| sparse            | 1.0208 +- 0.0205   | 1.0208 +- 0.0205   | 0.9970 +- 0.0032 | 0.9970 +- 0.0032 |
| speckle           | 46.4192 +- 73.2758 | 46.4192 +- 73.2758 | 1.0010 +- 0.0012 | 1.0010 +- 0.0012 |
| static            | -                  | -                  | -                | -                |

## Agreement with FicTrac on real recordings

Six 60 s trials (1600x1008 HEVC, 100 fps, `q_factor 12`, `accumulate_map: n`) tracked
with spintrack and compared with the lab fork's FicTrac output for the same video.
There is no ground truth here; differences are per-frame angles between the two
trackers' lab-frame rotation increments.

The scale columns are spintrack's reported amplitude over FicTrac's, per component,
by the lagged instrument in `spintrack_bench.agreement.scale_ratio` - which the
synthetic scenes, where both systems' gains against truth are known, put within 0.3%
(`notes/session_2026-09-08d/scale_estimator.py`; the statistics one reaches for first,
a regression either way or a ratio of standard deviations, are diluted by the noisier
series and read 0.78 to 0.99 there where the answer is 1.00). On a scene built with
these recordings' own geometry it reads the known forward and turning ratios within
0.7%, and the known *sideslip* ratio 4 to 6 points low - so the side column carries
an error bar this comparison has not pinned down, rather than agreement to a percent.

Forward agrees to within 3%. **spintrack reports 1.5 to 4.5% less turning than
FicTrac on the five trials the animal walked through, and 5% less on 008**, whose
correlations are the worst of the six and whose forward column the estimator refuses
outright for want of a low-frequency signal to instrument. Nothing here says which
tracker is right - that needs truth these recordings do not have - but it is
systematic and it is on the component most of these experiments report.

It is a **slow-turning** effect. Binned into 6 s windows by how fast the animal was
actually turning, the ratio runs 0.934 +- 0.011 below 0.08 deg/frame and 0.977
above 0.12. That is not, however, why no scene here shows it: re-rendering
this geometry with `fly_walk` scaled down to the real trials' own rate leaves
spintrack within 0.2% of truth and FicTrac within 1.2%, with the ratio moving the
wrong way (1.010, spintrack the higher). See `docs/verification.md`.

What is worth knowing about every table above: **these scenes have one activity
level.** The families vary the optics, the lighting, the occluders, the noise and the
codec, but all of them drive the ball with the same seeded `fly_walk`, so 14 of the
17 scenes with an identified turn gain turn at exactly 0.871 deg/frame and the other
three at 0.87 to 1.39 - 5 to 12 times more active than these recordings, whose
animals turn at 0.07 to 0.16 deg/frame and walk at 0.04 to 0.09. Read the gains as
one operating point rather than a range that covers a real experiment.
`docs/verification.md` has the rate table and what is ruled out.

|   trial |   frames |   dropped |   median diff (deg) |   p95 diff (deg) |   turn corr |   forward scale |   turn scale |   side scale |   heading diff (deg) |   endpoint diff (%) |   tracking ms/frame |   fps incl. decode |
|--------:|---------:|----------:|--------------------:|-----------------:|------------:|----------------:|-------------:|-------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|     005 |     6015 |         0 |               0.090 |            0.257 |       0.919 |           0.974 |        0.970 |        1.001 |               -1.925 |               0.125 |               3.214 |            246.831 |
|     006 |     6015 |         0 |               0.114 |            0.305 |       0.951 |           0.997 |        0.986 |        1.004 |                0.809 |               0.259 |               3.476 |            231.486 |
|     007 |     6012 |         0 |               0.084 |            0.281 |       0.942 |           1.004 |        0.977 |        1.006 |               -0.199 |               0.172 |               3.326 |            240.418 |
|     008 |     6010 |         0 |               0.105 |            0.465 |       0.781 |         nan     |        0.947 |        0.963 |               -3.489 |               6.135 |               3.999 |            204.815 |
|     009 |    12015 |         0 |               0.087 |            0.264 |       0.899 |           0.969 |        0.956 |        1.011 |                2.561 |               0.352 |               3.289 |            243.126 |
|     012 |     6010 |         0 |               0.216 |            0.508 |       0.961 |           0.995 |        0.983 |        1.005 |               18.713 |               2.294 |               3.764 |            215.935 |
