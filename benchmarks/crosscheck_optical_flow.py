"""Cross-check a run's rotation against phase correlation, independently.

It compares the image shift that phase correlation measures with the one a run's
rotation predicts.

The ball circle comes from the config's `ball.rim`. A 256 px patch on the upper ball is
tracked by phase correlation; the prediction is the mean of `(w x p) r` over the patch,
with `p` on the near side of the sphere. A fixed patch reads a ball that moves in its
holder as motion, so use it on recordings where the ball stays put.

    uv run python benchmarks/crosscheck_optical_flow.py CONFIG TRACKS [FIRST] [COUNT]

TRACKS is the run's `tracks.parquet`.
"""

import sys

import cv2
import numpy as np
import polars as pl

from spintrack.config import Config


def circle_from_points(pts):
    a = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    sol, *_ = np.linalg.lstsq(a, (pts**2).sum(1), rcond=None)
    return sol[0], sol[1], np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)


cfg = Config.load(sys.argv[1])
d = pl.read_parquet(sys.argv[2]).to_numpy()
n0 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
n = int(sys.argv[4]) if len(sys.argv) > 4 else 800
n = min(n, len(d) - n0 - 1)
cx, cy, r = circle_from_points(np.asarray(cfg.ball.rim, float))

s = 256
x0, y0 = int(cx - s / 2), int(cy - 260)  # upper ball, clear of the cut-off bottom
yy, xx = np.mgrid[y0 : y0 + s, x0 : x0 + s].astype(np.float64)
px, py = (xx - cx) / r, (yy - cy) / r
pz = -np.sqrt(np.clip(1 - px * px - py * py, 0, None))  # near side: toward the camera
p = np.stack([px, py, pz], -1)

cap = cv2.VideoCapture(str(cfg.video))
cap.set(cv2.CAP_PROP_POS_FRAMES, n0)
win = cv2.createHanningWindow((s, s), cv2.CV_64F)
prev, meas = None, []
for _ in range(n + 1):
    ok, img = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0 : y0 + s, x0 : x0 + s].astype(
        np.float64
    )
    if prev is not None:
        (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
        meas.append((dx, dy, resp))
    prev = g
meas = np.array(meas)

# Rows are looked up by their frame counter (column 0): a dropped frame would otherwise
# shift every later row by one and wreck the comparison.
row_of = {int(f): i for i, f in enumerate(d[:, 0])}
w = np.array(
    [
        d[row_of[f], 1:4] if f in row_of else np.zeros(3)
        for f in range(n0 + 1, n0 + 1 + len(meas))
    ]
)
v = np.cross(w[:, None, None, :], p[None]) * r
pred = v[..., :2].reshape(len(w), -1, 2).mean(1)

print(f"ball ({cx:.1f}, {cy:.1f}) r {r:.1f} px; frames {n0}-{n0 + len(meas)}")
for k, name in enumerate("xy"):
    a, b = pred[:, k], meas[:, k]
    slope = np.polyfit(b, a, 1)[0]
    print(
        f"  {name}: corr {np.corrcoef(a, b)[0, 1]:.4f}  slope {slope:.3f}  "
        f"rms resid {np.sqrt(np.mean((a - b) ** 2)):.3f} px  "
        f"(shift rms {np.sqrt(np.mean(b**2)):.3f} px)"
    )
mp, mm = np.hypot(*pred.T), np.hypot(*meas[:, :2].T)
big = mm > 1.0
print(
    f"  |shift|>1px: {big.sum()} frames, "
    f"median ratio pred/meas {np.median(mp[big] / mm[big]):.3f}"
)
