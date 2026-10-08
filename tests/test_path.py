from pathlib import Path

import numpy as np

from spintrack.io.dat import read_dat
from spintrack.path import PathIntegrator
from spintrack.tracker import FrameResult


def test_forward_walking_goes_along_world_x():
    integ = PathIntegrator()
    for _ in range(100):
        s = integ.step([0.0, 0.02, 0.0])
    assert np.isclose(s.pos_x, 2.0) and np.isclose(s.pos_y, 0.0)
    assert s.heading == 0.0 and np.isclose(s.int_x, 2.0) and s.step_dir == 0.0


def test_sidestep_right_goes_along_world_y():
    s = None
    integ = PathIntegrator()
    for _ in range(10):
        s = integ.step([-0.01, 0.0, 0.0])
    assert np.isclose(s.pos_y, 0.1) and np.isclose(s.pos_x, 0.0)
    assert np.isclose(s.step_dir, np.pi / 2) and np.isclose(s.int_y, 0.1)


def test_path_columns_match_fictrac():
    """FicTrac's own lab-frame rotations reproduce its path columns.

    The excerpt is the first 200 rows FicTrac wrote for lab trial
    AN07B017_260414_Fly4_003, which turn 150 degrees in all and wrap the heading
    through zero.
    """
    rows = read_dat(Path(__file__).parent / "data" / "fictrac-003-head.dat")
    integ = PathIntegrator()
    for row in rows:
        s = integ.step(row[5:8])
        got = (s.pos_x, s.pos_y, s.heading, s.step_dir, s.step_mag, s.int_x, s.int_y)
        assert np.allclose(got, row[14:21], rtol=0, atol=1e-12), (row[0], got)


def test_frame_result_motion_matches_fictrac_columns():
    """`FrameResult.forward`/`side`/`turn` are the increments of FicTrac's own columns.

    Summed, they give the integrated forward and side motion (columns 20-21) and the
    heading (column 17), on the same excerpt as above.
    """
    rows = read_dat(Path(__file__).parent / "data" / "fictrac-003-head.dat")
    frames = [
        FrameResult(int(r[0]), int(r[22]), r[21], r[1:4], r[5:8], None, None, None, r)
        for r in rows
    ]
    forward = np.array([f.forward for f in frames])
    side = np.array([f.side for f in frames])
    turn = np.array([f.turn for f in frames])
    assert np.allclose(np.cumsum(forward), rows[:, 19], rtol=0, atol=1e-12)
    assert np.allclose(np.cumsum(side), rows[:, 20], rtol=0, atol=1e-12)
    wrapped = np.angle(np.exp(1j * (np.cumsum(turn) - rows[:, 16])))
    assert np.allclose(wrapped, 0, atol=1e-12)
    assert np.allclose(np.hypot(forward, side), rows[:, 18], rtol=0, atol=1e-12)
    assert [f.x for f in frames] == list(rows[:, 14])
    assert [f.heading for f in frames] == list(rows[:, 16])
