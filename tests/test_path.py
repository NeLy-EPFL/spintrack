import numpy as np

from spintrack.path import PathIntegrator, integrate_path


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


def test_turning_while_walking_traces_a_circle():
    n = 628
    turn = -(2 * np.pi / n)  # dr_lab z; heading increases each frame, one lap total
    path = integrate_path(np.tile([0.0, 0.02, turn], (n, 1)))
    radius = 0.02 / (2 * np.pi / n)
    centre = np.array([0.0, radius])
    dist = np.hypot(path[:, 0] - centre[0], path[:, 1] - centre[1])
    assert np.allclose(dist, radius, atol=1e-9)
    assert np.allclose(path[-1, :2], 0.0, atol=1e-6)  # back at the start after one lap
    assert 0.0 <= path[:, 2].max() < 2 * np.pi
