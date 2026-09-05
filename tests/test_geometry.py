import numpy as np

from spintrack.geometry import (
    matrix_to_rotvec,
    rotation_angle,
    rotation_between,
    rotvec_to_matrix,
    skew,
)


def test_rotvec_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        axis = rng.normal(size=3)
        w = axis / np.linalg.norm(axis) * rng.uniform(1e-9, np.pi - 1e-3)
        R = rotvec_to_matrix(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(matrix_to_rotvec(R), w, atol=1e-9)


def test_small_angle_and_near_pi():
    w = np.array([1e-10, -2e-10, 3e-10])
    assert np.allclose(matrix_to_rotvec(rotvec_to_matrix(w)), w, atol=1e-16)
    axis = np.array([0.6, 0.0, 0.8])
    w = axis * (np.pi - 1e-7)
    assert np.allclose(matrix_to_rotvec(rotvec_to_matrix(w)), w, atol=1e-6)


def test_matrix_rotates_about_axis_with_right_hand_rule():
    R = rotvec_to_matrix([0.0, 0.0, np.pi / 2])
    assert np.allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(
        skew([1.0, 2.0, 3.0]) @ [4.0, 5.0, 6.0], np.cross([1, 2, 3], [4, 5, 6])
    )


def test_rotation_between_maps_a_to_b():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        R = rotation_between(a, b)
        assert np.allclose(R @ a, b, atol=1e-12)
    assert np.allclose(rotation_between(a, a), np.eye(3))
    R = rotation_between(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-12)
    assert np.isclose(rotation_angle(R), np.pi)
