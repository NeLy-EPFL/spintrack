import numpy as np

from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.calibrate.square import SQUARE_CORNERS, square_pose
from spintrack.camera import PinholeCamera
from spintrack.geometry import rotation_angle, rotvec_to_matrix

CAM = PinholeCamera(640, 480, 45.0)


def test_square_pose_recovers_synthetic_pose():
    R = rotvec_to_matrix([0.3, -0.2, 0.1])
    t = np.array([0.2, -0.1, 3.0])
    for plane in ("xy", "yz", "xz"):
        corners = SQUARE_CORNERS[plane] @ R.T + t
        x, y, valid = CAM.project(corners)
        assert valid.all()
        R_est, t_est = square_pose(np.stack([x, y], 1), CAM, plane)
        assert rotation_angle(R_est.T @ R) < 1e-6
        assert np.allclose(t_est, t, atol=1e-5)


def test_slider_frame_for_a_camera_in_front_of_the_animal():
    R = camera_to_lab_from_angles(0.0, 0.0)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 0, 1], [-1, 0, 0])  # camera looks backwards along -x
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1])  # image down = animal down
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])  # image right = animal left
    R_side = camera_to_lab_from_angles(30.0, 90.0)
    assert np.isclose((R_side @ [0, 0, 1]) @ [0, -1, 0], np.cos(np.radians(30)))
    R_top = camera_to_lab_from_angles(90.0, 0.0)
    assert np.allclose(R_top @ [0, 0, 1], [0, 0, 1])
