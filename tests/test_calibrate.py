import numpy as np

from spintrack.calibrate.sliders import FICTRAC_TO_ANIMAL, camera_to_lab_from_angles
from spintrack.calibrate.square import (
    SQUARE_CORNERS,
    camera_to_lab_from_square,
    square_pose,
)
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


def test_square_gives_the_slider_frame():
    # Each plane seen from the side its corners are clicked TL, TR, BR, BL from (above
    # with the head up, behind, the animal's left), a pose with elevation and twist,
    # and the other sides, whose corners FicTrac has clicked TR, TL, BL, BR. The left
    # and below are exact half turns in FicTrac's frame, where OpenCV's IPPE fails.
    for plane, position, order in [
        ("xy", (90.0, 0.0, 180.0), "TL TR BR BL"),
        ("yz", (0.0, 180.0, 0.0), "TL TR BR BL"),
        ("xz", (20.0, -90.0, 0.0), "TL TR BR BL"),
        ("yz", (30.0, 160.0, 15.0), "TL TR BR BL"),
        ("xz", (10.0, 80.0, -10.0), "TR TL BL BR"),
        ("xy", (-90.0, 0.0, 180.0), "TR TL BL BR"),
    ]:
        R = camera_to_lab_from_angles(*position)
        corners = (SQUARE_CORNERS[plane] @ FICTRAC_TO_ANIMAL + 4.0 * R[:, 2]) @ R
        x, y, valid = CAM.project(corners)
        assert valid.all()
        top, bottom = np.argsort(y)[:2], np.argsort(y)[2:]
        tl, tr = top[np.argsort(x[top])]
        bl, br = bottom[np.argsort(x[bottom])]
        at = {"TL": tl, "TR": tr, "BR": br, "BL": bl}
        clicks = np.stack([x, y], 1)[[at[k] for k in order.split()]]
        R_est = camera_to_lab_from_square(clicks, CAM, plane)
        assert rotation_angle(R_est.T @ R) < 1e-6, (plane, position)


def test_slider_frame_for_a_camera_in_front_of_the_animal():
    R = camera_to_lab_from_angles(0.0, 0.0)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 0, 1], [-1, 0, 0])  # camera looks backwards along -x
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1])  # image down = animal down
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])  # image right = animal left
    # At the animal's right, 30 deg up: it looks left (+y) and down.
    R_side = camera_to_lab_from_angles(30.0, 90.0)
    assert np.isclose((R_side @ [0, 0, 1]) @ [0, 1, 0], np.cos(np.radians(30)))
    R_top = camera_to_lab_from_angles(90.0, 0.0)
    assert np.allclose(R_top @ [0, 0, 1], [0, 0, -1])
