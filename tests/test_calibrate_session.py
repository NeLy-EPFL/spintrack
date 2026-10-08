import numpy as np

from spintrack.calibrate.session import CalibrationSession
from spintrack.calibrate.square import SQUARE_CORNERS
from spintrack.camera import PinholeCamera
from spintrack.config import Config
from spintrack.geometry import normalize, rotation_angle, rotvec_to_matrix
from spintrack.sphere import ball_outline

CAM = PinholeCamera(320, 240, 45.0)
CENTER = normalize(np.array([0.05, -0.02, 1.0]))
HALF = 0.2


def test_session_fits_ball_and_writes_config(tmp_path):
    cfg = Config(camera={"vfov_deg": 45.0})
    session = CalibrationSession(
        cfg, np.zeros((240, 320), np.uint8), tmp_path / "config.toml"
    )
    rim = ball_outline(CAM, CENTER, HALF, n_points=6)
    session.circle_points = [tuple(p) for p in rim]
    assert session.fit_circle()
    assert (
        np.arccos(np.clip(session.center @ CENTER, -1, 1)) < 1e-6
        and abs(session.half_angle - HALF) < 1e-6
    )
    session.current_polygon = [(10, 10), (30, 10), (30, 30)]
    assert session.close_polygon() and session.current_polygon == []
    R = rotvec_to_matrix([0.2, -0.1, 0.05])
    corners = SQUARE_CORNERS["yz"] @ R.T + np.array([0.1, 0.0, 3.0])
    x, y, _ = CAM.project(corners)
    session.square_plane = "yz"
    session.square_points = list(zip(x.tolist(), y.tolist()))
    assert session.solve_square()
    assert rotation_angle(session.cam_to_lab @ R) < 1e-6  # cam_to_lab == R^T
    path = session.save()
    again = Config.load(path)
    assert len(again.ball.rim) == 6
    assert again.mask.ignore == [[(10, 10), (30, 10), (30, 30)]]
    assert again.camera.position_deg is None
    assert rotation_angle(rotvec_to_matrix(again.camera.rotation) @ R) < 1e-6
    overlay = session.overlay()
    assert overlay.shape == (240, 320, 3) and overlay.max() > 0
    assert abs(session.cursor_angle(x=1000.0, y=CAM.project(CENTER)[1])) < 1.0


def test_session_angles_mode_writes_the_camera_position(tmp_path):
    rim = ball_outline(CAM, CENTER, HALF, n_points=6).tolist()
    cfg = Config(camera={"vfov_deg": 45.0, "rotation": (0.1, 0, 0)}, ball={"rim": rim})
    session = CalibrationSession(
        cfg, np.zeros((240, 320), np.uint8), tmp_path / "c.toml"
    )
    assert np.arccos(np.clip(session.center @ CENTER, -1, 1)) < 1e-6
    session.set_angles(30.0, 10.0, 0.0)
    out = Config.load(session.save())
    assert out.camera.position_deg == (30.0, 10.0, 0.0) and out.camera.rotation is None
    assert np.allclose(out.ball.rim, rim)
