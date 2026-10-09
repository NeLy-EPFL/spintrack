"""A camera from a deeperfly calibration: views, videos, and a run that uses one."""

import numpy as np
import pytest

from helpers import ball_config
from spintrack.autofit import complete_config
from spintrack.calibrate.deeperfly import check_size, read_view
from spintrack.calibrate.sliders import camera_to_lab_from_angles
from spintrack.geometry import matrix_to_rotvec, normalize
from test_autofit import CENTER, write_video


def solved(position, focal, size) -> str:
    """A solved view's table: the camera at `position` (spintrack's angles)."""
    world_to_camera = camera_to_lab_from_angles(*position).T
    rvec = ", ".join(repr(float(v)) for v in matrix_to_rotvec(world_to_camera))
    h, w = size
    return (
        f"rvec = [{rvec}]\ntvec = [0.0, 0.0, 100.0]\n"
        f"intr = [{focal}, {focal}, {(w - 1) / 2}, {(h - 1) / 2}]\n"
        f"image_size = [{h}, {w}]\n"
    )


def project(folder, views: dict[str, str], videos: dict[str, str]):
    """A deeperfly project in `folder`/deeperfly: one calibration, its manifest."""
    root = folder / "deeperfly"
    (root / "calibrations").mkdir(parents=True)
    tables = "".join(f"[views.{name}]\n{body}\n" for name, body in views.items())
    calibration = root / "calibrations" / "0001-bundle-adjustment.toml"
    calibration.write_text(f'solved = true\nlabel = "bundle-adjustment"\n\n{tables}')
    listed = "".join(f'{name} = ["{path}"]\n' for name, path in videos.items())
    (root / "deeperfly.toml").write_text(
        f"active_calibration = 1\n\n[videos]\n{listed}"
    )
    return calibration


def test_orbit_views_sit_where_deeperfly_puts_them(tmp_path):
    """deeperfly's azimuth runs to the animal's left; spintrack's to its right."""
    path = tmp_path / "initial.toml"
    path.write_text(
        'label = "initial"\n\n[views.f]\ndistance = 100\nfocal = 2000\n\n'
        "[views.lf]\ndistance = 100\nfocal = 2000\nazimuth_deg = 45.0\n\n"
        "[views.h]\ndistance = 100\nfocal = 2000\nazimuth_deg = 180.0\n"
        "elevation_deg = 20.0\n"
    )
    for name, position in [("f", (0, 0, 0)), ("LF", (0, -45, 0)), ("h", (20, 180, 0))]:
        view = read_view(path, name)
        assert np.allclose(view.to_animal, camera_to_lab_from_angles(*position))
    with pytest.raises(ValueError, match="camera.view to one of f, lf, h"):
        read_view(path)


def test_the_manifest_says_which_view_filmed_a_video(tmp_path):
    project(
        tmp_path,
        {
            "h": solved((5, 175, 1), 24168.591, (1008, 1600)),
            "f": solved((0, 0, 0), 9e3, (512, 960)),
        },
        {"h": "../camera_H.mp4", "f": "../camera_F.mp4"},
    )
    view = read_view(tmp_path, video=tmp_path / "camera_H.mp4")
    assert view.name == "h" and np.allclose(view.position_deg, (5, 175, 1))
    # A video at half the size has the same field of view.
    assert np.isclose(view.vfov_deg(504), 2.3893, atol=1e-4)
    check_size(view, 800, 504)
    with pytest.raises(ValueError, match="not a scaled copy"):
        check_size(view, 800, 600)


def test_a_run_takes_the_calibration_next_to_its_video(tmp_path):
    """Nothing in the config says where the camera is: the project next to the video
    does, and its focal length gives the field of view."""
    size, half = (320, 240), 0.15
    video = write_video(tmp_path / "ball.mp4", size, CENTER, half, 30)
    focal = 120 / np.tan(np.radians(20))  # a vertical field of view of 40 deg
    project(
        tmp_path, {"h": solved((10, 170, 2), focal, (240, 320))}, {"h": "../ball.mp4"}
    )
    cfg = ball_config(size, CENTER, half)
    cfg.camera.rotation = cfg.camera.vfov_deg = None
    prepared = complete_config(cfg, str(video))
    assert prepared.camera_position == "calibration" and prepared.vfov_from is not None
    assert np.allclose(cfg.camera.position_deg, (10, 170, 2))
    assert np.isclose(cfg.camera.vfov_deg, 40.0)


def test_pose_results_place_the_camera_relative_to_the_fly(tmp_path):
    """A fly turned 10 deg to its left in the rig: the camera behind the rig sits 10
    deg to the fly's right of straight behind it."""
    import json

    import cv2
    import h5py

    positions = {"h": (5, 180, 0), "rm": (0, 90, 0), "lm": (0, -90, 0)}
    size = (512, 960)
    calibration = project(
        tmp_path,
        {name: solved(p, 20000.0, size) for name, p in positions.items()},
        {"h": "../camera_H.mp4"},
    )
    heading = np.radians(10.0)
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])
    coxae = {  # mm, about the rig's origin: front and hind, left and right
        "lf_thorax_coxa": 0.4 * forward + 0.2 * left,
        "rf_thorax_coxa": 0.4 * forward - 0.2 * left,
        "lh_thorax_coxa": -0.3 * forward + 0.25 * left,
        "rh_thorax_coxa": -0.3 * forward - 0.25 * left,
        "neck": 0.6 * forward,
    }
    names = list(coxae)
    world = np.array(list(coxae.values()))
    points = []
    for p in positions.values():
        R = camera_to_lab_from_angles(*p).T  # world to camera
        K = np.array([[20000.0, 0, 479.5], [0, 20000.0, 255.5], [0, 0, 1]])
        px, _ = cv2.projectPoints(
            world, cv2.Rodrigues(R)[0], np.array([0, 0, 100.0]), K, None
        )
        points.append(np.repeat(px.reshape(1, -1, 2), 20, axis=0))  # 20 frames
    results = tmp_path / "deeperfly" / "results"
    results.mkdir()
    with h5py.File(results / "run.h5", "w") as f:
        f["pose2d/points"] = np.array(points)[None].astype(np.float32)
        f["pose2d/conf"] = np.ones((1, 3, 20, len(names)), np.float32)
        f.attrs.update(
            keypoints=json.dumps(names),
            views=json.dumps(list(positions)),
            animals=1,
            written=1,
        )
    view = read_view(calibration, video=tmp_path / "camera_H.mp4")
    assert np.isclose(view.heading_deg, 10.0, atol=0.05)
    el, az, tw = view.position_deg
    assert np.isclose(el, 5, atol=0.05) and np.isclose(tw, 0, atol=0.05)
    # Behind a fly that turned left, the camera is to its right: azimuth 180 + 10.
    assert np.isclose((az - 190 + 180) % 360 - 180, 0, atol=0.05)


def test_a_ball_fitted_to_the_leg_tips_places_the_camera_where_the_fly_stands(
    tmp_path,
):
    """A fly 12 deg behind the ball's top, as on the lab's octacam: the camera behind
    the rig, level in it, sits 12 deg above the fly's horizon."""
    import json

    import cv2
    import h5py

    from spintrack.calibrate.deeperfly import COXAE, TIPS, on_ball

    positions = {"h": (0, 180, 0), "rm": (0, 90, 0), "lm": (0, -90, 0)}
    size, focal, radius = (512, 960), 2000.0, 5.0  # the ball at the rig's origin
    calibration = project(
        tmp_path,
        {name: solved(p, focal, size) for name, p in positions.items()},
        {"h": "../camera_H.mp4"},
    )
    tilt = np.radians(12.0)
    up = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])  # toward the hind camera
    forward, left = np.array([np.cos(tilt), 0.0, np.sin(tilt)]), np.array([0, 1, 0])
    rng = np.random.default_rng(0)
    feet = [
        (0.3, 0.2),
        (0.0, 0.3),
        (-0.3, 0.25),
        (0.3, -0.2),
        (0.0, -0.3),
        (-0.3, -0.25),
    ]
    frames = []
    for _ in range(40):
        tips = [
            radius
            * normalize(up + (a + 0.05 * rng.standard_normal()) * forward + b * left)
            for a, b in feet
        ]
        tips[rng.integers(6)] *= 1.15  # one leg in swing
        bases = [(radius + 0.7) * up + 0.5 * (a * forward + b * left) for a, b in feet]
        frames.append(np.array(tips + bases))
    world = np.array(frames)  # (frames, 12, 3)
    points = []
    for p in positions.values():
        R = camera_to_lab_from_angles(*p).T  # world to camera
        K = np.array([[focal, 0, 479.5], [0, focal, 255.5], [0, 0, 1]])
        px, _ = cv2.projectPoints(
            world.reshape(-1, 3), cv2.Rodrigues(R)[0], np.array([0, 0, 100.0]), K, None
        )
        points.append(px.reshape(len(frames), -1, 2))
    (tmp_path / "deeperfly" / "results").mkdir()
    with h5py.File(tmp_path / "deeperfly" / "results" / "run.h5", "w") as f:
        f["pose2d/points"] = np.array(points)[None].astype(np.float32)
        f["pose2d/conf"] = np.ones((1, 3, len(frames), 12), np.float32)
        f.attrs.update(
            keypoints=json.dumps(list(TIPS + COXAE)),
            views=json.dumps(list(positions)),
            animals=1,
            written=1,
        )
    # The ball's image in view h: centered, its radius from its angular size.
    circle = (480.0, 256.0, focal * radius / np.sqrt(100.0**2 - radius**2))
    view = on_ball(
        read_view(calibration, video=tmp_path / "camera_H.mp4"), circle, size
    )
    assert view.contact is not None, "the fit was refused"
    assert np.isclose(view.contact.tilt_deg, 12.0, atol=0.3)
    assert np.isclose(view.contact.radius, radius, rtol=0.01)
    el, az, tw = view.position_deg
    assert np.isclose(el, 12.0, atol=0.3) and np.isclose(tw, 0.0, atol=0.3)
    assert np.isclose((az - 180 + 180) % 360 - 180, 0.0, atol=0.3)
