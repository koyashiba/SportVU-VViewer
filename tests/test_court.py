import numpy as np
import pytest

from btrv.court import NBA_COURT, fit_homography, project_to_court, project_to_image, in_court, apply_symmetry, resolve_orientation
from btrv.court.auto_calibrate import auto_calibrate
from btrv.court.homography import project_court_lines, foot_points_to_court
from btrv.court.moving import MovingCalibrator, MovingConfig
from btrv.court.homography import Calibration
from tests.synth import camera_homography, render_court, render_players, default_players, make_scene_video


def test_landmarks_consistent():
    lm = {l.name: (l.x, l.y) for l in NBA_COURT.landmarks}
    assert lm["corner_top_left"] == (0, 0) and lm["corner_bottom_right"] == (94, 50)
    assert lm["left_basket"] == (5.25, 25) and lm["right_basket"] == (88.75, 25)
    assert lm["left_free_throw_top"] == (19, 17)
    assert len(NBA_COURT.polylines()) > 20


def test_manual_fit_recovers_known_homography():
    H = camera_homography(1280, 720)
    names = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left", "half_top", "half_bottom", "left_free_throw_top", "right_free_throw_bottom"]
    court = np.array([[NBA_COURT.landmark(n).x, NBA_COURT.landmark(n).y] for n in names])
    img, ok = project_to_image(H, court)
    rng = np.random.default_rng(1)
    img_noisy = img + rng.normal(0, 0.8, img.shape)
    cal = fit_homography(img_noisy, court, (1280, 720), landmark_names=names, frame_index=3)
    assert cal.valid and cal.confidence > 0.7
    assert cal.reprojection_error_px < 2.0
    pts = np.array([[10, 10], [47, 25], [80, 40]], dtype=float)
    a, _ = project_to_image(H, pts)
    b, _ = project_to_image(cal.H, pts)
    assert np.max(np.linalg.norm(a - b, axis=1)) < 3.0
    assert cal.landmarks[0]["name"] == "corner_top_left"


def test_manual_fit_with_outlier_uses_ransac():
    H = camera_homography(1280, 720)
    names = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left", "half_top", "half_bottom", "center"]
    court = np.array([[NBA_COURT.landmark(n).x, NBA_COURT.landmark(n).y] for n in names])
    img, _ = project_to_image(H, court)
    img[-1] += np.array([120.0, -90.0])  # a badly clicked landmark
    cal = fit_homography(img, court, (1280, 720), landmark_names=names)
    assert cal.valid
    assert not cal.landmarks[-1]["inlier"]
    assert cal.inlier_ratio < 1.0


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_homography(np.zeros((3, 2)), np.zeros((3, 2)), (100, 100))


def test_points_behind_camera_are_invalid():
    H = camera_homography(640, 360)
    # a point far above the horizon maps behind the camera
    xy, ok = project_to_court(H, np.array([[320.0, -5000.0]]))
    assert not ok.all() or not in_court(xy).all()
    assert in_court(np.array([[47.0, 25.0]])).all()
    assert not in_court(np.array([[-5.0, 25.0]])).all()


def test_foot_points_exclude_off_court():
    H = camera_homography(640, 360)
    on, _ = project_to_image(H, np.array([[47.0, 25.0]]))
    off, _ = project_to_image(H, np.array([[47.0, 60.0]]))
    xy, ok = foot_points_to_court(H, np.concatenate([on, off]), margin_ft=2.0)
    assert ok.tolist() == [True, False]
    xy, ok = foot_points_to_court(None, on)
    assert not ok.any()


def test_projected_lines_are_drawable():
    H = camera_homography(640, 360)
    lines = project_court_lines(H, image_size=(640, 360))
    assert len(lines) > 10 and all(l.shape[1] == 2 for l in lines)


@pytest.mark.parametrize("variant", [0, 1])
def test_auto_calibration_synthetic_up_to_symmetry(variant):
    W, Hh = 640, 360
    H = camera_homography(W, Hh, variant)
    img = render_court(H, W, Hh)
    gt = render_players(img, H, default_players(8))
    cal = auto_calibrate(img, exclude_boxes=np.array([g["box"] for g in gt]))
    assert cal.valid and cal.confidence > 0.6, cal.notes
    assert "orientation_ambiguous" in cal.notes
    pts = np.array([[x, y] for x in range(0, 95, 10) for y in range(0, 51, 10)], dtype=float)
    best = min(np.nanmax(np.linalg.norm(project_to_image(apply_symmetry(cal.H, name), pts)[0] - project_to_image(H, pts)[0], axis=1)) for name in ("identity", "flip_ends", "flip_sides", "rotate_180"))
    assert best < 4.0, best
    # with an orientation hint the ambiguity is resolved
    cal2 = auto_calibrate(img, exclude_boxes=np.array([g["box"] for g in gt]), orientation_hint_H=H)
    err = np.nanmax(np.linalg.norm(project_to_image(cal2.H, pts)[0] - project_to_image(H, pts)[0], axis=1))
    assert err < 4.0, err


def test_auto_calibration_without_lines_is_invalid():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cal = auto_calibrate(img)
    assert not cal.valid and cal.confidence == 0.0


def test_refinement_from_perturbed_homography():
    W, Hh = 640, 360
    H = camera_homography(W, Hh, 0)
    img = render_court(H, W, Hh)
    T = np.array([[1, 0, 6.0], [0, 1, -4.0], [0, 0, 1]])
    H0 = H @ np.linalg.inv(T)
    cal = auto_calibrate(img, initial_H=H0)
    pts = np.array([[x, y] for x in range(0, 95, 10) for y in range(0, 51, 10)], dtype=float)
    err_before = np.nanmax(np.linalg.norm(project_to_image(H0, pts)[0] - project_to_image(H, pts)[0], axis=1))
    err_after = np.nanmax(np.linalg.norm(project_to_image(cal.H, pts)[0] - project_to_image(H, pts)[0], axis=1))
    assert cal.valid and err_after < err_before / 2, (err_before, err_after)


def test_moving_camera_propagation(tmp_path):
    path = tmp_path / "moving.mp4"
    Hs, gts, frames = make_scene_video(path, n_frames=12, moving=True)
    mc = MovingCalibrator(cfg=MovingConfig(revalidate_every=5))
    pts = np.array([[x, y] for x in range(10, 85, 15) for y in range(10, 41, 15)], dtype=float)
    cal0 = Calibration(H=Hs[0], method="manual", confidence=1.0, frame_index=0)
    errs = []
    for i, f in enumerate(frames):
        boxes = np.array([g["box"] for g in gts[i]])
        cal = mc.update(f, i, boxes, manual=cal0 if i == 0 else None)
        assert cal.valid, (i, cal.notes)
        err = np.nanmax(np.linalg.norm(project_to_image(cal.H, pts)[0] - project_to_image(Hs[i], pts)[0], axis=1))
        errs.append(err)
    assert max(errs) < 6.0, errs
