"""End-to-end vertical slice on a synthetic scene: decode -> detect -> track -> team -> court -> store -> reload."""
import json
from pathlib import Path

import numpy as np
import pytest

from btrv.core.types import FrameBoxes
from btrv.court.homography import Calibration, project_to_court
from btrv.detect.base import DetectorConfig
from btrv.io.boxes import read_boxes_file
from btrv.pipeline import ProcessingConfig, process_video
from btrv.pipeline.calibration_file import calibration_document, save_calibration
from btrv.storage import BoxesTable, RunStore
from tests.synth import make_scene_video


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    path = d / "scene.mp4"
    Hs, gts, frames = make_scene_video(path, n_frames=40, n_players=8)
    cal = Calibration(H=Hs[0], method="manual", confidence=1.0, frame_index=0)
    cal_path = d / "calibration.json"
    save_calibration(cal_path, calibration_document("scene", (640, 360), "fixed", [cal], roi=(0, 40, 640, 360)))
    return {"dir": d, "video": path, "Hs": Hs, "gts": gts, "cal_path": cal_path}


@pytest.fixture(scope="module")
def run(scene):
    cfg = ProcessingConfig(
        video_path=str(scene["video"]),
        run_dir=str(scene["dir"] / "run"),
        camera_mode="fixed",
        roi=(0, 40, 640, 360),
        detector=DetectorConfig(backend="fake", confidence=0.3, max_inference_side=480),
        calibration_path=str(scene["cal_path"]),
    )
    progress = []
    res = process_video(cfg, progress=lambda i, n, msg: progress.append((i, n)))
    assert progress and progress[-1][0] == progress[-1][1] == 40
    return res


def test_run_layout_and_manifest(run):
    store = RunStore(run.run_dir)
    for name in ("manifest.json", "raw_boxes.jsonl", "tracked_boxes.jsonl", "teams.json", "summary.json", "homographies.npz", "calibration.json"):
        assert store.exists(name), name
    m = store.read_manifest()
    assert m.detector["model_id"] == "fake"
    assert m.roi == [0, 40, 640, 360]
    assert m.inference_region["scale"] == pytest.approx(480 / 640)
    assert m.coordinate_conventions["image"] == "image_pixels_xyxy_topleft"
    assert m.software_versions["btrv"]
    assert m.status == "completed"


def test_raw_boxes_in_source_pixels_despite_roi_and_resize(run, scene):
    frames = read_boxes_file(run.run_dir / "raw_boxes.jsonl")
    assert len(frames) == 40
    for fb in frames[:10]:
        gt = scene["gts"][fb.frame_index]
        assert fb.image_width == 640 and fb.image_height == 360
        assert len(fb.boxes) >= 6
        # ground-truth foot points should be within a few pixels of a detected box bottom-centre
        # (overlapping synthetic players merge into one blob, so allow a small miss rate)
        hits = 0
        for g in gt:
            fx, fy = g["foot_px"]
            d = min(np.hypot((b.xyxy[0] + b.xyxy[2]) / 2 - fx, b.xyxy[3] - fy) for b in fb.boxes)
            hits += int(d < 6.0)
        assert hits >= len(gt) - 1, (fb.frame_index, hits, len(gt))


def test_tracker_establishes_and_keeps_ids(run):
    s = run.summary["tracking"]
    assert s["tracks_total"] >= 8
    # with 8 well separated synthetic players over 40 frames most tracks must span the clip
    assert s["tracks_spanning_whole_range"] >= 6, s
    assert s["active_tracks_per_frame"]["mean"] > 7


def test_team_assignment_matches_synthetic_colours(run, scene):
    table = BoxesTable().extend(read_boxes_file(run.run_dir / "tracked_boxes.jsonl"))
    agree = total = 0
    for fi in range(5, 40):
        fb = table.frame(fi)
        for b in fb.boxes:
            if b.team is None or b.team.team_id is None:
                continue
            g = min(scene["gts"][fi], key=lambda g: abs(g["foot_px"][0] - (b.xyxy[0] + b.xyxy[2]) / 2) + abs(g["foot_px"][1] - b.xyxy[3]))
            total += 1
            agree += int(g["team"] == b.team.team_id)
    assert total > 200
    # labels are arbitrary (0/1 swap allowed) but must be consistent
    ratio = agree / total
    assert ratio > 0.95 or ratio < 0.05, ratio


def test_court_positions_close_to_ground_truth(run, scene):
    table = BoxesTable().extend(read_boxes_file(run.run_dir / "tracked_boxes.jsonl"))
    errs = []
    for fi in range(0, 40, 3):
        fb = table.frame(fi)
        assert fb.calibration["valid"]
        for b in fb.boxes:
            assert b.court is not None and b.court.valid
            g = min(scene["gts"][fi], key=lambda g: abs(g["foot_px"][0] - (b.xyxy[0] + b.xyxy[2]) / 2) + abs(g["foot_px"][1] - b.xyxy[3]))
            errs.append(np.hypot(b.court.x - g["court_xy"][0], b.court.y - g["court_xy"][1]))
    assert np.median(errs) < 1.0 and np.max(errs) < 3.0, (np.median(errs), np.max(errs))


def test_boxes_table_roundtrip(run):
    frames = read_boxes_file(run.run_dir / "tracked_boxes.jsonl")
    table = BoxesTable().extend(frames)
    assert table.frame_range() == (0, 39)
    for fb in frames[::7]:
        back = table.frame(fb.frame_index)
        assert back.timestamp_s == fb.timestamp_s
        assert len(back.boxes) == len(fb.boxes)
        for a, b in zip(back.boxes, fb.boxes):
            assert a.track_id == b.track_id and a.xyxy == pytest.approx(b.xyxy, abs=1e-3)
            assert (a.team.team_id if a.team else None) == (b.team.team_id if b.team else None)
    assert table.frame(999) is None
    assert table.court_positions(10)


def test_cancel_marks_run(scene, tmp_path):
    import threading

    ev = threading.Event()
    cfg = ProcessingConfig(video_path=str(scene["video"]), run_dir=str(tmp_path / "run_cancel"), detector=DetectorConfig(backend="fake"), calibration_path=str(scene["cal_path"]))

    def prog(i, n, msg):
        if i >= 5:
            ev.set()

    res = process_video(cfg, progress=prog, cancel=ev)
    assert res.summary["status"] == "cancelled"
    assert 5 <= res.frames_processed < 40
