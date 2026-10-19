"""Offscreen viewer tests: playback clock, stepping, overlay sync, court view, imports."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, QEventLoop, QTimer  # noqa: E402

from btrv.detect.base import DetectorConfig  # noqa: E402
from btrv.pipeline import ProcessingConfig, process_video  # noqa: E402
from btrv.viewer.playback import SPEEDS, PlaybackController  # noqa: E402
from tests.conftest import marker_index  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_playback_clock_follows_pts_not_fps(qapp):
    clock = FakeClock()
    pc = PlaybackController(clock=clock, use_timer=False)
    # variable frame rate: timestamps are not evenly spaced
    ts = np.array([0.0, 0.04, 0.08, 0.20, 0.24, 0.28, 1.00, 1.04])
    seen = []
    pc.frame_changed.connect(seen.append)
    pc.set_timestamps(ts)
    pc.play()
    for speed in SPEEDS:
        pc.set_speed(speed)
        assert pc.speed == speed
    pc.set_speed(1.0)
    clock.t += 0.05
    pc.tick()
    assert pc.current == 1
    clock.t += 0.05  # t=0.10 -> still frame 2 (next frame at 0.20)
    pc.tick()
    assert pc.current == 2
    clock.t += 0.11  # t=0.21
    pc.tick()
    assert pc.current == 3
    pc.set_speed(4.0)
    clock.t += 0.2  # +0.8s media -> t=1.01 -> frame 6
    pc.tick()
    assert pc.current == 6
    pc.set_speed(0.1)
    clock.t += 0.2  # +0.02 media -> t=1.03 -> frame 6 still
    pc.tick()
    assert pc.current == 6
    clock.t += 0.2  # +0.02 -> t=1.05 >= last -> stops at the last frame
    pc.tick()
    assert pc.current == 7 and not pc.playing
    # stepping while paused moves exactly one frame
    pc.step(-1)
    assert pc.current == 6
    pc.step(-1)
    pc.step(+1)
    assert pc.current == 6
    pc.seek(-5)
    assert pc.current == 0
    pc.seek(99)
    assert pc.current == 7
    with pytest.raises(ValueError):
        pc.set_speed(3.0)


def _wait_for(pred, timeout=5.0):
    loop = QEventLoop()
    t0 = time.time()
    while not pred():
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        if time.time() - t0 > timeout:
            return False
    return True


@pytest.fixture
def window(qapp, marker_video):
    from btrv.viewer.main_window import MainWindow

    w = MainWindow(start_adapter=False)
    w.resize(900, 600)
    w.open_video(str(marker_video))
    assert _wait_for(lambda: w.video.frame_index == 0)
    yield w
    w.close()


def test_stepping_displays_exact_frames(window):
    w = window
    for target in [1, 2, 3, 2, 1, 0, 30, 29, 59]:
        w.playback.seek(target)
        assert _wait_for(lambda: w.video.frame_index == target)
        assert marker_index(w.video.frame) == target
        assert w.court.data.frame_index == target  # court view shows the same frame
        assert f"frame {target} /" in w.frame_lbl.text()
    w.playback.step(-1)
    assert _wait_for(lambda: w.video.frame_index == 58)
    assert marker_index(w.video.frame) == 58
    w.playback.step(+1)
    assert _wait_for(lambda: w.video.frame_index == 59)


def test_play_at_speed_advances_and_pauses_at_end(window):
    w = window
    w.playback.seek(50)
    assert _wait_for(lambda: w.video.frame_index == 50)
    w.speed_box.setCurrentIndex(SPEEDS.index(4.0))
    assert w.playback.speed == 4.0
    w.playback.play()
    assert w.playback.playing
    assert _wait_for(lambda: not w.playback.playing, timeout=6.0)  # 9 frames at 25fps*4x = ~0.09s
    assert w.playback.current == 59


def test_overlays_and_court_sync_with_run(window, tmp_path):
    w = window
    src = w.session.source
    from btrv.court.homography import Calibration
    from btrv.pipeline.calibration_file import calibration_document, save_calibration
    from tests.synth import camera_homography

    cal = Calibration(H=camera_homography(320, 240), method="manual", confidence=1.0, frame_index=0)
    cp = tmp_path / "cal.json"
    save_calibration(cp, calibration_document(src.info.source_id, (320, 240), "fixed", [cal]))
    cfg = ProcessingConfig(video_path=str(src.path), run_dir=str(tmp_path / "run"), detector=DetectorConfig(backend="fake", extra={"min_saturation": 0, "min_area": 10}), calibration_path=str(cp), frame_stop=20)
    process_video(cfg)
    w.load_run(str(tmp_path / "run"))
    w.playback.seek(5)
    assert _wait_for(lambda: w.video.frame_index == 5)
    ov = w.video.overlays
    assert ov.frame_index == 5 and ov.local is not None and ov.local.frame_index == 5
    assert ov.calibration is not None and ov.calibration.valid
    assert w.court.data.calibration_valid
    # toggles
    w.chk["local_boxes"].setChecked(False)
    assert not w.video.layers.local_boxes
    w.chk["court_lines"].setChecked(False)
    assert not w.video.layers.court_lines
    w.opacity.setValue(25)
    assert w.video.layers.heatmap_opacity == pytest.approx(0.25)
    # render offscreen without errors
    w.video.grab()
    w.court.grab()
    # frame 25 has no run data (processed only 20): overlays must be empty, not stale
    w.playback.seek(25)
    assert _wait_for(lambda: w.video.frame_index == 25)
    assert w.video.overlays.local is None


def test_imports_align_or_reject(window, tmp_path):
    w = window
    from btrv.core.types import BoxRecord, CourtPoint, FrameBoxes
    from btrv.io.boxes import write_boxes_file
    from btrv.io.errors import ImportValidationError
    from btrv.io.heatmap import HeatmapMeta, HeatmapWriter

    sid = w.session.source.info.source_id
    frames = [FrameBoxes(sid, i, w.session.timestamp(i), 320, 240, [BoxRecord((10 + i, 20, 40 + i, 80), "players", 0, 0.5, track_id=7, court=CourtPoint(10 + i, 20, True))]) for i in range(10)]
    p = tmp_path / "ext.jsonl"
    write_boxes_file(p, frames)
    rep = w.session.import_external_boxes(str(p))
    assert rep["records"] == 10 and rep["timestamp_mismatches"] == 0
    w.playback.seek(3)
    assert _wait_for(lambda: w.video.frame_index == 3)
    assert w.video.overlays.external.boxes[0].track_id == 7
    assert any(e.kind == "external" for e in w.court.data.entities)
    bad = [FrameBoxes("other", 0, 0.0, 320, 240, [])]
    write_boxes_file(p, bad)
    with pytest.raises(ImportValidationError, match="source_id"):
        w.session.import_external_boxes(str(p))
    hp = tmp_path / "hm.npz"
    with HeatmapWriter(hp, HeatmapMeta(sid, "image_pixels", [0, 0, 320, 240], 32, 24, normalization="max_one", image_width=320, image_height=240)) as hw:
        g = np.zeros((24, 32), np.float32)
        g[12, 16] = 1.0
        hw.append(3, w.session.timestamp(3), g)
    w.session.import_heatmap(str(hp))
    w._refresh_overlays()
    assert w.video.overlays.heatmap_grid is not None
    w.video.grab()
    w.playback.seek(4)
    assert _wait_for(lambda: w.video.frame_index == 4)
    assert w.video.overlays.heatmap_grid is None  # only the current frame's heat map is shown
    # video-space pick -> source pixel mapping is consistent
    w.video.reset_view()
    pt = w.video.to_widget(100, 50)
    assert w.video.to_source(pt) == pytest.approx((100, 50), abs=1e-6)


def test_sportvu_import_feeds_court_view(window, tmp_path):
    w = window
    from btrv.io.sportvu import write_sportvu

    sid = w.session.source.info.source_id
    moments = [[0, int(round(w.session.timestamp(i) * 1000)), 0.0, 0.0, None, [[-1, -1, 47.0, 25.0, 5.0], [1610612737, 11, 10.0 + i, 20.0, 0.0], [1610612738, 22, 60.0, 30.0, 0.0]]] for i in range(60)]
    doc = {"gameid": "t", "gamedate": "", "events": [{"eventId": "1", "home": {"teamid": 1610612737, "players": []}, "visitor": {"teamid": 1610612738, "players": []}, "moments": moments}]}
    path, _ = write_sportvu(tmp_path / "sv.json", doc, {"kind": "test", "placeholders": {"unix_time_ms": "epoch_ms (0) + round(source timestamp_s * 1000)"}})
    assert w.session.import_sportvu(str(path)) == 60
    w.playback.seek(7)
    assert _wait_for(lambda: w.video.frame_index == 7)
    ents = [e for e in w.court.data.entities if e.kind == "sportvu"]
    assert len(ents) == 2 and ents[0].x == pytest.approx(17.0)
    assert any(e.kind == "ball" for e in w.court.data.entities)
