import json

import numpy as np
import pytest

from btrv.core.types import BoxRecord, CourtPoint, FrameBoxes, TeamAssignment
from btrv.io.ball import BallRecord, read_ball_file, write_ball_file
from btrv.io.boxes import check_alignment, parse_frame_record, read_boxes_file, write_boxes_file
from btrv.io.errors import ExportValidationError, ImportValidationError
from btrv.io.heatmap import HeatmapMeta, HeatmapReader, HeatmapWriter, render_heatmap_overlay
from btrv.io.sportvu import export_sportvu, read_sportvu, write_sportvu, SportVUExportOptions, PLACEHOLDER_TEAM_IDS
from btrv.storage import BoxesTable


def make_frames(n=5, with_court=True):
    out = []
    for i in range(n):
        boxes = []
        for t in range(4):
            x = 100 + 50 * t + i
            court = CourtPoint(10.0 + t * 10 + i * 0.1, 20.0 + t, True) if with_court else None
            boxes.append(BoxRecord((x, 200.0, x + 30, 280.0), "players", 0, 0.9, track_id=t + 1, team=TeamAssignment(t % 2, 0.8), court=court))
        out.append(FrameBoxes("vid", i, i / 25.0, 1920, 1080, boxes, producer="test"))
    return out


# ---------------------------------------------------------------- boxes ----
def test_boxes_jsonl_roundtrip(tmp_path):
    frames = make_frames()
    p = tmp_path / "boxes.jsonl"
    assert write_boxes_file(p, frames) == 5
    back = read_boxes_file(p)
    assert [f.to_json() for f in back] == [f.to_json() for f in frames]
    # JSON array form is also accepted
    p2 = tmp_path / "boxes.json"
    p2.write_text(json.dumps([f.to_json() for f in frames]))
    assert len(read_boxes_file(p2)) == 5


def test_boxes_schema_errors_are_actionable(tmp_path):
    good = make_frames(1)[0].to_json()
    bad = dict(good)
    bad["schema"] = "something"
    with pytest.raises(ImportValidationError, match="schema"):
        parse_frame_record(bad, "x")
    bad = dict(good)
    bad["schema_version"] = "2.0"
    with pytest.raises(ImportValidationError, match="schema_version"):
        parse_frame_record(bad, "x")
    bad = json.loads(json.dumps(good))
    bad["boxes"][0]["xyxy"] = [1, 2, 3]
    with pytest.raises(ImportValidationError, match="xyxy"):
        parse_frame_record(bad, "x")
    bad = json.loads(json.dumps(good))
    bad["boxes"][0]["confidence"] = 1.7
    with pytest.raises(ImportValidationError, match="confidence"):
        parse_frame_record(bad, "x")
    bad = json.loads(json.dumps(good))
    bad["boxes"][0]["xyxy"] = [10, 10, 5, 20]
    with pytest.raises(ImportValidationError, match="non-positive size"):
        parse_frame_record(bad, "x")
    bad = json.loads(json.dumps(good))
    del bad["timestamp_s"]
    with pytest.raises(ImportValidationError, match="timestamp_s"):
        parse_frame_record(bad, "x")
    p = tmp_path / "broken.jsonl"
    p.write_text(json.dumps(good) + "\n{not json\n")
    with pytest.raises(ImportValidationError, match="broken.jsonl:2"):
        read_boxes_file(p)
    with pytest.raises(ImportValidationError, match="not found"):
        read_boxes_file(tmp_path / "missing.jsonl")


def test_boxes_alignment_checks():
    frames = make_frames(3)
    ts = np.arange(100) / 25.0
    rep = check_alignment(frames, "vid", (1920, 1080), 100, ts)
    assert rep == {"records": 3, "timestamp_mismatches": 0, "out_of_range": 0}
    with pytest.raises(ImportValidationError, match="source_id mismatch"):
        check_alignment(frames, "other", (1920, 1080), 100)
    with pytest.raises(ImportValidationError, match="image size mismatch"):
        check_alignment(frames, "vid", (1280, 720), 100)
    with pytest.raises(ImportValidationError, match="beyond"):
        check_alignment(frames, "vid", (1920, 1080), 2)
    with pytest.raises(ImportValidationError, match="duplicate"):
        check_alignment(frames + frames[:1], "vid", (1920, 1080), 100)


# -------------------------------------------------------------- heat map ----
@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_heatmap_roundtrip_and_lookup(tmp_path, suffix):
    meta = HeatmapMeta("vid", "image_pixels", [0, 0, 1920, 1080], 64, 36, image_width=1920, image_height=1080, producer="test", dtype="float16")
    p = tmp_path / f"hm{suffix}"
    rng = np.random.default_rng(0)
    grids = {}
    with HeatmapWriter(p, meta) as w:
        for fi in [0, 2, 3, 10]:
            g = rng.random((36, 64)).astype(np.float32)
            g /= g.sum()
            grids[fi] = g
            w.append(fi, fi / 25.0, g)
    with HeatmapReader(p) as r:
        assert len(r) == 4
        assert r.meta.space == "image_pixels" and r.meta.grid_width == 64
        assert r.has_frame(2) and not r.has_frame(1)
        assert r.grid_for_frame(1) is None
        g = r.grid_for_frame(10)
        assert g.shape == (36, 64) and np.allclose(g, grids[10], atol=1e-3)
        rep = r.check_alignment("vid", np.arange(100) / 25.0, 100, (1920, 1080))
        assert rep["timestamp_mismatches"] == 0
        with pytest.raises(ImportValidationError, match="source_id"):
            r.check_alignment("nope", None, 100)
        with pytest.raises(ImportValidationError, match="references frame"):
            r.check_alignment("vid", None, 5)
        with pytest.raises(ImportValidationError, match="image"):
            r.check_alignment("vid", None, 100, (1280, 720))
        ov = render_heatmap_overlay(g, r.meta, (1920, 1080))
        assert ov.shape == (1080, 1920, 4) and ov[..., 3].max() == 255


def test_heatmap_court_space_overlay(tmp_path):
    from tests.synth import camera_homography

    meta = HeatmapMeta("vid", "court_feet", [0, 0, 94, 50], 94, 50, normalization="max_one")
    p = tmp_path / "hm.npz"
    g = np.zeros((50, 94), np.float32)
    g[25, 47] = 1.0
    with HeatmapWriter(p, meta) as w:
        w.append(0, 0.0, g)
    with HeatmapReader(p) as r:
        H = camera_homography(640, 360)
        ov = render_heatmap_overlay(r.grid_for_frame(0), r.meta, (640, 360), np.linalg.inv(H))
        ys, xs = np.nonzero(ov[..., 3] > 100)
        from btrv.court.homography import project_to_image

        c, _ = project_to_image(H, np.array([[47.5, 25.5]]))
        assert abs(np.mean(xs) - c[0, 0]) < 6 and abs(np.mean(ys) - c[0, 1]) < 6
        with pytest.raises(ValueError):
            render_heatmap_overlay(r.grid_for_frame(0), r.meta, (640, 360))


def test_heatmap_malformed(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, foo=np.zeros(3))
    with pytest.raises(ImportValidationError, match="missing member"):
        HeatmapReader(p)
    meta = HeatmapMeta("vid", "image_pixels", [0, 0, 100, 100], 8, 8)
    np.savez(p, meta=json.dumps(meta.to_json()), frame_index=np.array([0, 1]), timestamp_s=np.array([0.0, 0.04]), grid=np.zeros((2, 4, 8)))
    with pytest.raises(ImportValidationError, match="grid shape"):
        HeatmapReader(p)
    np.savez(p, meta=json.dumps(meta.to_json()), frame_index=np.array([1, 0]), timestamp_s=np.array([0.0, 0.04]), grid=np.zeros((2, 8, 8)))
    with pytest.raises(ImportValidationError, match="strictly increasing"):
        HeatmapReader(p)
    g = np.full((1, 8, 8), 0.5)
    np.savez(p, meta=json.dumps(meta.to_json()), frame_index=np.array([0]), timestamp_s=np.array([0.0]), grid=g)
    with pytest.raises(ImportValidationError, match="sums to"):
        HeatmapReader(p)
    bad_meta = meta.to_json()
    bad_meta["space"] = "metres"
    np.savez(p, meta=json.dumps(bad_meta), frame_index=np.array([0]), timestamp_s=np.array([0.0]), grid=g / g.sum())
    with pytest.raises(ImportValidationError, match="space"):
        HeatmapReader(p)
    with pytest.raises(ImportValidationError, match="not found"):
        HeatmapReader(tmp_path / "nope.h5")
    (tmp_path / "garbage.h5").write_bytes(b"not hdf5")
    with pytest.raises(ImportValidationError, match="HDF5"):
        HeatmapReader(tmp_path / "garbage.h5")


# ------------------------------------------------------------------ ball ----
def test_ball_roundtrip_and_errors(tmp_path):
    recs = [BallRecord("vid", i, i / 25.0, 40.0 + i, 25.0, None if i % 2 else 7.5) for i in range(4)]
    p = tmp_path / "ball.jsonl"
    write_ball_file(p, recs)
    back = read_ball_file(p)
    assert [r.to_json() for r in back] == [r.to_json() for r in recs]
    p.write_text('{"schema": "btrv.ball", "schema_version": "1.0", "source_id": "vid", "frame_index": 0, "timestamp_s": 0}\n')
    with pytest.raises(ImportValidationError, match="'x'"):
        read_ball_file(p)


# --------------------------------------------------------------- sportvu ----
def test_sportvu_conforming_export_structure_and_roundtrip(tmp_path):
    table = BoxesTable().extend(make_frames(5))
    ball = [BallRecord("vid", i, i / 25.0, 47.0, 25.0, 6.0 if i else None) for i in range(5)]
    doc, meta = export_sportvu(table, ball, SportVUExportOptions(game_id="g1", epoch_ms=1000))
    assert set(doc) == {"gameid", "gamedate", "events"}
    ev = doc["events"][0]
    assert set(ev) == {"eventId", "visitor", "home", "moments"}
    assert len(ev["moments"]) == 5
    m0 = ev["moments"][0]
    assert m0[0] == 0 and m0[1] == 1000 and m0[2] == 0.0 and m0[3] == 0.0 and m0[4] is None
    ents = m0[5]
    assert ents[0] == [-1, -1, 47.0, 25.0, 0.0]  # ball first, documented z default when unknown
    assert ev["moments"][1][5][0][4] == 6.0
    assert len(ents) == 5 and all(len(e) == 5 for e in ents)
    team_ids = [e[0] for e in ents[1:]]
    assert team_ids == sorted(team_ids) and set(team_ids) == set(PLACEHOLDER_TEAM_IDS)
    assert ev["moments"][1][1] == 1000 + 40
    assert meta["conforming"] and meta["stats"]["frames"] == 5
    assert {p["playerid"] for p in ev["home"]["players"]} == {1, 3}
    path, meta_path = write_sportvu(tmp_path / "out.json", doc, meta)
    track = read_sportvu(path)
    assert len(track) == 5
    assert track.moments[0].ball == (47.0, 25.0, 0.0)
    assert track.moments[0].players[0][:2] == (PLACEHOLDER_TEAM_IDS[0], 1)
    assert track.moment_for_time(1040).unix_ms == 1040 and track.moment_for_time(999) is None
    assert track.meta["kind"] == "sportvu_conforming"
    # reference-viewer style parse (Moment.py indices)
    for m in ev["moments"]:
        quarter, game_clock, shot_clock = m[0], m[2], m[3]
        ball_e, players = m[5][0], m[5][1:]
        assert ball_e[0] == -1 and all(len(p) == 5 for p in players)


def test_sportvu_export_requires_ball():
    table = BoxesTable().extend(make_frames(3))
    with pytest.raises(ExportValidationError, match="ball coordinates"):
        export_sportvu(table, None)
    with pytest.raises(ExportValidationError, match="missing for 2 of 3 frames"):
        export_sportvu(table, [BallRecord("vid", 0, 0.0, 1, 1)])
    # restricting the range makes it exportable
    doc, _ = export_sportvu(table, [BallRecord("vid", 0, 0.0, 1, 1)], SportVUExportOptions(frame_stop=1))
    assert len(doc["events"][0]["moments"]) == 1


def test_sportvu_player_only_debug_never_synthesises_ball():
    frames = make_frames(3)
    frames[0].boxes[0].team = None
    frames[0].boxes[1].track_id = None
    table = BoxesTable().extend(frames)
    doc, meta = export_sportvu(table, None, player_only_debug=True)
    ents = doc["events"][0]["moments"][0][5]
    assert all(e[0] != -1 for e in ents)
    assert not meta["conforming"] and "NONSTANDARD" in meta["kind"]
    assert ents[-1][0] == 0 or any(e[0] == 0 for e in ents)  # unknown team id 0
    assert any(e[1] == -2 for e in ents)  # untracked player id


def test_sportvu_export_drops_invalid_court_points():
    frames = make_frames(2)
    frames[0].boxes[0].court = CourtPoint(0, 0, False)
    frames[1].boxes[1].court = None
    table = BoxesTable().extend(frames)
    doc, meta = export_sportvu(table, [BallRecord("vid", i, i / 25.0, 1, 1) for i in range(2)])
    assert meta["stats"]["dropped_invalid_court"] == 2
    assert len(doc["events"][0]["moments"][0][5]) == 4


def test_sportvu_import_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{}")
    with pytest.raises(ImportValidationError, match="events"):
        read_sportvu(p)
    p.write_text(json.dumps({"events": [{"moments": [[1, 2, 3]]}]}))
    with pytest.raises(ImportValidationError, match="moment 0"):
        read_sportvu(p)
    p.write_text("[1,")
    with pytest.raises(ImportValidationError, match="invalid JSON"):
        read_sportvu(p)
