import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from btrv.adapter.server import DataHub, create_app
from btrv.io.heatmap import HeatmapMeta
from btrv.storage import BoxesTable
from tests.test_io import make_frames


@pytest.fixture
def client():
    hub = DataHub(source_id="vid", image_size=(1920, 1080), frame_count=100)
    hub.tracking = BoxesTable().extend(make_frames(5))
    hub.raw = BoxesTable().extend(make_frames(5))
    app = create_app(hub)
    return TestClient(app), hub


def test_info_and_tracking_pull(client):
    c, hub = client
    r = c.get("/api/v1/info")
    assert r.status_code == 200 and r.json()["source_id"] == "vid" and r.json()["tracking_frames"] == 5
    r = c.get("/api/v1/tracking/frames/2")
    assert r.status_code == 200 and r.json()["frame_index"] == 2 and r.json()["schema"] == "btrv.boxes"
    assert c.get("/api/v1/tracking/frames/99").status_code == 404
    assert len(c.get("/api/v1/tracking/frames", params={"start": 1, "stop": 4}).json()) == 3
    assert c.get("/api/v1/tracking/raw/0").status_code == 200
    assert c.get("/api/v1/schemas/boxes").json()["title"].startswith("btrv.boxes")
    assert c.get("/api/v1/schemas/nope").status_code == 404
    assert c.get("/api/v1/calibration").status_code == 404


def test_push_boxes_validation(client):
    c, hub = client
    rec = make_frames(1)[0].to_json()
    r = c.post("/api/v1/boxes", json=rec)
    assert r.status_code == 200 and r.json()["accepted_frames"] == 1
    assert hub.external_boxes.has_frame(0)
    r = c.post("/api/v1/boxes", json=[make_frames(3)[1].to_json(), make_frames(3)[2].to_json()])
    assert r.json()["accepted_frames"] == 2
    bad = dict(rec)
    bad["source_id"] = "other"
    r = c.post("/api/v1/boxes", json=bad)
    assert r.status_code == 422 and "source_id" in r.json()["detail"]
    bad = dict(rec)
    bad["frame_index"] = 1000
    assert "beyond" in c.post("/api/v1/boxes", json=bad).json()["detail"]
    bad = json.loads(json.dumps(rec))
    bad["boxes"][0]["xyxy"] = "nope"
    assert c.post("/api/v1/boxes", json=bad).status_code == 422
    assert c.post("/api/v1/boxes", json="garbage").status_code == 422
    assert hub.version >= 2


def test_push_ball_heatmap_sportvu(client):
    c, hub = client
    r = c.post("/api/v1/ball", json={"schema": "btrv.ball", "schema_version": "1.0", "source_id": "vid", "frame_index": 3, "timestamp_s": 0.12, "x": 10, "y": 20})
    assert r.status_code == 200 and hub.ball[3].x == 10
    assert c.post("/api/v1/heatmap/frame", json={"frame_index": 0, "grid": [[1.0]]}).status_code == 422
    meta = HeatmapMeta("vid", "court_feet", [0, 0, 94, 50], 4, 2, normalization="max_one").to_json()
    assert c.post("/api/v1/heatmap/meta", json=meta).status_code == 200
    g = np.zeros((2, 4)); g[1, 2] = 1.0
    assert c.post("/api/v1/heatmap/frame", json={"frame_index": 7, "timestamp_s": 0.28, "grid": g.tolist()}).status_code == 200
    assert hub.live_heatmap.has_frame(7) and hub.live_heatmap.grid_for_frame(7)[1, 2] == 1.0
    assert c.post("/api/v1/heatmap/frame", json={"frame_index": 8, "grid": [[2.0, 0, 0, 0]]}).status_code == 422
    assert c.post("/api/v1/heatmap/frame", json={"frame_index": 8, "grid": [[0.5, 0, 0, 0, 0]]}).status_code == 422
    bad_meta = dict(meta); bad_meta["source_id"] = "x"
    assert c.post("/api/v1/heatmap/meta", json=bad_meta).status_code == 422
    assert c.post("/api/v1/sportvu", json={"events": []}).status_code == 422
    assert c.post("/api/v1/sportvu", json={"gameid": "g", "gamedate": "", "events": [{"moments": [[0, 0, 0, 0, None, [[-1, -1, 1, 1, 0]]]]}]}).json()["moments"] == 1


def test_websocket_roundtrip(client):
    c, hub = client
    with c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "boxes", "data": make_frames(1)[0].to_json()}))
        assert ws.receive_json()["accepted"] == 1
        ws.send_text(json.dumps({"type": "get_tracking", "frame_index": 1}))
        msg = ws.receive_json()
        assert msg["type"] == "tracking" and msg["data"]["frame_index"] == 1
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "boxes", "data": {"schema": "nope"}}))
        assert "schema" in ws.receive_json()["detail"]
        ws.send_text(json.dumps({"type": "get_state"}))
        assert ws.receive_json()["type"] == "state"
