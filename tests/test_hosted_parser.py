"""The exact response shape returned by Roboflow hosted inference for the pinned model
(captured via the Roboflow connector on 2026-10-19 for a 1920x1080 dataset image)."""
from btrv.detect.roboflow_hosted import RoboflowHostedDetector

CAPTURED = {
    "model_id": "bachelor-nozvk/basketball-player-detection-jwk5o-t33zb-2-rfdetr-small-t1",
    "image": {"width": 1920, "height": 1080},
    "predictions": [
        {"x": 1382, "y": 864, "width": 144, "height": 356, "confidence": 0.9445492625236511, "class": "players", "class_id": 0},
        {"x": 1210.5, "y": 735.5, "width": 171, "height": 263, "confidence": 0.9423415064811707, "class": "players", "class_id": 0},
        {"x": 1047.5, "y": 506.5, "width": 125, "height": 219, "confidence": 0.9355148077011108, "class": "players", "class_id": 0},
    ],
}


def test_parse_hosted_predictions_centre_based_to_xyxy():
    dets = RoboflowHostedDetector.parse_predictions(CAPTURED, 1920, 1080)
    assert len(dets) == 3
    d = dets[0]
    assert d.class_name == "players" and d.class_id == 0
    assert d.xyxy == (1310.0, 686.0, 1454.0, 1042.0)
    assert dets[1].xyxy == (1210.5 - 85.5, 735.5 - 131.5, 1210.5 + 85.5, 735.5 + 131.5)
    assert abs(d.confidence - 0.9445) < 1e-3


def test_hosted_request_route_and_fallback(monkeypatch):
    """Hosted API v2 route: POST {api_url}/{model_id}?api_key&confidence(percent) with a base64 body; 404 -> fallback id."""
    import numpy as np

    from btrv.detect.base import DetectorConfig

    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    calls = []

    class Resp:
        def __init__(self, status, payload):
            self.status_code = status
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    class Session:
        def post(self, url, params=None, data=None, headers=None, timeout=None):
            calls.append((url, params, headers))
            if url.endswith("/ws/slug"):
                return Resp(404, {})
            return Resp(200, CAPTURED)

    cfg = DetectorConfig(backend="roboflow_hosted", model_id="ws/slug", confidence=0.3, extra={"fallback_model_id": "proj/2", "retries": 1})
    det = RoboflowHostedDetector(cfg, session=Session())
    dets = det.detect(np.zeros((32, 32, 3), np.uint8))
    assert len(dets) == 3
    assert calls[0][0] == "https://serverless.roboflow.com/ws/slug"
    assert calls[1][0] == "https://serverless.roboflow.com/proj/2"
    assert calls[0][1]["api_key"] == "k" and calls[0][1]["confidence"] == 30.0
    assert calls[0][2]["Content-Type"] == "application/x-www-form-urlencoded"
    assert det.served_model_id == "proj/2"
    det.detect(np.zeros((32, 32, 3), np.uint8))
    assert calls[-1][0].endswith("/proj/2") and len(calls) == 3  # served id is remembered
