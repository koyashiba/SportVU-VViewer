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
