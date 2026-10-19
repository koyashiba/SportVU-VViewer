"""Test and offline detectors (no model download)."""
from __future__ import annotations

from typing import Any

import numpy as np

from btrv.core.types import Detection
from .base import Detector, DetectorConfig, PLAYER_CLASS


class FakeDetector(Detector):
    """Detects bright saturated blobs (the synthetic 'players' of the test videos).

    It exists so the whole pipeline (ROI transforms, tracker, team assignment,
    court mapping, storage, viewer) can be exercised without model weights.
    """

    name = "fake_blob_detector"

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.min_area = int(cfg.extra.get("min_area", 12))
        self.min_saturation = int(cfg.extra.get("min_saturation", 150))

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] > self.min_saturation) & (hsv[..., 2] > 120)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area or h < w * 0.8:
                continue
            conf = min(1.0, 0.6 + area / 400.0)
            if conf < self.cfg.confidence:
                continue
            out.append(Detection((float(x), float(y), float(x + w), float(y + h)), PLAYER_CLASS, 0, conf))
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": "fake", "confidence": self.cfg.confidence}


class BoxesFileDetector(Detector):
    """Replays raw boxes from a btrv.boxes file (e.g. hosted-inference output) by frame index.

    ``cfg.extra['path']`` is the file; detections are returned in source pixels and
    the caller must use a full-frame InferenceRegion (no crop/scale).
    """

    name = "boxes_file_detector"

    def __init__(self, cfg: DetectorConfig):
        from btrv.io.boxes import read_boxes_file

        self.cfg = cfg
        self.path = cfg.extra["path"]
        self.frames = {fb.frame_index: fb for fb in read_boxes_file(self.path)}
        self.current_frame_index: int | None = None

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        fb = self.frames.get(self.current_frame_index)
        if fb is None:
            return []
        return [Detection(b.xyxy, b.class_name, b.class_id, b.confidence) for b in fb.boxes if b.confidence >= self.cfg.confidence]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": f"file:{self.path}", "confidence": self.cfg.confidence}
