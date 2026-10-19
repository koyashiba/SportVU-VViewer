"""Hosted inference against Roboflow's serverless API with the pinned model id.

Uses the inference server's ``/infer/object_detection`` endpoint (the same one
``inference_sdk`` calls).  Needs ROBOFLOW_API_KEY and network access; every
frame is uploaded as JPEG, so this is meant for short excerpts and validation,
not for whole games.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any

import numpy as np
import requests

from btrv.core.types import Detection
from .base import Detector, DetectorConfig


class RoboflowHostedDetector(Detector):
    name = "roboflow_hosted"

    def __init__(self, cfg: DetectorConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"{cfg.api_key_env} is not set; hosted inference needs a Roboflow API key")
        self.session = session or requests.Session()
        self.jpeg_quality = int(cfg.extra.get("jpeg_quality", 92))
        self.retries = int(cfg.extra.get("retries", 3))
        self.endpoint = cfg.api_url.rstrip("/") + "/infer/object_detection"
        self.class_names: list[str] = []

    def _encode(self, image_bgr: np.ndarray) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        payload = {
            "api_key": self.api_key,
            "model_id": self.cfg.model_id,
            "image": {"type": "base64", "value": self._encode(image_bgr)},
            "confidence": self.cfg.confidence,
            "disable_active_learning": True,
        }
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.endpoint, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list):
                    data = data[0]
                return self.parse_predictions(data, image_bgr.shape[1], image_bgr.shape[0])
            except Exception as e:  # pragma: no cover - network
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"hosted inference failed after {self.retries} attempts: {last_err}")

    @staticmethod
    def parse_predictions(data: dict[str, Any], width: int, height: int) -> list[Detection]:
        """Roboflow returns centre-based boxes (x, y, width, height) in the uploaded image's pixels."""
        out: list[Detection] = []
        for p in data.get("predictions", []):
            cx, cy, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            out.append(Detection((x1, y1, x2, y2), str(p.get("class", "")), int(p.get("class_id", 0)), float(p.get("confidence", 0.0))))
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.cfg.model_id, "runtime": "roboflow-serverless", "api_url": self.cfg.api_url, "confidence": self.cfg.confidence}
