"""Hosted inference against Roboflow's serverless API with the pinned model id.

Uses the Hosted API v2 route ``POST https://serverless.roboflow.com/{model_id}``
with a base64 JPEG body (``application/x-www-form-urlencoded``) and the query
parameters ``api_key`` and ``confidence`` (percent).  ``model_id`` may be the
pinned model slug (``workspace/model-slug``) or a ``project/version`` id; an
optional ``extra['fallback_model_id']`` is tried when the primary id returns
404.  Needs ROBOFLOW_API_KEY and network access; every frame is uploaded, so
this is meant for short excerpts and validation, not for whole games.
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
        self.overlap_percent = float(cfg.extra.get("overlap_percent", 30))
        self.model_ids = [cfg.model_id] + ([cfg.extra["fallback_model_id"]] if cfg.extra.get("fallback_model_id") else [])
        self.served_model_id: str | None = None
        self.class_names: list[str] = []

    def endpoint(self, model_id: str) -> str:
        return self.cfg.api_url.rstrip("/") + "/" + model_id.strip("/")

    def _encode(self, image_bgr: np.ndarray) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        body = self._encode(image_bgr)
        params = {"api_key": self.api_key, "confidence": round(self.cfg.confidence * 100, 2), "overlap": self.overlap_percent, "format": "json"}
        candidates = [self.served_model_id] if self.served_model_id else self.model_ids
        last_err: Exception | None = None
        for model_id in candidates:
            for attempt in range(self.retries):
                try:
                    r = self.session.post(self.endpoint(model_id), params=params, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=90)
                    if r.status_code == 404:
                        last_err = RuntimeError(f"model id {model_id!r} not found on {self.cfg.api_url} (404)")
                        break  # try the next id
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, list):
                        data = data[0]
                    self.served_model_id = model_id
                    return self.parse_predictions(data, image_bgr.shape[1], image_bgr.shape[0])
                except requests.HTTPError as e:  # pragma: no cover - network
                    last_err = e
                    if e.response is not None and e.response.status_code in (401, 403):
                        raise RuntimeError(f"hosted inference rejected the API key ({e.response.status_code}); check {self.cfg.api_key_env}") from e
                    time.sleep(1.5 * (attempt + 1))
                except Exception as e:  # pragma: no cover - network
                    last_err = e
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"hosted inference failed: {last_err}")

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
        return {"name": self.name, "model_id": self.cfg.model_id, "served_model_id": self.served_model_id, "candidate_model_ids": self.model_ids,
                "runtime": "roboflow-serverless-v2", "api_url": self.cfg.api_url, "confidence": self.cfg.confidence}
