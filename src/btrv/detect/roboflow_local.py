"""Local inference with the pinned Roboflow RF-DETR model via ``inference-models``.

Weights are downloaded once (requires ROBOFLOW_API_KEY) and cached by the
library; afterwards inference runs fully offline on CUDA, MPS or CPU.
Install with ``pip install "btrv[local-inference]"``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from btrv.core.compute import resolve_backend
from btrv.core.types import Detection
from .base import Detector, DetectorConfig


class RoboflowLocalDetector(Detector):
    name = "roboflow_local_rfdetr"

    def __init__(self, cfg: DetectorConfig):
        try:
            from inference_models import AutoModel  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Local inference needs the optional dependency 'inference-models' "
                "(pip install 'btrv[local-inference]'); or use backend 'roboflow_hosted'."
            ) from e
        self.cfg = cfg
        self.backend = resolve_backend(cfg.compute)
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{cfg.api_key_env} is not set. The pinned model weights are downloaded once from Roboflow; "
                "put the key into a git-ignored .env or the environment."
            )
        self._model = AutoModel.from_pretrained(cfg.model_id, api_key=api_key, device=self.backend.resolved)
        self.class_names = list(getattr(self._model, "class_names", []))
        self._lib_version = None
        try:
            import inference_models  # type: ignore

            self._lib_version = inference_models.__version__
        except Exception:  # pragma: no cover
            pass

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        from inference_models import ColorFormat  # type: ignore

        preds = self._model.infer(image_bgr, confidence=self.cfg.confidence, input_color_format=ColorFormat.BGR)
        det = preds[0].to_supervision() if hasattr(preds[0], "to_supervision") else preds[0]
        out: list[Detection] = []
        xyxy = np.asarray(det.xyxy, dtype=float)
        conf = np.asarray(det.confidence, dtype=float)
        cls = np.asarray(det.class_id, dtype=int)
        for i in range(xyxy.shape[0]):
            cid = int(cls[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            out.append(Detection(tuple(float(v) for v in xyxy[i]), name, cid, float(conf[i])))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.cfg.model_id,
            "runtime": "inference-models",
            "runtime_version": self._lib_version,
            "device": self.backend.resolved,
            "device_reason": self.backend.reason,
            "confidence": self.cfg.confidence,
            "class_names": self.class_names,
        }
