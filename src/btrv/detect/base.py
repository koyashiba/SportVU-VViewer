"""Detector interface and the crop/resize bookkeeping that maps detections back
to source-frame pixels.

Any detector implementation receives the *inference image* (ROI crop, possibly
resized) and returns boxes in that image's pixels; :class:`InferenceRegion`
transforms them back to the full source frame before anything is stored.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from btrv.core.types import Detection

PINNED_MODEL_ID = "bachelor-nozvk/basketball-player-detection-jwk5o-t33zb-2-rfdetr-small-t1"
PLAYER_CLASS = "players"


@dataclass
class DetectorConfig:
    backend: str = "roboflow_local"  # roboflow_local | roboflow_hosted | fake
    model_id: str = PINNED_MODEL_ID
    confidence: float = 0.3
    compute: str = "auto"  # auto | cuda | mps | cpu
    max_inference_side: Optional[int] = None  # downscale the inference image so its longest side <= this (None = no resize)
    api_key_env: str = "ROBOFLOW_API_KEY"
    api_url: str = "https://serverless.roboflow.com"
    classes_keep: tuple[str, ...] = (PLAYER_CLASS,)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["classes_keep"] = list(self.classes_keep)
        return d


@dataclass(frozen=True)
class InferenceRegion:
    """Describes how the inference image relates to the source frame.

    source pixel = offset + inference pixel / scale
    """

    x0: int
    y0: int
    x1: int
    y1: int
    scale: float = 1.0  # inference_size / crop_size

    @property
    def crop_width(self) -> int:
        return self.x1 - self.x0

    @property
    def crop_height(self) -> int:
        return self.y1 - self.y0

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        crop = frame[self.y0 : self.y1, self.x0 : self.x1]
        if abs(self.scale - 1.0) < 1e-9:
            return np.ascontiguousarray(crop)
        w = max(1, int(round(self.crop_width * self.scale)))
        h = max(1, int(round(self.crop_height * self.scale)))
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)

    def to_source(self, xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        s = 1.0 / self.scale
        return (self.x0 + xyxy[0] * s, self.y0 + xyxy[1] * s, self.x0 + xyxy[2] * s, self.y0 + xyxy[3] * s)

    def to_inference(self, xy: tuple[float, float]) -> tuple[float, float]:
        return ((xy[0] - self.x0) * self.scale, (xy[1] - self.y0) * self.scale)

    @staticmethod
    def for_frame(width: int, height: int, roi: Optional[tuple[int, int, int, int]] = None, max_side: Optional[int] = None) -> "InferenceRegion":
        if roi is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0, x1, y1 = roi
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"empty ROI {roi} for a {width}x{height} frame")
        scale = 1.0
        if max_side:
            longest = max(x1 - x0, y1 - y0)
            if longest > max_side:
                scale = max_side / float(longest)
        return InferenceRegion(x0, y0, x1, y1, scale)

    def to_json(self) -> dict[str, Any]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "scale": self.scale}


class Detector(ABC):
    """Returns detections in the pixels of the image it was given."""

    name: str = "detector"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[Detection]: ...

    def describe(self) -> dict[str, Any]:
        """Exact identity of the model and runtime, stored in the run manifest."""
        return {"name": self.name}

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def detect_in_region(self, frame: np.ndarray, region: InferenceRegion, keep: Optional[tuple[str, ...]] = None) -> list[Detection]:
        """Run on the region's inference image and return source-pixel detections."""
        image = region.prepare(frame)
        out: list[Detection] = []
        for d in self.detect(image):
            if keep and d.class_name not in keep:
                continue
            out.append(Detection(region.to_source(d.xyxy), d.class_name, d.class_id, d.confidence))
        return out


def create_detector(cfg: DetectorConfig) -> Detector:
    if cfg.backend == "fake":
        from .fake import FakeDetector

        return FakeDetector(cfg)
    if cfg.backend == "roboflow_hosted":
        from .roboflow_hosted import RoboflowHostedDetector

        return RoboflowHostedDetector(cfg)
    if cfg.backend == "roboflow_local":
        from .roboflow_local import RoboflowLocalDetector

        return RoboflowLocalDetector(cfg)
    if cfg.backend == "boxes_file":
        from .fake import BoxesFileDetector

        return BoxesFileDetector(cfg)
    raise ValueError(f"unknown detector backend {cfg.backend!r}")
