"""Multi-object tracker interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Optional

from btrv.core.types import Detection


@dataclass
class TrackedDetection:
    detection: Detection
    track_id: Optional[int]


@dataclass
class TrackerConfig:
    backend: str = "bytetrack"
    track_activation_threshold: float = 0.3  # detections above this may start/continue tracks (Workflow draft: detector conf 0.3)
    lost_track_buffer: int = 30  # frames (at frame_rate) a lost track is kept before its id is retired
    minimum_matching_threshold: float = 0.8  # IoU-distance threshold as in the Workflow block (IoU >= 1 - value)
    minimum_consecutive_frames: int = 1  # Workflow draft uses 1
    frame_rate: float = 30.0  # nominal rate used only to convert lost_track_buffer to seconds
    high_conf_det_threshold: Optional[float] = None  # ByteTrack first-stage threshold; None = same as track_activation_threshold (supervision/Workflow behaviour)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class Tracker(ABC):
    name = "tracker"

    @abstractmethod
    def update(self, detections: list[Detection], frame_index: int, timestamp_s: float) -> list[TrackedDetection]: ...

    @abstractmethod
    def reset(self) -> None: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def create_tracker(cfg: TrackerConfig) -> Tracker:
    if cfg.backend == "bytetrack":
        from .bytetrack import ByteTrackTracker

        return ByteTrackTracker(cfg)
    if cfg.backend == "none":
        return NullTracker()
    raise ValueError(f"unknown tracker backend {cfg.backend!r}")


class NullTracker(Tracker):
    name = "none"

    def update(self, detections, frame_index, timestamp_s):
        return [TrackedDetection(d, None) for d in detections]

    def reset(self):
        pass
