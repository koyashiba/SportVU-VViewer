"""ByteTrack via Roboflow's ``trackers`` package (the implementation behind the
Workflow ByteTrack block).  Decoded presentation timestamps are passed to the
tracker so motion prediction uses real frame times, not an assumed rate."""
from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as _ByteTrack

from btrv.core.types import Detection
from .base import TrackedDetection, Tracker, TrackerConfig


class ByteTrackTracker(Tracker):
    name = "bytetrack"

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._make()

    def _make(self) -> None:
        self._bt = _ByteTrack(
            lost_track_buffer=self.cfg.lost_track_buffer,
            frame_rate=self.cfg.frame_rate,
            track_activation_threshold=self.cfg.track_activation_threshold,
            minimum_consecutive_frames=self.cfg.minimum_consecutive_frames,
            minimum_iou_threshold=max(0.05, 1.0 - self.cfg.minimum_matching_threshold),
            high_conf_det_threshold=self.cfg.high_conf_det_threshold if self.cfg.high_conf_det_threshold is not None else self.cfg.track_activation_threshold,
        )
        self._t0: float | None = None

    def reset(self) -> None:
        self._make()

    def update(self, detections: list[Detection], frame_index: int, timestamp_s: float) -> list[TrackedDetection]:
        if self._t0 is None:
            self._t0 = timestamp_s
        t = max(0.0, timestamp_s - self._t0)
        if not detections:
            self._bt.update(sv.Detections.empty(), timestamp=t)
            return []
        xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array([d.class_id for d in detections], dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, data={"orig": np.arange(len(detections))})
        tracked = self._bt.update(det, timestamp=t)
        out: list[TrackedDetection] = []
        matched: set[int] = set()
        n = len(tracked)
        if n:
            origs = tracked.data.get("orig") if tracked.data else None
            for j in range(n):
                tid = int(tracked.tracker_id[j]) if tracked.tracker_id is not None else -1
                if origs is not None:
                    oi = int(origs[j])
                else:  # fall back to exact box match
                    oi = int(np.argmin(np.abs(xyxy - tracked.xyxy[j]).sum(axis=1)))
                if oi in matched:
                    continue
                matched.add(oi)
                out.append(TrackedDetection(detections[oi], None if tid < 0 else tid))
        for i, d in enumerate(detections):
            if i not in matched:
                out.append(TrackedDetection(d, None))
        return out

    def describe(self) -> dict[str, Any]:
        try:
            ver = metadata.version("trackers")
        except metadata.PackageNotFoundError:  # pragma: no cover
            ver = None
        return {"name": self.name, "library": "trackers", "library_version": ver, "timestamps": "decoded pts", **self.cfg.to_json()}
