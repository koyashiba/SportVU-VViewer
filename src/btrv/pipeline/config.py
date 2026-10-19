from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from btrv.detect.base import DetectorConfig
from btrv.team.color import TeamConfig
from btrv.track.base import TrackerConfig


@dataclass
class ProcessingConfig:
    video_path: str
    run_dir: str
    source_id: Optional[str] = None
    camera_mode: str = "fixed"  # fixed | moving
    roi: Optional[tuple[int, int, int, int]] = None  # fixed mode only, source pixels (x0, y0, x1, y1)
    frame_start: int = 0
    frame_stop: Optional[int] = None  # exclusive
    frame_step: int = 1
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    calibration_path: Optional[str] = None  # btrv.calibration file (manual/auto reference calibration)
    auto_calibrate: bool = False  # fixed mode: estimate calibration automatically on the first frame if no file
    court_margin_ft: float = 2.0  # detections farther outside the court are marked off-court
    moving_revalidate_every: int = 15
    keep_untracked: bool = True

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["detector"] = self.detector.to_json()
        d["tracker"] = self.tracker.to_json()
        d["team"] = self.team.to_json()
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ProcessingConfig":
        d = dict(d)
        det = d.pop("detector", {})
        det["classes_keep"] = tuple(det.get("classes_keep", ("players",)))
        d["detector"] = DetectorConfig(**det)
        d["tracker"] = TrackerConfig(**d.pop("tracker", {}))
        d["team"] = TeamConfig(**d.pop("team", {}))
        if d.get("roi") is not None:
            d["roi"] = tuple(int(v) for v in d["roi"])
        return ProcessingConfig(**d)
