"""Core value types shared by every module.

Conventions (see docs/coordinates.md):

* ``IMAGE_CONVENTION`` -- boxes are ``[x1, y1, x2, y2]`` floats in *source frame*
  pixels, origin at the top-left corner of the full decoded frame, x to the
  right, y down.  Detections made on an ROI crop or a resized image are always
  transformed back to this space before they are stored or displayed.
* ``COURT_CONVENTION`` -- SportVU-style court feet: x in [0, 94] along the
  court length, y in [0, 50] across the width, origin at the top-left corner of
  the court when drawn in the reference viewer's orientation (y grows "down"
  the drawing).  z is height in feet.
* Synchronisation keys are the *source frame index* (0-based, presentation
  order) and the decoded presentation timestamp in seconds.  A constant FPS is
  never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

BOX_SCHEMA = "btrv.boxes"
BOX_SCHEMA_VERSION = "1.0"
HEATMAP_SCHEMA = "btrv.heatmap"
HEATMAP_SCHEMA_VERSION = "1.0"
BALL_SCHEMA = "btrv.ball"
BALL_SCHEMA_VERSION = "1.0"
IMAGE_CONVENTION = "image_pixels_xyxy_topleft"
COURT_CONVENTION = "sportvu_feet_v1"

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0


@dataclass(frozen=True)
class FrameInfo:
    """Identity of a source frame."""

    frame_index: int
    timestamp_s: float
    width: int
    height: int


@dataclass
class Detection:
    """A raw detector output in source-frame pixels."""

    xyxy: tuple[float, float, float, float]
    class_name: str
    class_id: int
    confidence: float


@dataclass
class CourtPoint:
    x: float
    y: float
    valid: bool = True
    confidence: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "valid": self.valid, "confidence": self.confidence}

    @staticmethod
    def from_json(d: Optional[dict[str, Any]]) -> Optional["CourtPoint"]:
        if d is None:
            return None
        return CourtPoint(float(d["x"]), float(d["y"]), bool(d.get("valid", True)), d.get("confidence"))


@dataclass
class TeamAssignment:
    """Team id 0 or 1 (or None if undecided) with explicit confidence in [0, 1]."""

    team_id: Optional[int]
    confidence: float
    method: str = "jersey_color_kmeans"

    def to_json(self) -> dict[str, Any]:
        return {"id": self.team_id, "confidence": self.confidence, "method": self.method}

    @staticmethod
    def from_json(d: Optional[dict[str, Any]]) -> Optional["TeamAssignment"]:
        if d is None:
            return None
        return TeamAssignment(d.get("id"), float(d.get("confidence", 0.0)), d.get("method", "unknown"))


@dataclass
class BoxRecord:
    """One box of one frame, as stored and exchanged (schema btrv.boxes 1.0)."""

    xyxy: tuple[float, float, float, float]
    class_name: str
    class_id: int
    confidence: float
    track_id: Optional[int] = None
    team: Optional[TeamAssignment] = None
    court: Optional[CourtPoint] = None  # ground-contact point projected to court feet
    foot_xy: Optional[tuple[float, float]] = None  # ground-contact point in image pixels
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "xyxy": [float(v) for v in self.xyxy],
            "class_name": self.class_name,
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "track_id": self.track_id,
            "team": self.team.to_json() if self.team else None,
        }
        if self.court is not None:
            d["court"] = self.court.to_json()
        if self.foot_xy is not None:
            d["foot_xy"] = [float(self.foot_xy[0]), float(self.foot_xy[1])]
        if self.attrs:
            d["attrs"] = self.attrs
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "BoxRecord":
        x = d["xyxy"]
        foot = d.get("foot_xy")
        return BoxRecord(
            xyxy=(float(x[0]), float(x[1]), float(x[2]), float(x[3])),
            class_name=str(d["class_name"]),
            class_id=int(d.get("class_id", 0)),
            confidence=float(d["confidence"]),
            track_id=None if d.get("track_id") is None else int(d["track_id"]),
            team=TeamAssignment.from_json(d.get("team")),
            court=CourtPoint.from_json(d.get("court")),
            foot_xy=None if foot is None else (float(foot[0]), float(foot[1])),
            attrs=dict(d.get("attrs", {})),
        )

    @property
    def foot_point(self) -> tuple[float, float]:
        """Ground-contact point: bottom-centre of the box unless an explicit foot point is set."""
        if self.foot_xy is not None:
            return self.foot_xy
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)


@dataclass
class FrameBoxes:
    """All boxes of one frame plus the synchronisation keys (one JSONL line)."""

    source_id: str
    frame_index: int
    timestamp_s: float
    image_width: int
    image_height: int
    boxes: list[BoxRecord] = field(default_factory=list)
    coordinate_convention: str = IMAGE_CONVENTION
    producer: Optional[str] = None
    calibration: Optional[dict[str, Any]] = None  # {"valid": bool, "confidence": float}

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": BOX_SCHEMA,
            "schema_version": BOX_SCHEMA_VERSION,
            "source_id": self.source_id,
            "frame_index": int(self.frame_index),
            "timestamp_s": float(self.timestamp_s),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "coordinate_convention": self.coordinate_convention,
            "boxes": [b.to_json() for b in self.boxes],
        }
        if self.producer is not None:
            d["producer"] = self.producer
        if self.calibration is not None:
            d["calibration"] = self.calibration
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "FrameBoxes":
        return FrameBoxes(
            source_id=str(d["source_id"]),
            frame_index=int(d["frame_index"]),
            timestamp_s=float(d["timestamp_s"]),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            boxes=[BoxRecord.from_json(b) for b in d.get("boxes", [])],
            coordinate_convention=str(d.get("coordinate_convention", IMAGE_CONVENTION)),
            producer=d.get("producer"),
            calibration=d.get("calibration"),
        )


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
