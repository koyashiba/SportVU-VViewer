"""btrv.ball 1.0: externally supplied ball coordinates in court feet (JSONL or JSON array)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from btrv.core.types import BALL_SCHEMA, BALL_SCHEMA_VERSION, COURT_CONVENTION
from .errors import ImportValidationError
from .schemas import validate


@dataclass
class BallRecord:
    source_id: str
    frame_index: int
    timestamp_s: float
    x: float
    y: float
    z: Optional[float] = None
    confidence: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        return {"schema": BALL_SCHEMA, "schema_version": BALL_SCHEMA_VERSION, "source_id": self.source_id, "frame_index": self.frame_index,
                "timestamp_s": self.timestamp_s, "coordinate_convention": COURT_CONVENTION, "x": self.x, "y": self.y, "z": self.z, "confidence": self.confidence}


def parse_ball_record(obj: dict[str, Any], context: str = "") -> BallRecord:
    if not isinstance(obj, dict) or obj.get("schema") != BALL_SCHEMA:
        raise ImportValidationError(f"{context}: expected a btrv.ball record (schema='{BALL_SCHEMA}')")
    validate(obj, "ball.schema.json", context)
    return BallRecord(str(obj["source_id"]), int(obj["frame_index"]), float(obj["timestamp_s"]), float(obj["x"]), float(obj["y"]),
                      None if obj.get("z") is None else float(obj["z"]), obj.get("confidence"))


def read_ball_file(path: str | os.PathLike) -> list[BallRecord]:
    path = Path(path)
    if not path.exists():
        raise ImportValidationError(f"ball file not found: {path}")
    text = path.read_text(encoding="utf-8")
    out = []
    if text.lstrip().startswith("["):
        for i, obj in enumerate(json.loads(text)):
            out.append(parse_ball_record(obj, f"{path.name}[{i}]"))
    else:
        for ln, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    out.append(parse_ball_record(json.loads(line), f"{path.name}:{ln}"))
                except json.JSONDecodeError as e:
                    raise ImportValidationError(f"{path.name}:{ln}: invalid JSON: {e.msg}") from e
    return out


def write_ball_file(path: str | os.PathLike, records: Iterable[BallRecord]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
            n += 1
    return n
