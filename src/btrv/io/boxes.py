"""Read/write btrv.boxes 1.0 (JSON Lines or a JSON array of frame records)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from btrv.core.types import BOX_SCHEMA, FrameBoxes, IMAGE_CONVENTION
from .errors import ImportValidationError
from .schemas import validate


def parse_frame_record(obj: dict[str, Any], context: str = "", strict: bool = True) -> FrameBoxes:
    if not isinstance(obj, dict):
        raise ImportValidationError(f"{context}: expected a JSON object per frame, got {type(obj).__name__}")
    if obj.get("schema") != BOX_SCHEMA:
        raise ImportValidationError(f"{context}: 'schema' must be '{BOX_SCHEMA}' (got {obj.get('schema')!r}); see docs/schemas.md")
    ver = str(obj.get("schema_version", ""))
    if not ver.startswith("1."):
        raise ImportValidationError(f"{context}: unsupported schema_version {ver!r}; this build reads btrv.boxes 1.x")
    if obj.get("coordinate_convention", IMAGE_CONVENTION) != IMAGE_CONVENTION:
        raise ImportValidationError(f"{context}: coordinate_convention must be '{IMAGE_CONVENTION}' (boxes in full source-frame pixels)")
    if strict:
        validate(obj, "boxes.schema.json", context)
    fb = FrameBoxes.from_json(obj)
    for i, b in enumerate(fb.boxes):
        x1, y1, x2, y2 = b.xyxy
        if not (x2 > x1 and y2 > y1):
            raise ImportValidationError(f"{context}: box {i} of frame {fb.frame_index} has non-positive size {b.xyxy}")
    return fb


def iter_boxes_file(path: str | os.PathLike, strict: bool = True) -> Iterator[FrameBoxes]:
    path = Path(path)
    if not path.exists():
        raise ImportValidationError(f"boxes file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            try:
                arr = json.load(f)
            except json.JSONDecodeError as e:
                raise ImportValidationError(f"{path.name}: invalid JSON array: {e}") from e
            for i, obj in enumerate(arr):
                yield parse_frame_record(obj, f"{path.name}[{i}]", strict)
            return
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ImportValidationError(f"{path.name}:{ln}: invalid JSON line: {e.msg} (column {e.colno})") from e
            yield parse_frame_record(obj, f"{path.name}:{ln}", strict)


def read_boxes_file(path: str | os.PathLike, strict: bool = True) -> list[FrameBoxes]:
    return list(iter_boxes_file(path, strict))


def write_boxes_file(path: str | os.PathLike, frames: Iterable[FrameBoxes]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for fb in frames:
            f.write(json.dumps(fb.to_json(), separators=(",", ":"), allow_nan=False) + "\n")
            n += 1
    return n


def check_alignment(frames: Iterable[FrameBoxes], source_id: Optional[str], image_size: Optional[tuple[int, int]], frame_count: Optional[int], timestamps: Optional[Any] = None, tol_s: float = 0.02) -> dict[str, Any]:
    """Validate records against the loaded video. Returns a report; raises on hard mismatches."""
    n = 0
    seen: set[int] = set()
    ts_mismatch = 0
    out_of_range = 0
    for fb in frames:
        n += 1
        if source_id is not None and fb.source_id != source_id:
            raise ImportValidationError(f"source_id mismatch: file says {fb.source_id!r} but the loaded video is {source_id!r}. Set the file's source_id to the video's id (or load the matching video).")
        if image_size is not None and (fb.image_width, fb.image_height) != tuple(image_size):
            raise ImportValidationError(f"image size mismatch at frame {fb.frame_index}: file says {fb.image_width}x{fb.image_height}, video is {image_size[0]}x{image_size[1]}. Boxes must be in full source-frame pixels.")
        if fb.frame_index in seen:
            raise ImportValidationError(f"duplicate record for frame {fb.frame_index}")
        seen.add(fb.frame_index)
        if frame_count is not None and fb.frame_index >= frame_count:
            out_of_range += 1
        elif timestamps is not None and abs(float(timestamps[fb.frame_index]) - fb.timestamp_s) > tol_s:
            ts_mismatch += 1
    if out_of_range:
        raise ImportValidationError(f"{out_of_range} record(s) reference frames beyond the video's {frame_count} frames")
    return {"records": n, "timestamp_mismatches": ts_mismatch, "out_of_range": out_of_range}
