"""btrv.calibration 1.0 file: reference calibration(s) for a video."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from btrv.court.homography import Calibration
from btrv.court.model import NBA_COURT
from btrv.io.errors import ImportValidationError

CAL_SCHEMA = "btrv.calibration"
CAL_VERSION = "1.0"


def calibration_document(source_id: str, image_size: tuple[int, int], camera_mode: str, keyframes: list[Calibration], roi: Optional[tuple[int, int, int, int]] = None, notes: str = "") -> dict[str, Any]:
    return {
        "schema": CAL_SCHEMA,
        "schema_version": CAL_VERSION,
        "source_id": source_id,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_mode": camera_mode,
        "roi": None if roi is None else [int(v) for v in roi],
        "court": NBA_COURT.to_json(),
        "homography_direction": "image_pixels -> court_feet (x' = H x, homogeneous)",
        "keyframes": [k.to_json() for k in keyframes],
        "notes": notes,
    }


def save_calibration(path: str | os.PathLike, doc: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_calibration(path: str | os.PathLike) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ImportValidationError(f"calibration file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ImportValidationError(f"{path.name}: invalid JSON: {e.msg}") from e
    if doc.get("schema") != CAL_SCHEMA:
        raise ImportValidationError(f"{path.name}: expected schema '{CAL_SCHEMA}'")
    if not isinstance(doc.get("keyframes"), list) or not doc["keyframes"]:
        raise ImportValidationError(f"{path.name}: 'keyframes' must be a non-empty list")
    for i, k in enumerate(doc["keyframes"]):
        try:
            H = np.asarray(k["H"], dtype=float).reshape(3, 3)
        except Exception:
            raise ImportValidationError(f"{path.name}: keyframe {i} has no 3x3 'H'")
        if not np.isfinite(H).all():
            raise ImportValidationError(f"{path.name}: keyframe {i} 'H' contains non-finite values")
    return doc


def keyframes_from_document(doc: dict[str, Any]) -> list[Calibration]:
    ks = [Calibration.from_json(k) for k in doc["keyframes"]]
    ks.sort(key=lambda c: (c.frame_index if c.frame_index is not None else -1))
    return ks
