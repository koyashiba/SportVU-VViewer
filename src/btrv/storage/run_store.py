"""On-disk layout of a processing run (everything needed to reproduce and view it).

runs/<run_id>/
  manifest.json          exact model id, software versions, settings, ROI, calibration, conventions
  raw_boxes.jsonl        detector output per frame (btrv.boxes 1.0, no track ids)
  tracked_boxes.jsonl    tracked output per frame with team + court positions (btrv.boxes 1.0)
  calibration.json       reference calibration(s) (btrv.calibration 1.0)
  homographies.npz       per-frame H / confidence / valid for moving cameras (or constant for fixed)
  teams.json             team clustering result
  summary.json           counts, ID-switch statistics, timing
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from btrv.core.types import FrameBoxes
from btrv.core.versions import software_versions

MANIFEST_SCHEMA = "btrv.run_manifest"
MANIFEST_VERSION = "1.0"


@dataclass
class RunManifest:
    run_id: str
    source: dict[str, Any]
    detector: dict[str, Any]
    tracker: dict[str, Any]
    team: dict[str, Any]
    camera_mode: str
    roi: Optional[list[int]]
    inference_region: Optional[dict[str, Any]]
    court: dict[str, Any]
    calibration: dict[str, Any]
    processing: dict[str, Any]
    compute: dict[str, Any]
    coordinate_conventions: dict[str, str]
    software_versions: dict[str, Any] = field(default_factory=software_versions)
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["schema"] = MANIFEST_SCHEMA
        d["schema_version"] = MANIFEST_VERSION
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "RunManifest":
        d = dict(d)
        d.pop("schema", None)
        d.pop("schema_version", None)
        return RunManifest(**d)


class _JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class RunStore:
    def __init__(self, run_dir: str | os.PathLike):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, _JsonlWriter] = {}

    # -- paths -----------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def path(self, name: str) -> Path:
        return self.dir / name

    # -- manifest ------------------------------------------------------------
    def write_manifest(self, manifest: RunManifest) -> None:
        self.write_json("manifest.json", manifest.to_json())

    def read_manifest(self) -> RunManifest:
        return RunManifest.from_json(self.read_json("manifest.json"))

    def write_json(self, name: str, obj: Any) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, self.dir / name)

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    # -- jsonl streams -----------------------------------------------------
    def open_boxes_writer(self, name: str) -> None:
        self._writers[name] = _JsonlWriter(self.dir / name)

    def append_frame(self, name: str, fb: FrameBoxes) -> None:
        self._writers[name].write(fb.to_json())

    def close_writers(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()

    def iter_boxes(self, name: str) -> Iterator[FrameBoxes]:
        p = self.dir / name
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield FrameBoxes.from_json(json.loads(line))

    # -- homographies ----------------------------------------------------------
    def write_homographies(self, frame_index: np.ndarray, H: np.ndarray, confidence: np.ndarray, valid: np.ndarray, method: list[str]) -> None:
        np.savez_compressed(self.dir / "homographies.npz", frame_index=np.asarray(frame_index, dtype=np.int64), H=np.asarray(H, dtype=np.float64).reshape(-1, 3, 3),
                            confidence=np.asarray(confidence, dtype=np.float32), valid=np.asarray(valid, dtype=bool), method=np.asarray(method, dtype="U16"))

    def read_homographies(self) -> Optional[dict[str, np.ndarray]]:
        p = self.dir / "homographies.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()
