"""Viewer data model: everything is addressed by source frame index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from btrv.adapter.server import DataHub, LiveHeatmap
from btrv.core.types import FrameBoxes
from btrv.court.homography import Calibration
from btrv.io.ball import read_ball_file
from btrv.io.boxes import check_alignment, iter_boxes_file
from btrv.io.errors import ImportValidationError
from btrv.io.heatmap import HeatmapReader
from btrv.io.sportvu import SportVUMoment, SportVUTrack, read_sportvu
from btrv.pipeline.calibration_file import keyframes_from_document, load_calibration
from btrv.source.file_source import FileSource
from btrv.storage import BoxesTable, RunStore


class ViewerSession:
    def __init__(self):
        self.hub = DataHub()
        self.source: Optional[FileSource] = None
        self.timestamps: Optional[np.ndarray] = None
        self.run_store: Optional[RunStore] = None
        self.manifest: Optional[dict[str, Any]] = None
        self.homographies: Optional[dict[str, np.ndarray]] = None
        self._hom_lookup: dict[int, int] = {}
        self.manual_calibration: Optional[Calibration] = None  # set interactively; overrides run calibration for display
        self.roi: Optional[tuple[int, int, int, int]] = None
        self.heatmap_file: Optional[HeatmapReader] = None
        self.heatmap_source: str = "none"  # none | file | live
        self.sportvu: Optional[SportVUTrack] = None
        self.sportvu_epoch_ms: int = 0
        self.sportvu_tolerance_ms: int = 200
        self.current_frame: int = 0
        self.messages: list[str] = []

    # -- video ---------------------------------------------------------------
    def open_video(self, path: str, source_id: Optional[str] = None) -> None:
        self.close_video()
        self.source = FileSource(path, source_id=source_id)
        self.timestamps = self.source.index.timestamps_s()
        info = self.source.info
        self.hub.source_id = info.source_id
        self.hub.image_size = (info.width, info.height)
        self.hub.frame_count = info.frame_count
        self.current_frame = 0

    def close_video(self) -> None:
        if self.source is not None:
            self.source.close()
        self.source = None
        self.timestamps = None

    @property
    def frame_count(self) -> int:
        return 0 if self.source is None else self.source.info.frame_count

    @property
    def image_size(self) -> tuple[int, int]:
        return (0, 0) if self.source is None else (self.source.info.width, self.source.info.height)

    def timestamp(self, i: int) -> float:
        return float(self.timestamps[i]) if self.timestamps is not None and 0 <= i < len(self.timestamps) else 0.0

    def pts(self, i: int) -> Optional[int]:
        return None if self.source is None else int(self.source.index.pts[i])

    # -- run -----------------------------------------------------------------
    def load_run(self, run_dir: str) -> dict[str, Any]:
        store = RunStore(run_dir)
        if not store.exists("manifest.json"):
            raise ImportValidationError(f"{run_dir} has no manifest.json (not a btrv run directory)")
        manifest = store.read_json("manifest.json")
        src = manifest.get("source", {})
        if self.source is None and src.get("uri") and Path(src["uri"]).exists():
            self.open_video(src["uri"], src.get("source_id"))
        if self.source is not None and src.get("source_id") != self.source.info.source_id:
            raise ImportValidationError(f"run was produced for source_id {src.get('source_id')!r}, loaded video is {self.source.info.source_id!r}")
        tracked = BoxesTable().extend(store.iter_boxes("tracked_boxes.jsonl"))
        raw = BoxesTable().extend(store.iter_boxes("raw_boxes.jsonl"))
        self.run_store, self.manifest = store, manifest
        self.hub.tracking, self.hub.raw = tracked, raw
        self.homographies = store.read_homographies()
        self._hom_lookup = {} if self.homographies is None else {int(f): i for i, f in enumerate(self.homographies["frame_index"])}
        self.roi = tuple(manifest["roi"]) if manifest.get("roi") else None
        if store.exists("calibration.json"):
            self.hub.calibration_doc = store.read_json("calibration.json")
        return {"tracked": tracked.stats(), "raw": raw.stats(), "summary": manifest.get("extra", {}).get("summary")}

    # -- per-frame data --------------------------------------------------------
    def local_boxes(self, i: int, raw: bool = False) -> Optional[FrameBoxes]:
        table = self.hub.raw if raw else self.hub.tracking
        return None if table is None else table.frame(i)

    def external_boxes(self, i: int) -> Optional[FrameBoxes]:
        return self.hub.external_boxes.frame(i)

    def calibration_for_frame(self, i: int) -> Optional[Calibration]:
        if self.manual_calibration is not None:
            return self.manual_calibration
        if self.homographies is None:
            return None
        j = self._hom_lookup.get(i)
        if j is None:
            fi = self.homographies["frame_index"]
            if fi.size == 0:
                return None
            k = int(np.searchsorted(fi, i, side="right") - 1)
            if k < 0:
                return None
            if (self.manifest or {}).get("camera_mode") == "moving":
                return None  # no calibration recorded for this frame: do not invent one
            j = k
        H = self.homographies["H"][j]
        return Calibration(H=H, method=str(self.homographies["method"][j]), confidence=float(self.homographies["confidence"][j]), valid=bool(self.homographies["valid"][j]), frame_index=i)

    def heatmap_grid(self, i: int):
        if self.heatmap_source == "file" and self.heatmap_file is not None:
            return self.heatmap_file.grid_for_frame(i), self.heatmap_file.meta
        if self.heatmap_source == "live" and self.hub.live_heatmap is not None:
            return self.hub.live_heatmap.grid_for_frame(i), self.hub.live_heatmap.meta
        return None, None

    def ball_for_frame(self, i: int):
        return self.hub.ball.get(i)

    def sportvu_moment_for_frame(self, i: int) -> Optional[SportVUMoment]:
        track = self.sportvu
        if track is None and self.hub.sportvu_doc is not None:
            self._adopt_live_sportvu()
            track = self.sportvu
        if track is None or self.timestamps is None:
            return None
        t_ms = int(self.sportvu_epoch_ms + round(self.timestamp(i) * 1000))
        return track.moment_for_time(t_ms, self.sportvu_tolerance_ms)

    def _adopt_live_sportvu(self) -> None:
        import tempfile

        doc = self.hub.sportvu_doc
        if doc is None:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            p = f.name
        self.sportvu = read_sportvu(p)
        self.sportvu_epoch_ms = 0

    # -- imports ---------------------------------------------------------------
    def import_external_boxes(self, path: str) -> dict[str, Any]:
        frames = list(iter_boxes_file(path))
        sid = None if self.source is None else self.source.info.source_id
        report = check_alignment(frames, sid, self.image_size if self.source else None, self.frame_count or None, self.timestamps)
        self.hub.external_boxes = BoxesTable().extend(frames)
        self.hub._notify("boxes", {"frames": len(frames)})
        return report

    def import_heatmap(self, path: str) -> dict[str, Any]:
        reader = HeatmapReader(path)
        if self.source is not None:
            report = reader.check_alignment(self.source.info.source_id, self.timestamps, self.frame_count, self.image_size)
        else:
            report = {"frames": len(reader)}
        if self.heatmap_file is not None:
            self.heatmap_file.close()
        self.heatmap_file = reader
        self.heatmap_source = "file"
        return report

    def use_live_heatmap(self) -> None:
        self.heatmap_source = "live"

    def import_ball(self, path: str) -> int:
        recs = read_ball_file(path)
        sid = None if self.source is None else self.source.info.source_id
        for r in recs:
            if sid is not None and r.source_id != sid:
                raise ImportValidationError(f"ball file is for source_id {r.source_id!r}, loaded video is {sid!r}")
        self.hub.ball = {r.frame_index: r for r in recs}
        return len(recs)

    def import_sportvu(self, path: str, epoch_ms: Optional[int] = None) -> int:
        track = read_sportvu(path)
        self.sportvu = track
        if epoch_ms is not None:
            self.sportvu_epoch_ms = epoch_ms
        else:
            # btrv exports record their epoch in the sidecar; otherwise align the first moment with the first frame
            ph = (track.meta or {}).get("placeholders", {})
            txt = str(ph.get("unix_time_ms", ""))
            if "epoch_ms (" in txt:
                try:
                    self.sportvu_epoch_ms = int(txt.split("epoch_ms (")[1].split(")")[0])
                except ValueError:
                    self.sportvu_epoch_ms = 0
            elif track.moments:
                first_ts_ms = int(round(self.timestamp(0) * 1000)) if self.timestamps is not None else 0
                self.sportvu_epoch_ms = int(track.moments[0].unix_ms) - first_ts_ms
        return len(track)

    def import_calibration(self, path: str) -> Calibration:
        doc = load_calibration(path)
        if self.source is not None and doc.get("source_id") not in (None, self.source.info.source_id):
            self.messages.append(f"calibration file is for source_id {doc.get('source_id')!r}")
        ks = keyframes_from_document(doc)
        self.manual_calibration = ks[0]
        if doc.get("roi"):
            self.roi = tuple(doc["roi"])
        self.hub.calibration_doc = doc
        return ks[0]

    def close(self) -> None:
        self.close_video()
        if self.heatmap_file is not None:
            self.heatmap_file.close()
