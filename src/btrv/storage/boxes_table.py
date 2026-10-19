"""Columnar, frame-indexed box storage for fast per-frame lookup without
holding millions of Python objects.

Built from btrv.boxes records (JSONL/JSON or live messages); frames are
retrieved with a binary search over the sorted frame index.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import numpy as np

from btrv.core.types import BoxRecord, CourtPoint, FrameBoxes, IMAGE_CONVENTION, TeamAssignment


class BoxesTable:
    def __init__(self, source_id: str = "", image_width: int = 0, image_height: int = 0, producer: Optional[str] = None):
        self.source_id = source_id
        self.image_width = image_width
        self.image_height = image_height
        self.producer = producer
        self.class_names: list[str] = []
        self._frame_index = np.zeros(0, dtype=np.int64)
        self._timestamp = np.zeros(0, dtype=np.float64)
        self._xyxy = np.zeros((0, 4), dtype=np.float32)
        self._conf = np.zeros(0, dtype=np.float32)
        self._class = np.zeros(0, dtype=np.int32)
        self._track = np.full(0, -1, dtype=np.int64)
        self._team = np.full(0, -1, dtype=np.int8)
        self._team_conf = np.zeros(0, dtype=np.float32)
        self._court = np.full((0, 2), np.nan, dtype=np.float32)
        self._court_valid = np.zeros(0, dtype=bool)
        self._foot = np.full((0, 2), np.nan, dtype=np.float32)
        self._frame_meta: dict[int, tuple[float, Optional[dict]]] = {}  # frame -> (timestamp, calibration)
        self._pending: list[FrameBoxes] = []
        self._sorted = True

    # -- building ----------------------------------------------------------
    def add_frame(self, fb: FrameBoxes) -> None:
        if not self.source_id:
            self.source_id, self.image_width, self.image_height, self.producer = fb.source_id, fb.image_width, fb.image_height, fb.producer
        self._pending.append(fb)
        if len(self._pending) >= 2000:
            self.flush()

    def extend(self, frames: Iterable[FrameBoxes]) -> "BoxesTable":
        for fb in frames:
            self.add_frame(fb)
        self.flush()
        return self

    def flush(self) -> None:
        if not self._pending:
            return
        rows_fi, rows_ts, rows_xyxy, rows_conf, rows_cls, rows_tr, rows_team, rows_tc, rows_court, rows_cv, rows_foot = ([] for _ in range(11))
        for fb in self._pending:
            self._frame_meta[fb.frame_index] = (fb.timestamp_s, fb.calibration)
            for b in fb.boxes:
                if b.class_name not in self.class_names:
                    self.class_names.append(b.class_name)
                rows_fi.append(fb.frame_index)
                rows_ts.append(fb.timestamp_s)
                rows_xyxy.append(b.xyxy)
                rows_conf.append(b.confidence)
                rows_cls.append(self.class_names.index(b.class_name))
                rows_tr.append(-1 if b.track_id is None else b.track_id)
                rows_team.append(-1 if (b.team is None or b.team.team_id is None) else b.team.team_id)
                rows_tc.append(0.0 if b.team is None else b.team.confidence)
                rows_court.append((b.court.x, b.court.y) if b.court is not None else (np.nan, np.nan))
                rows_cv.append(bool(b.court is not None and b.court.valid))
                rows_foot.append(b.foot_xy if b.foot_xy is not None else (np.nan, np.nan))
        self._pending.clear()
        if not rows_fi:
            return
        self._frame_index = np.concatenate([self._frame_index, np.asarray(rows_fi, dtype=np.int64)])
        self._timestamp = np.concatenate([self._timestamp, np.asarray(rows_ts, dtype=np.float64)])
        self._xyxy = np.concatenate([self._xyxy, np.asarray(rows_xyxy, dtype=np.float32).reshape(-1, 4)])
        self._conf = np.concatenate([self._conf, np.asarray(rows_conf, dtype=np.float32)])
        self._class = np.concatenate([self._class, np.asarray(rows_cls, dtype=np.int32)])
        self._track = np.concatenate([self._track, np.asarray(rows_tr, dtype=np.int64)])
        self._team = np.concatenate([self._team, np.asarray(rows_team, dtype=np.int8)])
        self._team_conf = np.concatenate([self._team_conf, np.asarray(rows_tc, dtype=np.float32)])
        self._court = np.concatenate([self._court, np.asarray(rows_court, dtype=np.float32).reshape(-1, 2)])
        self._court_valid = np.concatenate([self._court_valid, np.asarray(rows_cv, dtype=bool)])
        self._foot = np.concatenate([self._foot, np.asarray(rows_foot, dtype=np.float32).reshape(-1, 2)])
        self._sorted = bool(np.all(np.diff(self._frame_index) >= 0))

    def _ensure_sorted(self) -> None:
        self.flush()
        if not self._sorted:
            order = np.argsort(self._frame_index, kind="stable")
            for name in ("_frame_index", "_timestamp", "_xyxy", "_conf", "_class", "_track", "_team", "_team_conf", "_court", "_court_valid", "_foot"):
                setattr(self, name, getattr(self, name)[order])
            self._sorted = True

    # -- queries -------------------------------------------------------------
    def __len__(self) -> int:
        self.flush()
        return int(self._frame_index.shape[0])

    @property
    def frame_indices(self) -> np.ndarray:
        self.flush()
        return np.asarray(sorted(self._frame_meta.keys()), dtype=np.int64)

    def has_frame(self, frame_index: int) -> bool:
        self.flush()
        return frame_index in self._frame_meta

    def frame_range(self) -> tuple[int, int]:
        fi = self.frame_indices
        return (int(fi[0]), int(fi[-1])) if fi.size else (0, -1)

    def _slice(self, frame_index: int) -> slice:
        self._ensure_sorted()
        lo = int(np.searchsorted(self._frame_index, frame_index, side="left"))
        hi = int(np.searchsorted(self._frame_index, frame_index, side="right"))
        return slice(lo, hi)

    def frame(self, frame_index: int) -> Optional[FrameBoxes]:
        """Boxes of exactly this frame (None if the frame was never recorded)."""
        self.flush()
        meta = self._frame_meta.get(frame_index)
        if meta is None:
            return None
        s = self._slice(frame_index)
        boxes = []
        for i in range(s.start, s.stop):
            team = None
            if self._team[i] >= 0 or self._team_conf[i] > 0:
                team = TeamAssignment(None if self._team[i] < 0 else int(self._team[i]), float(self._team_conf[i]))
            court = None
            if np.isfinite(self._court[i]).all() or self._court_valid[i]:
                court = CourtPoint(float(self._court[i, 0]), float(self._court[i, 1]), bool(self._court_valid[i]))
            foot = None if not np.isfinite(self._foot[i]).all() else (float(self._foot[i, 0]), float(self._foot[i, 1]))
            boxes.append(BoxRecord(
                xyxy=tuple(float(v) for v in self._xyxy[i]),
                class_name=self.class_names[int(self._class[i])],
                class_id=int(self._class[i]),
                confidence=float(self._conf[i]),
                track_id=None if self._track[i] < 0 else int(self._track[i]),
                team=team,
                court=court,
                foot_xy=foot,
            ))
        return FrameBoxes(self.source_id, frame_index, meta[0], self.image_width, self.image_height, boxes, IMAGE_CONVENTION, self.producer, meta[1])

    def iter_frames(self) -> Iterator[FrameBoxes]:
        for fi in self.frame_indices:
            fb = self.frame(int(fi))
            if fb is not None:
                yield fb

    def track_ids(self) -> np.ndarray:
        self.flush()
        t = self._track[self._track >= 0]
        return np.unique(t)

    def court_positions(self, frame_index: int) -> list[tuple[int, int, float, float, float]]:
        """[(track_id, team_id, x, y, conf)] for boxes with a valid court point in this frame."""
        if frame_index not in self._frame_meta:
            return []
        s = self._slice(frame_index)
        out = []
        for i in range(s.start, s.stop):
            if self._court_valid[i]:
                out.append((int(self._track[i]), int(self._team[i]), float(self._court[i, 0]), float(self._court[i, 1]), float(self._team_conf[i])))
        return out

    def stats(self) -> dict:
        self.flush()
        return {
            "frames": len(self._frame_meta),
            "boxes": int(self._frame_index.shape[0]),
            "tracks": int(self.track_ids().shape[0]),
            "boxes_with_track": int((self._track >= 0).sum()),
            "boxes_with_team": int((self._team >= 0).sum()),
            "boxes_with_court": int(self._court_valid.sum()),
        }
