"""Local processing worker.

file decoder -> camera mode (ROI for fixed / full frame for moving) -> pinned
player detector -> tracker -> team-colour assignment -> court mapping ->
frame-indexed results on disk (see storage/run_store.py).

The worker never assumes a constant FPS: every record carries the source frame
index and decoded presentation timestamp.  Detections made on an ROI crop or a
downscaled inference image are mapped back to full-frame pixels before
anything is stored.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from btrv.core.compute import resolve_backend
from btrv.core.types import BoxRecord, COURT_CONVENTION, CourtPoint, FrameBoxes, IMAGE_CONVENTION, TeamAssignment
from btrv.court.auto_calibrate import AutoCalibConfig, auto_calibrate
from btrv.court.homography import Calibration, foot_points_to_court
from btrv.court.model import NBA_COURT
from btrv.court.moving import MovingCalibrator, MovingConfig
from btrv.detect.base import InferenceRegion, create_detector
from btrv.source.file_source import FileSource
from btrv.storage.run_store import RunManifest, RunStore
from btrv.team.color import TeamAssigner
from btrv.track.base import create_tracker

from .calibration_file import keyframes_from_document, load_calibration
from .config import ProcessingConfig

ProgressFn = Callable[[int, int, str], None]


@dataclass
class ProcessingResult:
    run_dir: Path
    frames_processed: int
    summary: dict[str, Any]


class _IdSwitchMonitor:
    """Counts track births, deaths and re-identifications to report ID stability honestly."""

    def __init__(self):
        self.first_seen: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}
        self.gaps: list[int] = []
        self.frames_with_tracks = 0
        self.active_per_frame: list[int] = []

    def update(self, frame_ordinal: int, ids: list[int]) -> None:
        self.active_per_frame.append(len(ids))
        if ids:
            self.frames_with_tracks += 1
        for t in ids:
            if t not in self.first_seen:
                self.first_seen[t] = frame_ordinal
            elif frame_ordinal - self.last_seen[t] > 1:
                self.gaps.append(frame_ordinal - self.last_seen[t] - 1)
            self.last_seen[t] = frame_ordinal

    def summary(self, n_frames: int) -> dict[str, Any]:
        lengths = [self.last_seen[t] - self.first_seen[t] + 1 for t in self.first_seen]
        return {
            "tracks_total": len(self.first_seen),
            "track_length_frames": {"mean": float(np.mean(lengths)) if lengths else 0.0, "median": float(np.median(lengths)) if lengths else 0.0, "max": int(max(lengths)) if lengths else 0, "min": int(min(lengths)) if lengths else 0},
            "tracks_spanning_whole_range": int(sum(1 for L in lengths if L >= n_frames - 1)),  # ids appear from a track's 2nd sighting
            "reappearances_after_gap": len(self.gaps),
            "mean_gap_frames": float(np.mean(self.gaps)) if self.gaps else 0.0,
            "active_tracks_per_frame": {"mean": float(np.mean(self.active_per_frame)) if self.active_per_frame else 0.0, "max": int(max(self.active_per_frame)) if self.active_per_frame else 0},
        }


def process_video(cfg: ProcessingConfig, progress: Optional[ProgressFn] = None, cancel: Optional[threading.Event] = None) -> ProcessingResult:
    t0 = time.time()
    store = RunStore(cfg.run_dir)
    src = FileSource(cfg.video_path, source_id=cfg.source_id)
    info = src.info
    W, Hh = info.width, info.height
    stop = info.frame_count if cfg.frame_stop is None else min(cfg.frame_stop, info.frame_count)
    frames = list(range(cfg.frame_start, stop, max(1, cfg.frame_step)))
    if not frames:
        raise ValueError(f"empty frame range [{cfg.frame_start}, {stop}) step {cfg.frame_step}")
    if cfg.camera_mode not in ("fixed", "moving"):
        raise ValueError("camera_mode must be 'fixed' or 'moving'")
    roi = cfg.roi if cfg.camera_mode == "fixed" else None
    region = InferenceRegion.for_frame(W, Hh, roi, cfg.detector.max_inference_side)
    detector = create_detector(cfg.detector)
    tracker = create_tracker(cfg.tracker)
    teams = TeamAssigner(cfg.team)
    compute = resolve_backend(cfg.detector.compute)
    # ---- calibration -------------------------------------------------------
    keyframes: list[Calibration] = []
    cal_doc = None
    if cfg.calibration_path:
        cal_doc = load_calibration(cfg.calibration_path)
        keyframes = keyframes_from_document(cal_doc)
    moving = MovingCalibrator(NBA_COURT, MovingConfig(revalidate_every=cfg.moving_revalidate_every)) if cfg.camera_mode == "moving" else None
    fixed_cal: Optional[Calibration] = keyframes[0] if (keyframes and cfg.camera_mode == "fixed") else None
    run_id = Path(cfg.run_dir).name or uuid.uuid4().hex[:8]
    manifest = RunManifest(
        run_id=run_id,
        source=info.to_json(),
        detector=detector.describe(),
        tracker=tracker.describe(),
        team=cfg.team.to_json(),
        camera_mode=cfg.camera_mode,
        roi=None if roi is None else [int(v) for v in roi],
        inference_region=region.to_json(),
        court=NBA_COURT.to_json(),
        calibration={"file": cfg.calibration_path, "keyframes": [k.to_json() for k in keyframes], "auto_calibrate": cfg.auto_calibrate},
        processing=cfg.to_json(),
        compute={"requested": compute.requested, "resolved": compute.resolved, "reason": compute.reason},
        coordinate_conventions={"image": IMAGE_CONVENTION, "court": COURT_CONVENTION, "sync_keys": "source frame_index (presentation order) + decoded pts timestamp_s"},
    )
    store.write_manifest(manifest)
    store.open_boxes_writer("raw_boxes.jsonl")
    # tracked boxes are written after team assignment (needs all observations) -> keep compact records in memory
    tracked_frames: list[FrameBoxes] = []
    hom_fi, hom_H, hom_conf, hom_valid, hom_method = [], [], [], [], []
    monitor = _IdSwitchMonitor()
    n_det_total = 0
    detector_name = detector.describe().get("name", "detector")
    producer = f"btrv/{detector_name}"
    try:
        for ordinal, fi in enumerate(frames):
            if cancel is not None and cancel.is_set():
                manifest.status = "cancelled"
                break
            finfo, frame = src.read(fi)
            if hasattr(detector, "current_frame_index"):
                detector.current_frame_index = fi
            dets = detector.detect_in_region(frame, region, cfg.detector.classes_keep)
            n_det_total += len(dets)
            raw = FrameBoxes(info.source_id, fi, finfo.timestamp_s, W, Hh, [BoxRecord(d.xyxy, d.class_name, d.class_id, d.confidence) for d in dets], producer=producer + "/raw")
            store.append_frame("raw_boxes.jsonl", raw)
            tracked = tracker.update(dets, fi, finfo.timestamp_s)
            if not cfg.keep_untracked:
                tracked = [t for t in tracked if t.track_id is not None]
            boxes_xyxy = np.array([t.detection.xyxy for t in tracked], dtype=float).reshape(-1, 4)
            # ---- calibration for this frame ----
            if cfg.camera_mode == "fixed":
                if fixed_cal is None and cfg.auto_calibrate:
                    fixed_cal = auto_calibrate(frame, roi, boxes_xyxy, NBA_COURT, AutoCalibConfig(), fi)
                    manifest.calibration["auto_result"] = fixed_cal.to_json()
                cal = fixed_cal
            else:
                manual = None
                for k in keyframes:
                    if k.frame_index == fi:
                        manual = k
                cal = moving.update(frame, fi, boxes_xyxy, manual=manual)
            # ---- court mapping of foot points ----
            feet = np.array([((t.detection.xyxy[0] + t.detection.xyxy[2]) / 2, t.detection.xyxy[3]) for t in tracked], dtype=float).reshape(-1, 2)
            if cal is not None and cal.valid:
                court_xy, court_ok = foot_points_to_court(cal.H, feet, NBA_COURT, cfg.court_margin_ft)
            else:
                court_xy, court_ok = np.full((len(tracked), 2), np.nan), np.zeros(len(tracked), dtype=bool)
            recs: list[BoxRecord] = []
            for j, t in enumerate(tracked):
                teams.observe(frame, t.detection.xyxy, t.track_id, fi, j)
                cp = None
                if cal is not None:
                    cp = CourtPoint(float(court_xy[j, 0]) if np.isfinite(court_xy[j, 0]) else float("nan"), float(court_xy[j, 1]) if np.isfinite(court_xy[j, 1]) else float("nan"), bool(court_ok[j]), cal.confidence if cal.valid else 0.0)
                    if not np.isfinite(court_xy[j]).all():
                        cp = CourtPoint(0.0, 0.0, False, 0.0)
                recs.append(BoxRecord(t.detection.xyxy, t.detection.class_name, t.detection.class_id, t.detection.confidence, t.track_id, None, cp, (float(feet[j, 0]), float(feet[j, 1]))))
            cal_state = None if cal is None else {"valid": bool(cal.valid), "confidence": float(cal.confidence), "method": cal.method}
            tracked_frames.append(FrameBoxes(info.source_id, fi, finfo.timestamp_s, W, Hh, recs, producer=producer + "/tracked", calibration=cal_state))
            if cal is not None:
                hom_fi.append(fi)
                hom_H.append(cal.H)
                hom_conf.append(cal.confidence)
                hom_valid.append(cal.valid)
                hom_method.append(cal.method)
            monitor.update(ordinal, [t.track_id for t in tracked if t.track_id is not None])
            if progress and (ordinal % 5 == 0 or ordinal == len(frames) - 1):
                progress(ordinal + 1, len(frames), f"frame {fi}")
        else:
            manifest.status = "completed"
    finally:
        store.close_writers()
        src.close()
        detector.close()
    # ---- team assignment over all track observations ----
    track_teams = teams.fit()
    untracked_teams = teams.assign_untracked()
    store.open_boxes_writer("tracked_boxes.jsonl")
    for fb in tracked_frames:
        for j, b in enumerate(fb.boxes):
            if b.track_id is not None and b.track_id in track_teams:
                b.team = track_teams[b.track_id]
            elif (fb.frame_index, j) in untracked_teams:
                b.team = untracked_teams[(fb.frame_index, j)]
            else:
                b.team = TeamAssignment(None, 0.0)
        store.append_frame("tracked_boxes.jsonl", fb)
    store.close_writers()
    store.write_json("teams.json", teams.to_json())
    if hom_fi:
        store.write_homographies(np.asarray(hom_fi), np.stack(hom_H), np.asarray(hom_conf), np.asarray(hom_valid), hom_method)
    if cal_doc is not None:
        store.write_json("calibration.json", cal_doc)
    elif fixed_cal is not None:
        from .calibration_file import calibration_document

        store.write_json("calibration.json", calibration_document(info.source_id, (W, Hh), cfg.camera_mode, [fixed_cal], roi, "automatic calibration (orientation may be ambiguous)"))
    n_frames = len(tracked_frames)
    valid_cal = int(sum(1 for v in hom_valid if v))
    summary = {
        "frames_requested": len(frames),
        "frames_processed": n_frames,
        "detections_total": n_det_total,
        "detections_per_frame": n_det_total / max(1, n_frames),
        "tracking": monitor.summary(n_frames),
        "team": {"tracks_assigned": int(sum(1 for a in track_teams.values() if a.team_id is not None)), "tracks_undecided": int(sum(1 for a in track_teams.values() if a.team_id is None)), "centroid_separation": teams.separation()},
        "calibration": {"frames_with_valid_calibration": valid_cal, "frames_without": n_frames - valid_cal, "mean_confidence": float(np.mean(hom_conf)) if hom_conf else None},
        "elapsed_s": time.time() - t0,
        "status": manifest.status,
    }
    manifest.extra["summary"] = summary
    manifest.detector = detector.describe()  # refresh: hosted backend learns the served model id on first use
    store.write_manifest(manifest)
    store.write_json("summary.json", summary)
    return ProcessingResult(Path(cfg.run_dir), n_frames, summary)
