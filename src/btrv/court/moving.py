"""Moving-camera calibration: propagate a calibrated homography frame to frame
and revalidate it periodically against court-line evidence.

H_t (image_t -> court) = H_{t-1} @ G_t^{-1}, where G_t maps image_{t-1} -> image_t
and is estimated from feature matches on the court surface (player boxes are
masked out).  Every ``revalidate_every`` frames, or when propagation fails, the
homography is refined against the line mask; if the refined confidence falls
below ``min_confidence`` the frame is marked invalid and a fresh automatic
calibration is attempted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .auto_calibrate import AutoCalibConfig, auto_calibrate
from .homography import Calibration
from .model import CourtModel, NBA_COURT


@dataclass
class MovingConfig:
    revalidate_every: int = 15
    min_confidence: float = 0.3
    min_matches: int = 25
    max_features: int = 2000
    decay_per_frame: float = 0.985  # confidence decay while propagating without revalidation


class MovingCalibrator:
    def __init__(self, court: CourtModel = NBA_COURT, cfg: MovingConfig = MovingConfig(), auto_cfg: AutoCalibConfig = AutoCalibConfig()):
        self.court = court
        self.cfg = cfg
        self.auto_cfg = auto_cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_desc = None
        self.current: Optional[Calibration] = None
        self._since_revalidation = 0
        self._since_attempt = 0  # frames since the last from-scratch attempt while uncalibrated

    def reset(self, calibration: Optional[Calibration] = None) -> None:
        self._prev_gray = None
        self._prev_kp = self._prev_desc = None
        self.current = calibration
        self._since_revalidation = 0

    def _features(self, gray: np.ndarray, exclude_boxes: Optional[np.ndarray]):
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if exclude_boxes is not None:
            h, w = gray.shape
            for x1, y1, x2, y2 in np.asarray(exclude_boxes).reshape(-1, 4):
                mask[int(max(0, y1)) : int(min(h, y2)), int(max(0, x1)) : int(min(w, x2))] = 0
        return self._orb.detectAndCompute(gray, mask)

    def _frame_to_frame(self, gray: np.ndarray, exclude_boxes) -> Optional[np.ndarray]:
        kp, desc = self._features(gray, exclude_boxes)
        G = None
        if self._prev_desc is not None and desc is not None and len(kp) >= self.cfg.min_matches and len(self._prev_kp) >= self.cfg.min_matches:
            matches = self._matcher.match(self._prev_desc, desc)
            if len(matches) >= self.cfg.min_matches:
                src = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                G, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if G is None or inl is None or inl.sum() < self.cfg.min_matches:
                    G = None
        self._prev_gray, self._prev_kp, self._prev_desc = gray, kp, desc
        return G

    def update(self, frame_bgr: np.ndarray, frame_index: int, exclude_boxes: Optional[np.ndarray] = None, manual: Optional[Calibration] = None) -> Calibration:
        """Return the calibration for this frame (possibly invalid)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if manual is not None:
            self.current = manual
            self._since_revalidation = 0
            self._frame_to_frame(gray, exclude_boxes)
            return manual
        G = self._frame_to_frame(gray, exclude_boxes)
        if self.current is None or not self.current.valid:
            # uncalibrated: a from-scratch search is expensive, so retry only every revalidate_every frames
            if self.current is not None and self._since_attempt < self.cfg.revalidate_every:
                self._since_attempt += 1
                return self.current.invalid_copy("no valid calibration yet (retry pending)")
            self._since_attempt = 0
            hint = self.current.H if self.current is not None else None
            cal = auto_calibrate(frame_bgr, None, exclude_boxes, self.court, self.auto_cfg, frame_index, orientation_hint_H=hint if hint is not None and not np.allclose(hint, np.eye(3)) else None)
            self.current = cal
            self._since_revalidation = 0
            return cal
        if G is None:
            cal = auto_calibrate(frame_bgr, None, exclude_boxes, self.court, self.auto_cfg, frame_index, initial_H=self.current.H)
            if not cal.valid:
                cal = auto_calibrate(frame_bgr, None, exclude_boxes, self.court, self.auto_cfg, frame_index)
            self.current = cal
            self._since_revalidation = 0
            return cal
        try:
            H = self.current.H @ np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return self.current.invalid_copy("singular frame-to-frame homography")
        H /= H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0
        self._since_revalidation += 1
        conf = self.current.confidence * self.cfg.decay_per_frame
        cal = Calibration(H=H, method="propagated", confidence=conf, valid=conf >= self.cfg.min_confidence, frame_index=frame_index, notes="propagated from previous frame")
        if self._since_revalidation >= self.cfg.revalidate_every:
            refined = auto_calibrate(frame_bgr, None, exclude_boxes, self.court, self.auto_cfg, frame_index, initial_H=H)
            self._since_revalidation = 0
            if refined.valid:
                cal = refined
            else:
                fresh = auto_calibrate(frame_bgr, None, exclude_boxes, self.court, self.auto_cfg, frame_index)
                cal = fresh if fresh.valid else refined
        self.current = cal
        return cal
