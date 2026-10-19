"""Court-line evidence extraction used by automatic calibration and refinement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class LineEvidence:
    mask: np.ndarray  # uint8 line mask (255 on candidate line pixels), full source frame size
    dist: np.ndarray  # float32 distance transform (pixels to the nearest line pixel)
    segments: np.ndarray  # [K, 4] x1,y1,x2,y2 Hough segments (source pixels)


def line_mask(frame_bgr: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None, exclude_boxes: Optional[np.ndarray] = None, tophat: int = 15) -> np.ndarray:
    """Bright, thin, low-saturation structures (painted lines) inside the ROI."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat, tophat))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    thr = max(20, int(np.percentile(th, 99.0)))
    mask = ((th >= thr) & (hsv[..., 1] < 110) & (gray > 90)).astype(np.uint8) * 255
    if roi is not None:
        x0, y0, x1, y1 = roi
        m = np.zeros_like(mask)
        m[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        mask = m
    if exclude_boxes is not None:
        for x1_, y1_, x2_, y2_ in np.asarray(exclude_boxes).reshape(-1, 4):
            mask[int(max(0, y1_)) : int(min(h, y2_)), int(max(0, x1_)) : int(min(w, x2_))] = 0
    return mask


def extract_evidence(frame_bgr: np.ndarray, roi=None, exclude_boxes=None, min_segment_px: int = 40) -> LineEvidence:
    mask = line_mask(frame_bgr, roi, exclude_boxes)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3).astype(np.float32)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60, minLineLength=min_segment_px, maxLineGap=12)
    segments = segs.reshape(-1, 4).astype(float) if segs is not None else np.zeros((0, 4))
    return LineEvidence(mask, dist, segments)


def merge_segments(segments: np.ndarray, angle_tol_deg: float = 2.5, dist_tol_px: float = 10.0) -> list[dict]:
    """Group Hough segments into infinite lines (normal form) weighted by total length."""
    lines: list[dict] = []
    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length  # unit normal
        rho = nx * x1 + ny * y1
        if rho < 0:
            nx, ny, rho = -nx, -ny, -rho
        theta = float(np.degrees(np.arctan2(ny, nx)))
        merged = False
        for ln in lines:
            dtheta = abs((ln["theta"] - theta + 180) % 360 - 180)
            if dtheta < angle_tol_deg and abs(ln["rho"] - rho) < dist_tol_px:
                wsum = ln["weight"] + length
                ln["theta"] = (ln["theta"] * ln["weight"] + theta * length) / wsum
                ln["rho"] = (ln["rho"] * ln["weight"] + rho * length) / wsum
                ln["weight"] = wsum
                ln["xs"] += [x1, x2]
                ln["ys"] += [y1, y2]
                merged = True
                break
        if not merged:
            lines.append({"theta": theta, "rho": rho, "weight": length, "xs": [x1, x2], "ys": [y1, y2]})
    for ln in lines:
        t = np.radians(ln["theta"])
        ln["n"] = np.array([np.cos(t), np.sin(t)])
        ln["abc"] = np.array([np.cos(t), np.sin(t), -ln["rho"]])  # ax+by+c=0
    lines.sort(key=lambda d: -d["weight"])
    return lines


def line_score(H_image_to_court: np.ndarray, court_samples_ft: np.ndarray, dist: np.ndarray, roi=None, tol_px: float = 4.0) -> tuple[float, float]:
    """Fraction of projected court-marking samples that land near a line pixel.

    Returns (score, coverage): score = mean(exp(-d/tol)) over samples inside the
    image/ROI; coverage = fraction of samples inside the image/ROI.
    """
    from .homography import project_to_image

    pts, ok = project_to_image(H_image_to_court, court_samples_ft)
    h, w = dist.shape
    if roi is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = roi
    inside = ok & (pts[:, 0] >= x0) & (pts[:, 0] < x1 - 1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1 - 1)
    if inside.sum() < 20:
        return 0.0, float(inside.mean()) if inside.size else 0.0
    xi = pts[inside, 0].astype(int)
    yi = pts[inside, 1].astype(int)
    d = dist[yi, xi]
    return float(np.mean(np.exp(-d / tol_px))), float(inside.mean())
