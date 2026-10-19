"""Automatic court calibration from painted-line evidence.

Strategy (hypothesise-and-verify):

1. Extract a line mask and Hough segments, merge them into dominant straight
   lines, and split those into two orientation families ("along" and "across"
   the court as seen in the image).
2. Enumerate hypotheses: two image lines of each family matched to two court
   lines of each family (sidelines / corner-three lines / lane lines, and
   baselines / free-throw lines / half-court line), in both family assignments.
   Each hypothesis gives a homography from four line correspondences (lines
   transform with the inverse transpose), solved in normalised coordinates and
   evaluated in batches.
3. Score hypotheses two-sidedly: how well projected court markings land on line
   pixels (sample score) and how much of the line mask is explained by the
   projected markings (mask score).  The best candidates are refined with a
   score-guarded iterative-closest-point step against the mask.

The confidence is the final two-sided score; low-confidence results are marked
invalid rather than used.  Validated on synthetic renders (tests); behaviour on
real footage is reported separately in docs/verification.md.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .homography import Calibration, project_to_image
from .lines import LineEvidence, extract_evidence, line_score, merge_segments
from .model import CourtModel, NBA_COURT


@dataclass
class AutoCalibConfig:
    max_lines_per_family: int = 5
    min_confidence: float = 0.5
    refine_iterations: int = 10
    icp_max_dist_px: float = 25.0
    sample_spacing_ft: float = 0.75
    search_spacing_ft: float = 2.5
    tol_px: float = 4.0
    top_k: int = 300
    min_area_fraction: float = 0.05  # projected court area must cover at least this fraction of the image/ROI
    batch: int = 4096


def _court_line_families(court: CourtModel):
    W, L = court.width, court.length
    cy, lw = W / 2, court.lane_width / 2
    c3 = court.three_point_corner_from_sideline
    along = [("sideline_top", 0.0), ("three_top", c3), ("lane_top", cy - lw), ("lane_bottom", cy + lw), ("three_bottom", W - c3), ("sideline_bottom", W)]
    across = [("baseline_left", 0.0), ("free_throw_left", court.free_throw_from_baseline), ("half", L / 2), ("free_throw_right", L - court.free_throw_from_baseline), ("baseline_right", L)]
    return along, across


def _split_families(lines: list[dict], max_per: int) -> tuple[list[dict], list[dict]]:
    if not lines:
        return [], []
    ref = lines[0]["theta"]

    def dang(t):
        return abs((t - ref + 90) % 180 - 90)

    fam_a = [ln for ln in lines if dang(ln["theta"]) < 35][:max_per]
    fam_b = [ln for ln in lines if dang(ln["theta"]) >= 35][:max_per]
    return fam_a, fam_b


def homographies_from_line_quads(img_lines: np.ndarray, court_lines: np.ndarray, T_img: np.ndarray, S_court: np.ndarray) -> np.ndarray:
    """Batched DLT. img_lines, court_lines: [N, 4, 3] (ax+by+c=0). Returns H (image->court) [N, 3, 3]
    (rows with NaN where degenerate).

    Points transform x' = H x; lines transform l' = H^{-T} l.  Solve M with
    l_court ~ M l_img in normalised coordinates, then H_n = inv(M)^T and
    H = S^{-1} H_n T.
    """
    Tl = np.linalg.inv(T_img).T  # normalised image line = T^{-T} l
    Sl = np.linalg.inv(S_court).T
    li = img_lines @ Tl.T
    lc = court_lines @ Sl.T
    li = li / np.linalg.norm(li[..., :2], axis=-1, keepdims=True).clip(1e-12)
    lc = lc / np.linalg.norm(lc[..., :2], axis=-1, keepdims=True).clip(1e-12)
    N = li.shape[0]
    A = np.zeros((N, 8, 9), dtype=np.float64)
    x, y, w = li[..., 0], li[..., 1], li[..., 2]
    u, v, t = lc[..., 0], lc[..., 1], lc[..., 2]
    z = np.zeros_like(x)
    r1 = np.stack([z, z, z, -t * x, -t * y, -t * w, v * x, v * y, v * w], axis=-1)
    r2 = np.stack([t * x, t * y, t * w, z, z, z, -u * x, -u * y, -u * w], axis=-1)
    A[:, 0::2, :] = r1
    A[:, 1::2, :] = r2
    _, _, vt = np.linalg.svd(A)
    M = vt[:, -1, :].reshape(N, 3, 3)
    det = np.linalg.det(M)
    ok = np.isfinite(det) & (np.abs(det) > 1e-14)
    H = np.full((N, 3, 3), np.nan)
    if ok.any():
        Minv = np.linalg.inv(M[ok])
        Hn = np.transpose(Minv, (0, 2, 1))
        Hf = np.linalg.inv(S_court)[None] @ Hn @ T_img[None]
        Hf = Hf / Hf[:, 2:3, 2:3].clip(1e-12)
        H[ok] = Hf
    return H


def batch_sample_score(H: np.ndarray, samples_ft: np.ndarray, dist: np.ndarray, roi, tol_px: float, min_area_fraction: float, court: CourtModel) -> np.ndarray:
    """Vectorised version of lines.line_score with validity gating. Returns score [N] (0 for invalid)."""
    N = H.shape[0]
    h, w = dist.shape
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else roi
    finite = np.isfinite(H).all(axis=(1, 2))
    Hz = np.where(finite[:, None, None], H, 0.0)
    det = np.linalg.det(Hz)
    good = finite & (np.abs(det) > 1e-18)
    Hinv = np.zeros_like(H)
    if good.any():
        Hinv[good] = np.linalg.inv(Hz[good])
    S = np.concatenate([samples_ft, np.ones((samples_ft.shape[0], 1))], axis=1)  # [M,3]
    P = S @ np.transpose(Hinv, (0, 2, 1))  # [N,M,3]
    wv = P[..., 2]
    front = wv > 1e-9
    wsafe = np.where(front, wv, 1.0)
    px = P[..., 0] / wsafe
    py = P[..., 1] / wsafe
    inside = front & (px >= x0) & (px < x1 - 1) & (py >= y0) & (py < y1 - 1)
    xi = np.where(inside, px, 0).astype(np.int32)
    yi = np.where(inside, py, 0).astype(np.int32)
    d = dist[yi, xi]
    sc = np.where(inside, np.exp(-d / tol_px), 0.0)
    n_in = inside.sum(axis=1)
    score = np.where(n_in > 0, sc.sum(axis=1) / np.maximum(n_in, 1), 0.0)
    coverage = n_in / samples_ft.shape[0]
    # court centre in front, projected court polygon area large enough
    corners = np.array([[0, 0, 1], [court.length, 0, 1], [court.length, court.width, 1], [0, court.width, 1]], dtype=float)
    C = corners @ np.transpose(Hinv, (0, 2, 1))
    cw = C[..., 2]
    cfront = cw > 1e-9
    cws = np.where(cfront, cw, 1.0)
    cx = np.clip(C[..., 0] / cws, -4 * w, 5 * w)
    cy = np.clip(C[..., 1] / cws, -4 * h, 5 * h)
    area = 0.5 * np.abs(np.sum(cx * np.roll(cy, -1, axis=1) - np.roll(cx, -1, axis=1) * cy, axis=1))
    roi_area = float((x1 - x0) * (y1 - y0))
    valid = good & (coverage > 0.05) & (cfront.sum(axis=1) >= 3) & (area >= min_area_fraction * roi_area) & (area < 50 * roi_area)
    return np.where(valid, score, 0.0)


def mask_explained_score(H: np.ndarray, evidence: LineEvidence, court: CourtModel, roi=None, scale: float = 0.25, tol_px: float = 4.0) -> float:
    """Fraction of line-mask pixels lying within tol of the projected court markings (downscaled)."""
    h, w = evidence.mask.shape
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    canvas = np.zeros((sh, sw), dtype=np.uint8)
    for pl in court.polylines():
        pts, ok = project_to_image(H, pl)
        if not ok.all():
            continue
        pts = pts * scale
        if np.abs(pts).max() > 1e5:
            continue
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 255, 1)
    if not canvas.any():
        return 0.0
    dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
    small = cv2.resize(evidence.mask, (sw, sh), interpolation=cv2.INTER_AREA) > 40
    if roi is not None:
        x0, y0, x1, y1 = roi
        m = np.zeros_like(small)
        m[int(y0 * scale) : int(y1 * scale), int(x0 * scale) : int(x1 * scale)] = small[int(y0 * scale) : int(y1 * scale), int(x0 * scale) : int(x1 * scale)]
        small = m
    if small.sum() < 10:
        return 0.0
    d = dist[small]
    return float(np.mean(np.exp(-d / max(1.0, tol_px * scale))))


def two_sided_score(H: np.ndarray, evidence: LineEvidence, samples: np.ndarray, court: CourtModel, roi, cfg: AutoCalibConfig) -> tuple[float, float, float]:
    s1, cov = line_score(H, samples, evidence.dist, roi, cfg.tol_px)
    s2 = mask_explained_score(H, evidence, court, roi, tol_px=cfg.tol_px)
    return float(np.sqrt(max(s1, 0) * max(s2, 0))), s1, s2


def refine_with_icp(H: np.ndarray, evidence: LineEvidence, court: CourtModel, cfg: AutoCalibConfig, roi=None) -> tuple[np.ndarray, float]:
    """Score-guarded ICP: pull projected samples to the nearest line pixels, re-fit, keep only improvements."""
    samples = court.sample_points(cfg.sample_spacing_ft)
    h, w = evidence.mask.shape
    if (evidence.mask > 0).sum() < 50:
        return H, two_sided_score(H, evidence, samples, court, roi, cfg)[0]
    gx = cv2.Sobel(evidence.dist, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(evidence.dist, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    cur = H.copy()
    best_score = two_sided_score(cur, evidence, samples, court, roi, cfg)[0]
    thr = 6.0
    for it in range(cfg.refine_iterations):
        pts, ok = project_to_image(cur, samples)
        inside = ok & (pts[:, 0] >= 1) & (pts[:, 0] < w - 2) & (pts[:, 1] >= 1) & (pts[:, 1] < h - 2)
        if roi is not None:
            x0, y0, x1, y1 = roi
            inside &= (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        if inside.sum() < 30:
            break
        p = pts[inside]
        xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
        d = evidence.dist[yi, xi]
        near = d < cfg.icp_max_dist_px
        if near.sum() < 30:
            break
        g = np.stack([gx[yi[near], xi[near]], gy[yi[near], xi[near]]], axis=1)
        gn = np.linalg.norm(g, axis=1, keepdims=True).clip(1e-6)
        target = p[near] - g / gn * d[near, None]
        src_ft = samples[inside][near]
        Hci, _ = cv2.findHomography(src_ft.astype(np.float32), target.astype(np.float32), cv2.RANSAC, thr)
        if Hci is None or abs(np.linalg.det(Hci)) < 1e-12:
            break
        new = np.linalg.inv(Hci)
        new /= new[2, 2]
        s = two_sided_score(new, evidence, samples, court, roi, cfg)[0]
        if s <= best_score + 1e-4:
            thr = max(2.0, thr * 0.7)
            if thr <= 2.0 and it > 2:
                break
            continue
        best_score, cur = s, new
        thr = max(2.0, thr * 0.8)
    return cur, best_score


def auto_calibrate(
    frame_bgr: np.ndarray,
    roi: Optional[tuple[int, int, int, int]] = None,
    exclude_boxes: Optional[np.ndarray] = None,
    court: CourtModel = NBA_COURT,
    cfg: AutoCalibConfig = AutoCalibConfig(),
    frame_index: Optional[int] = None,
    initial_H: Optional[np.ndarray] = None,
    orientation_hint_H: Optional[np.ndarray] = None,
) -> Calibration:
    """Estimate image->court. With ``initial_H`` only the refinement/revalidation step runs.

    Court markings are symmetric under end/side flips, so a from-scratch result is
    only determined up to that symmetry.  ``orientation_hint_H`` (e.g. a manual
    calibration or the previous frame) selects the variant closest to the hint;
    without a hint the result is flagged ``orientation_ambiguous`` in its notes.
    """
    h, w = frame_bgr.shape[:2]
    evidence = extract_evidence(frame_bgr, roi, exclude_boxes)

    def finish(H: np.ndarray, method: str, notes: str) -> Calibration:
        H, score = refine_with_icp(H, evidence, court, cfg, roi)
        samples = court.sample_points(cfg.sample_spacing_ft)
        score, s1, s2 = two_sided_score(H, evidence, samples, court, roi, cfg)
        center, ok = project_to_image(H, np.array([[court.length / 2, court.width / 2]]))
        conf = float(np.clip(score, 0, 1))
        valid = bool(ok.all() and conf >= cfg.min_confidence)
        orient = ""
        if method == "auto":
            if orientation_hint_H is not None:
                from .homography import resolve_orientation

                H, chosen = resolve_orientation(H, orientation_hint_H, (w, h), court)
                orient = f"; orientation={chosen} (from hint)"
            else:
                orient = "; orientation_ambiguous (court symmetry: use Flip ends / Flip sides or a hint)"
        return Calibration(H=H, method=method, confidence=conf, valid=valid, frame_index=frame_index, notes=f"{notes}; sample_score={s1:.3f}; mask_score={s2:.3f}{orient}")

    if initial_H is not None:
        return finish(np.asarray(initial_H, dtype=np.float64), "refined", "refined from initial homography")

    lines = merge_segments(evidence.segments)
    fam_a, fam_b = _split_families(lines, cfg.max_lines_per_family)
    if len(fam_a) < 2 or len(fam_b) < 2:
        return Calibration(H=np.eye(3), method="auto", confidence=0.0, valid=False, frame_index=frame_index, notes="not enough line evidence")
    along, across = _court_line_families(court)
    along_l = np.array([[0.0, 1.0, -y] for _, y in along])
    across_l = np.array([[1.0, 0.0, -x] for _, x in across])
    la = np.array([ln["abc"] for ln in fam_a])
    lb = np.array([ln["abc"] for ln in fam_b])
    T_img = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
    S_court = np.array([[2.0 / court.length, 0, -1], [0, 2.0 / court.width, -1], [0, 0, 1]])
    search_samples = court.sample_points(cfg.search_spacing_ft)
    quads_img, quads_court, labels = [], [], []
    for img_along, img_across in ((la, lb), (lb, la)):
        for ia in itertools.combinations(range(len(img_along)), 2):
            for ib in itertools.combinations(range(len(img_across)), 2):
                for ca in itertools.permutations(range(len(along_l)), 2):
                    for cb in itertools.permutations(range(len(across_l)), 2):
                        quads_img.append([img_along[ia[0]], img_along[ia[1]], img_across[ib[0]], img_across[ib[1]]])
                        quads_court.append([along_l[ca[0]], along_l[ca[1]], across_l[cb[0]], across_l[cb[1]]])
                        labels.append(f"{along[ca[0]][0]},{along[ca[1]][0]},{across[cb[0]][0]},{across[cb[1]][0]}")
    quads_img = np.asarray(quads_img)
    quads_court = np.asarray(quads_court)
    scores = np.zeros(len(labels))
    Hs = np.zeros((len(labels), 3, 3))
    for s in range(0, len(labels), cfg.batch):
        H = homographies_from_line_quads(quads_img[s : s + cfg.batch], quads_court[s : s + cfg.batch], T_img, S_court)
        Hs[s : s + cfg.batch] = H
        scores[s : s + cfg.batch] = batch_sample_score(H, search_samples, evidence.dist, roi, cfg.tol_px, cfg.min_area_fraction, court)
    order = np.argsort(-scores)[: cfg.top_k]
    best = (-1.0, None, "")
    samples = court.sample_points(cfg.sample_spacing_ft)
    for i in order:
        if scores[i] <= 0:
            break
        sc, s1, s2 = two_sided_score(Hs[i], evidence, samples, court, roi, cfg)
        if sc > best[0]:
            best = (sc, Hs[i], f"hypothesis={labels[i]} search_score={scores[i]:.3f} two_sided={sc:.3f}")
    if best[1] is None:
        return Calibration(H=np.eye(3), method="auto", confidence=0.0, valid=False, frame_index=frame_index, notes="no consistent hypothesis")
    return finish(best[1], "auto", best[2])
