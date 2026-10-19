"""Image <-> court homographies with validation and confidence."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import cv2
import numpy as np

from .model import CourtModel, NBA_COURT


def _h(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return projected points [N,2] and a boolean validity mask (w > 0: in front of the camera)."""
    p = _h(pts) @ np.asarray(H, dtype=np.float64).T
    w = p[:, 2]
    ok = np.isfinite(w) & (w > 1e-9)
    out = np.full((p.shape[0], 2), np.nan)
    out[ok] = p[ok, :2] / w[ok, None]
    return out, ok


def project_to_court(H_image_to_court: np.ndarray, pts_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return apply_homography(H_image_to_court, pts_px)


def project_to_image(H_image_to_court: np.ndarray, pts_ft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return apply_homography(np.linalg.inv(H_image_to_court), pts_ft)


def in_court(xy: np.ndarray, court: CourtModel = NBA_COURT, margin_ft: float = 0.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= -margin_ft)
        & (xy[:, 0] <= court.length + margin_ft)
        & (xy[:, 1] >= -margin_ft)
        & (xy[:, 1] <= court.width + margin_ft)
    )


@dataclass
class Calibration:
    """A validated image->court homography for one reference frame (or a propagated one)."""

    H: np.ndarray  # 3x3 image pixels -> court feet
    method: str  # manual | auto | propagated | refined
    confidence: float  # 0..1
    valid: bool = True
    frame_index: Optional[int] = None
    reprojection_error_px: Optional[float] = None
    inlier_ratio: Optional[float] = None
    landmarks: list[dict[str, Any]] = field(default_factory=list)  # [{name, image_xy, court_xy}]
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["H"] = np.asarray(self.H, dtype=float).tolist()
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Calibration":
        return Calibration(
            H=np.asarray(d["H"], dtype=np.float64).reshape(3, 3),
            method=str(d.get("method", "manual")),
            confidence=float(d.get("confidence", 0.0)),
            valid=bool(d.get("valid", True)),
            frame_index=d.get("frame_index"),
            reprojection_error_px=d.get("reprojection_error_px"),
            inlier_ratio=d.get("inlier_ratio"),
            landmarks=list(d.get("landmarks", [])),
            notes=str(d.get("notes", "")),
        )

    def invalid_copy(self, notes: str) -> "Calibration":
        return Calibration(self.H.copy(), self.method, 0.0, False, self.frame_index, self.reprojection_error_px, self.inlier_ratio, [], notes)


def confidence_from_errors(rmse_px: float, inlier_ratio: float, image_diag_px: float, n_points: int) -> float:
    """Heuristic confidence: 1 for sub-pixel fits with many inliers, decaying with error."""
    rel = rmse_px / max(image_diag_px, 1.0)
    err_term = float(np.exp(-rel / 0.004))  # rmse of 0.4% of the diagonal -> ~0.37
    n_term = min(1.0, n_points / 6.0)
    return float(np.clip(err_term * inlier_ratio * (0.5 + 0.5 * n_term), 0.0, 1.0))


def fit_homography(
    image_pts: np.ndarray,
    court_pts: np.ndarray,
    image_size: tuple[int, int],
    method: str = "manual",
    frame_index: Optional[int] = None,
    landmark_names: Optional[list[str]] = None,
    ransac_threshold_px: float = 8.0,
    court: CourtModel = NBA_COURT,
) -> Calibration:
    """Fit image->court from >= 4 point correspondences and validate it.

    Validation: reprojection error (court->image), inlier ratio, non-degenerate
    determinant, and that the court centre projects in front of the camera.
    """
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    court_pts = np.asarray(court_pts, dtype=np.float64).reshape(-1, 2)
    n = image_pts.shape[0]
    if n < 4 or court_pts.shape[0] != n:
        raise ValueError(f"need at least 4 point pairs, got {n} image and {court_pts.shape[0]} court points")
    if n == 4:
        H, mask = cv2.findHomography(image_pts, court_pts, 0)
        mask = np.ones((n, 1), dtype=np.uint8)
    else:
        # RANSAC in image space: fit court->image, then invert, so the threshold is in pixels
        Hci, mask = cv2.findHomography(court_pts, image_pts, cv2.RANSAC, ransac_threshold_px)
        if Hci is None or abs(np.linalg.det(Hci)) < 1e-12:
            raise ValueError("homography fit failed (degenerate configuration)")
        H = np.linalg.inv(Hci)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        raise ValueError("homography fit failed (degenerate configuration)")
    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(n, dtype=bool)
    reproj, ok = project_to_image(H, court_pts)
    err = np.linalg.norm(reproj - image_pts, axis=1)
    err_in = err[inliers & ok]
    rmse = float(np.sqrt(np.mean(err_in**2))) if err_in.size else float("inf")
    w, h = image_size
    diag = float(np.hypot(w, h))
    center_img, center_ok = project_to_image(H, np.array([[court.length / 2, court.width / 2]]))
    valid = bool(np.isfinite(rmse) and center_ok.all() and inliers.sum() >= 4)
    conf = confidence_from_errors(rmse, float(inliers.mean()), diag, int(inliers.sum())) if valid else 0.0
    lms = []
    for i in range(n):
        lms.append({
            "name": landmark_names[i] if landmark_names and i < len(landmark_names) else f"p{i}",
            "image_xy": image_pts[i].tolist(),
            "court_xy": court_pts[i].tolist(),
            "inlier": bool(inliers[i]),
            "error_px": float(err[i]) if np.isfinite(err[i]) else None,
        })
    return Calibration(H=H, method=method, confidence=conf, valid=valid, frame_index=frame_index, reprojection_error_px=rmse, inlier_ratio=float(inliers.mean()), landmarks=lms)


def project_court_lines(H_image_to_court: np.ndarray, court: CourtModel = NBA_COURT, image_size: Optional[tuple[int, int]] = None) -> list[np.ndarray]:
    """Court polylines projected to image pixels; segments behind the camera are dropped."""
    out = []
    for pl in court.polylines():
        pts, ok = project_to_image(H_image_to_court, pl)
        if image_size is not None:
            w, h = image_size
            ok &= (np.abs(pts[:, 0]) < 8 * w) & (np.abs(pts[:, 1]) < 8 * h)
        # split into runs of valid points
        run = []
        for p, o in zip(pts, ok):
            if o:
                run.append(p)
            elif run:
                if len(run) > 1:
                    out.append(np.array(run))
                run = []
        if len(run) > 1:
            out.append(np.array(run))
    return out


def foot_points_to_court(H: Optional[np.ndarray], foot_xy: np.ndarray, court: CourtModel = NBA_COURT, margin_ft: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Project ground-contact points; returns court xy [N,2] and validity (finite, in front, on court within margin)."""
    foot_xy = np.asarray(foot_xy, dtype=float).reshape(-1, 2)
    if H is None or foot_xy.shape[0] == 0:
        return np.full((foot_xy.shape[0], 2), np.nan), np.zeros(foot_xy.shape[0], dtype=bool)
    xy, ok = project_to_court(H, foot_xy)
    ok &= in_court(xy, court, margin_ft)
    return xy, ok


# ---------------------------------------------------------------------------
# Court symmetry: line evidence cannot tell the two ends / two sidelines apart.
# ---------------------------------------------------------------------------
def symmetry_transforms(court: CourtModel = NBA_COURT) -> dict[str, np.ndarray]:
    L, W = court.length, court.width
    return {
        "identity": np.eye(3),
        "flip_ends": np.array([[-1, 0, L], [0, 1, 0], [0, 0, 1]], dtype=float),  # x -> L - x
        "flip_sides": np.array([[1, 0, 0], [0, -1, W], [0, 0, 1]], dtype=float),  # y -> W - y
        "rotate_180": np.array([[-1, 0, L], [0, -1, W], [0, 0, 1]], dtype=float),
    }


def apply_symmetry(H_image_to_court: np.ndarray, name: str, court: CourtModel = NBA_COURT) -> np.ndarray:
    S = symmetry_transforms(court)[name]
    H = S @ np.asarray(H_image_to_court, dtype=np.float64)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def resolve_orientation(H_image_to_court: np.ndarray, hint_H: np.ndarray, image_size: tuple[int, int], court: CourtModel = NBA_COURT) -> tuple[np.ndarray, str]:
    """Pick the symmetric variant of H whose court->image mapping is closest to the hint's."""
    w, h = image_size
    pts = np.array([[x, y] for x in np.linspace(0, court.length, 5) for y in np.linspace(0, court.width, 4)], dtype=float)
    ref, ok_ref = project_to_image(hint_H, pts)
    best, best_name = None, "identity"
    best_err = float("inf")
    for name in symmetry_transforms(court):
        Hs = apply_symmetry(H_image_to_court, name, court)
        cand, ok = project_to_image(Hs, pts)
        m = ok & ok_ref
        if m.sum() < 4:
            continue
        err = float(np.mean(np.linalg.norm(cand[m] - ref[m], axis=1)))
        if err < best_err:
            best, best_name, best_err = Hs, name, err
    return (best if best is not None else H_image_to_court), best_name
