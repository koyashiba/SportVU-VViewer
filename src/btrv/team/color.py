"""Unsupervised two-team assignment from jersey colour.

Per observation we take the torso region of the box, drop pixels that look
like the floor (dominant colour of the frame around the box) or skin/low
saturation, and keep a compact Lab-colour feature.  Per track the features are
aggregated (median) and all tracks are clustered with k=2 k-means (numpy, no
labelling).  Confidence is derived from the relative distance to the two
centroids and the number of observations, and stays explicit in the output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import cv2
import numpy as np

from btrv.core.types import TeamAssignment


@dataclass
class TeamConfig:
    torso_top: float = 0.18  # fraction of box height
    torso_bottom: float = 0.55
    torso_left: float = 0.25
    torso_right: float = 0.75
    min_saturation: int = 40
    min_observations: int = 3
    ambiguous_ratio: float = 0.8  # d_near/d_far above this -> low confidence
    seed: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def jersey_feature(frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float], cfg: TeamConfig = TeamConfig()) -> Optional[np.ndarray]:
    """Lab (a, b, L/2) mean of saturated torso pixels, or None if too few pixels."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    tx1, tx2 = int(max(0, x1 + cfg.torso_left * bw)), int(min(w, x1 + cfg.torso_right * bw))
    ty1, ty2 = int(max(0, y1 + cfg.torso_top * bh)), int(min(h, y1 + cfg.torso_bottom * bh))
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        return None
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    keep = (sat >= cfg.min_saturation) & (val > 40)
    if keep.sum() < 8:
        keep = val > 40  # white/black jerseys: fall back to brightness
        if keep.sum() < 8:
            return None
    px = lab[keep]
    return np.array([np.median(px[:, 1]), np.median(px[:, 2]), np.median(px[:, 0]) * 0.5], dtype=np.float32)


def kmeans2(X: np.ndarray, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two-cluster k-means with farthest-pair initialisation. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int), np.stack([X[0], X[0]])
    i0 = int(rng.integers(n))
    d0 = np.linalg.norm(X - X[i0], axis=1)
    i1 = int(np.argmax(d0))
    d1 = np.linalg.norm(X - X[i1], axis=1)
    i2 = int(np.argmax(d1))
    C = np.stack([X[i1], X[i2]]).astype(np.float64)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.stack([np.linalg.norm(X - C[0], axis=1), np.linalg.norm(X - C[1], axis=1)], axis=1)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                C[k] = X[labels == k].mean(axis=0)
    return labels, C


class TeamAssigner:
    """Collects per-track jersey features during processing and clusters them at the end."""

    def __init__(self, cfg: TeamConfig = TeamConfig()):
        self.cfg = cfg
        self._obs: dict[int, list[np.ndarray]] = {}
        self._untracked: list[tuple[int, int, np.ndarray]] = []  # (frame_index, box_ordinal, feature)
        self.centroids: Optional[np.ndarray] = None
        self.track_teams: dict[int, TeamAssignment] = {}

    def observe(self, frame_bgr: np.ndarray, xyxy, track_id: Optional[int], frame_index: int = -1, ordinal: int = -1) -> None:
        f = jersey_feature(frame_bgr, xyxy, self.cfg)
        if f is None:
            return
        if track_id is None:
            self._untracked.append((frame_index, ordinal, f))
        else:
            self._obs.setdefault(int(track_id), []).append(f)

    def _assign_feature(self, f: np.ndarray, n_obs: int) -> TeamAssignment:
        d = np.array([np.linalg.norm(f - self.centroids[0]), np.linalg.norm(f - self.centroids[1])])
        k = int(np.argmin(d))
        near, far = d[k], d[1 - k]
        ratio = near / far if far > 1e-6 else 1.0
        sep = 1.0 - min(1.0, ratio / self.cfg.ambiguous_ratio) if self.cfg.ambiguous_ratio > 0 else 0.0
        obs_term = min(1.0, n_obs / max(1, self.cfg.min_observations))
        conf = float(np.clip(0.15 + 0.85 * sep, 0, 1) * (0.5 + 0.5 * obs_term))
        if ratio >= self.cfg.ambiguous_ratio or n_obs < 1:
            return TeamAssignment(None, conf)
        return TeamAssignment(k, conf)

    def fit(self) -> dict[int, TeamAssignment]:
        tracks = [t for t, obs in self._obs.items() if len(obs) >= 1]
        if len(tracks) < 2:
            self.track_teams = {t: TeamAssignment(None, 0.0) for t in tracks}
            return self.track_teams
        X = np.stack([np.median(np.stack(self._obs[t]), axis=0) for t in tracks]).astype(np.float64)
        labels, C = kmeans2(X, seed=self.cfg.seed)
        # deterministic ordering: team 0 = cluster with the smaller 'a' channel mean (stable across runs)
        if C[1][0] < C[0][0]:
            C = C[::-1]
        self.centroids = C
        self.track_teams = {t: self._assign_feature(X[i], len(self._obs[t])) for i, t in enumerate(tracks)}
        return self.track_teams

    def assign_untracked(self) -> dict[tuple[int, int], TeamAssignment]:
        if self.centroids is None:
            return {}
        return {(fi, o): self._assign_feature(f, 1) for fi, o, f in self._untracked}

    def separation(self) -> Optional[float]:
        if self.centroids is None:
            return None
        return float(np.linalg.norm(self.centroids[0] - self.centroids[1]))

    def to_json(self) -> dict[str, Any]:
        return {
            "method": "jersey_color_kmeans",
            "feature": "median Lab(a, b, L/2) of saturated torso pixels",
            "config": self.cfg.to_json(),
            "centroids_lab": None if self.centroids is None else self.centroids.tolist(),
            "centroid_separation": self.separation(),
            "tracks": {str(t): a.to_json() for t, a in self.track_teams.items()},
            "n_tracks": len(self._obs),
        }
