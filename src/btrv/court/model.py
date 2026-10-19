"""Court geometry in SportVU-style feet (see docs/coordinates.md).

Origin (0, 0) is the top-left corner of the court as drawn by the reference
viewer (linouk23/NBA-Player-Movements): x runs along the 94 ft length, y runs
across the 50 ft width and increases *downwards* in that drawing.  The left
basket is at (5.25, 25), the right basket at (88.75, 25).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from btrv.core.types import COURT_LENGTH_FT, COURT_WIDTH_FT


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    description: str = ""


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 48) -> np.ndarray:
    t = np.linspace(math.radians(a0), math.radians(a1), n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


@dataclass
class CourtModel:
    """Parametric court (NBA dimensions by default). All values in feet."""

    name: str = "nba"
    length: float = COURT_LENGTH_FT
    width: float = COURT_WIDTH_FT
    basket_from_baseline: float = 5.25
    lane_width: float = 16.0
    free_throw_from_baseline: float = 19.0
    free_throw_circle_radius: float = 6.0
    center_circle_radius: float = 6.0
    three_point_radius: float = 23.75
    three_point_corner_from_sideline: float = 3.0
    restricted_area_radius: float = 4.0
    landmarks: list[Landmark] = field(default_factory=list)

    def __post_init__(self):
        if not self.landmarks:
            self.landmarks = self._default_landmarks()

    # -- landmarks -----------------------------------------------------------
    def _default_landmarks(self) -> list[Landmark]:
        L, W = self.length, self.width
        cy = W / 2
        ft, lw = self.free_throw_from_baseline, self.lane_width / 2
        c3 = self.three_point_corner_from_sideline
        bx = self.basket_from_baseline
        r3 = self.three_point_radius
        dy = cy - c3
        x3 = bx + math.sqrt(max(r3 * r3 - dy * dy, 0.0))
        lm = [
            ("corner_top_left", 0, 0, "Intersection of left baseline and top sideline"),
            ("corner_top_right", L, 0, "Intersection of right baseline and top sideline"),
            ("corner_bottom_left", 0, W, "Intersection of left baseline and bottom sideline"),
            ("corner_bottom_right", L, W, "Intersection of right baseline and bottom sideline"),
            ("half_top", L / 2, 0, "Half-court line meets top sideline"),
            ("half_bottom", L / 2, W, "Half-court line meets bottom sideline"),
            ("center", L / 2, cy, "Centre of the court"),
            ("center_circle_top", L / 2, cy - self.center_circle_radius, "Top of the centre circle"),
            ("center_circle_bottom", L / 2, cy + self.center_circle_radius, "Bottom of the centre circle"),
            ("left_lane_baseline_top", 0, cy - lw, "Left lane (key) meets baseline, top side"),
            ("left_lane_baseline_bottom", 0, cy + lw, "Left lane meets baseline, bottom side"),
            ("left_free_throw_top", ft, cy - lw, "Left free-throw line, top corner of the key"),
            ("left_free_throw_bottom", ft, cy + lw, "Left free-throw line, bottom corner of the key"),
            ("left_free_throw_center", ft, cy, "Middle of the left free-throw line"),
            ("left_basket", bx, cy, "Centre of the left hoop (projected to the floor)"),
            ("left_three_corner_top_baseline", 0, c3, "Left corner-three line meets baseline, top"),
            ("left_three_corner_bottom_baseline", 0, W - c3, "Left corner-three line meets baseline, bottom"),
            ("left_three_corner_top_end", x3, c3, "Left corner-three straight line meets the arc, top"),
            ("left_three_corner_bottom_end", x3, W - c3, "Left corner-three straight line meets the arc, bottom"),
            ("left_three_apex", bx + r3, cy, "Farthest point of the left three-point arc"),
            ("right_lane_baseline_top", L, cy - lw, "Right lane meets baseline, top side"),
            ("right_lane_baseline_bottom", L, cy + lw, "Right lane meets baseline, bottom side"),
            ("right_free_throw_top", L - ft, cy - lw, "Right free-throw line, top corner of the key"),
            ("right_free_throw_bottom", L - ft, cy + lw, "Right free-throw line, bottom corner of the key"),
            ("right_free_throw_center", L - ft, cy, "Middle of the right free-throw line"),
            ("right_basket", L - bx, cy, "Centre of the right hoop (projected to the floor)"),
            ("right_three_corner_top_baseline", L, c3, "Right corner-three line meets baseline, top"),
            ("right_three_corner_bottom_baseline", L, W - c3, "Right corner-three line meets baseline, bottom"),
            ("right_three_corner_top_end", L - x3, c3, "Right corner-three straight line meets the arc, top"),
            ("right_three_corner_bottom_end", L - x3, W - c3, "Right corner-three straight line meets the arc, bottom"),
            ("right_three_apex", L - bx - r3, cy, "Farthest point of the right three-point arc"),
        ]
        return [Landmark(n, float(x), float(y), d) for n, x, y, d in lm]

    def landmark(self, name: str) -> Landmark:
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        raise KeyError(name)

    def landmark_names(self) -> list[str]:
        return [lm.name for lm in self.landmarks]

    # -- lines ---------------------------------------------------------------
    def straight_lines(self) -> dict[str, list[tuple[float, float, float, float]]]:
        """Named straight segments (x1, y1, x2, y2) used by the automatic calibrator."""
        L, W = self.length, self.width
        cy = W / 2
        ft, lw = self.free_throw_from_baseline, self.lane_width / 2
        c3 = self.three_point_corner_from_sideline
        x3 = self.landmark("left_three_corner_top_end").x
        return {
            "sideline_top": [(0, 0, L, 0)],
            "sideline_bottom": [(0, W, L, W)],
            "baseline_left": [(0, 0, 0, W)],
            "baseline_right": [(L, 0, L, W)],
            "half": [(L / 2, 0, L / 2, W)],
            "lane_left_top": [(0, cy - lw, ft, cy - lw)],
            "lane_left_bottom": [(0, cy + lw, ft, cy + lw)],
            "free_throw_left": [(ft, cy - lw, ft, cy + lw)],
            "lane_right_top": [(L - ft, cy - lw, L, cy - lw)],
            "lane_right_bottom": [(L - ft, cy + lw, L, cy + lw)],
            "free_throw_right": [(L - ft, cy - lw, L - ft, cy + lw)],
            "three_left_top": [(0, c3, x3, c3)],
            "three_left_bottom": [(0, W - c3, x3, W - c3)],
            "three_right_top": [(L - x3, c3, L, c3)],
            "three_right_bottom": [(L - x3, W - c3, L, W - c3)],
        }

    def polylines(self) -> list[np.ndarray]:
        """All court markings as polylines [N, 2] in feet (for overlays and scoring)."""
        L, W = self.length, self.width
        cy = W / 2
        bx = self.basket_from_baseline
        r3 = self.three_point_radius
        ftr = self.free_throw_circle_radius
        out: list[np.ndarray] = []
        for segs in self.straight_lines().values():
            for x1, y1, x2, y2 in segs:
                out.append(np.array([[x1, y1], [x2, y2]], dtype=float))
        # centre circle
        out.append(_arc(L / 2, cy, self.center_circle_radius, 0, 360, 72))
        # free-throw circles
        out.append(_arc(self.free_throw_from_baseline, cy, ftr, -90, 90, 36))
        out.append(_arc(L - self.free_throw_from_baseline, cy, ftr, 90, 270, 36))
        # three-point arcs
        c3 = self.three_point_corner_from_sideline
        ang = math.degrees(math.asin((cy - c3) / r3))
        out.append(_arc(bx, cy, r3, -ang, ang, 64))
        out.append(_arc(L - bx, cy, r3, 180 - ang, 180 + ang, 64))
        # restricted areas
        out.append(_arc(bx, cy, self.restricted_area_radius, -90, 90, 24))
        out.append(_arc(L - bx, cy, self.restricted_area_radius, 90, 270, 24))
        # hoops (0.75 ft radius)
        out.append(_arc(bx, cy, 0.75, 0, 360, 24))
        out.append(_arc(L - bx, cy, 0.75, 0, 360, 24))
        return out

    def sample_points(self, spacing_ft: float = 0.5) -> np.ndarray:
        """Dense points along all markings [M, 2] (used for scoring automatic calibration)."""
        pts = []
        for pl in self.polylines():
            for a, b in zip(pl[:-1], pl[1:]):
                n = max(2, int(np.linalg.norm(b - a) / spacing_ft) + 1)
                t = np.linspace(0, 1, n)[:, None]
                pts.append(a[None, :] * (1 - t) + b[None, :] * t)
        return np.unique(np.round(np.concatenate(pts), 3), axis=0)

    def contains(self, x: float, y: float, margin_ft: float = 0.0) -> bool:
        return (-margin_ft <= x <= self.length + margin_ft) and (-margin_ft <= y <= self.width + margin_ft)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "convention": "sportvu_feet_v1",
            "length_ft": self.length,
            "width_ft": self.width,
            "origin": "top-left corner of the court in the reference viewer orientation; x along length, y across width (downwards)",
            "left_basket_xy": [self.basket_from_baseline, self.width / 2],
            "right_basket_xy": [self.length - self.basket_from_baseline, self.width / 2],
        }


NBA_COURT = CourtModel()
