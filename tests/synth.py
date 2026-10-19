"""Synthetic basketball scene generator for tests and demos.

Renders a 94x50 ft court through a known homography onto a frame, with N
"players" drawn as coloured ellipses standing on the court plane (two jersey
colours), moving along deterministic paths.  Ground truth (foot points in
pixels and court feet, team labels) is returned with every frame, so the
detector/tracker/team/court pipeline can be checked end to end without model
weights or real footage.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from btrv.court.model import NBA_COURT
from btrv.court.homography import project_to_image

TEAM_COLORS_BGR = [(40, 40, 230), (230, 160, 30)]  # red-ish, blue-ish


def camera_homography(width: int, height: int, variant: int = 0) -> np.ndarray:
    """A plausible broadcast-like view: court -> image homography (returns image->court)."""
    L, W = NBA_COURT.length, NBA_COURT.width
    # court corners (TL, TR, BR, BL) mapped to a trapezoid in the image
    pads = [
        (0.08, 0.22, 0.92, 0.22, 1.00, 0.90, 0.00, 0.90),
        (0.12, 0.25, 0.95, 0.20, 1.05, 0.88, -0.05, 0.92),
    ][variant % 2]
    dst = np.float32([[pads[0] * width, pads[1] * height], [pads[2] * width, pads[3] * height], [pads[4] * width, pads[5] * height], [pads[6] * width, pads[7] * height]])
    src = np.float32([[0, 0], [L, 0], [L, W], [0, W]])
    Hci = cv2.getPerspectiveTransform(src, dst)  # court -> image
    H = np.linalg.inv(Hci)
    return H / H[2, 2]


@dataclass
class Player:
    team: int
    x: float
    y: float
    vx: float
    vy: float
    height_ft: float = 6.5


def default_players(n: int = 10, seed: int = 0) -> list[Player]:
    rng = np.random.default_rng(seed)
    ps = []
    for i in range(n):
        ps.append(Player(team=i % 2, x=float(rng.uniform(10, 84)), y=float(rng.uniform(6, 44)), vx=float(rng.uniform(-0.25, 0.25)), vy=float(rng.uniform(-0.15, 0.15))))
    return ps


def step_players(players: list[Player]) -> None:
    for p in players:
        p.x += p.vx
        p.y += p.vy
        if not (3 < p.x < 91):
            p.vx = -p.vx
        if not (3 < p.y < 47):
            p.vy = -p.vy


def render_court(H: np.ndarray, width: int, height: int, floor=(90, 140, 170), line=(245, 245, 245), thickness: int = 2) -> np.ndarray:
    img = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    L, W = NBA_COURT.length, NBA_COURT.width
    poly, _ = project_to_image(H, np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=float))
    cv2.fillPoly(img, [poly.astype(np.int32)], floor)
    for pl in NBA_COURT.polylines():
        pts, ok = project_to_image(H, pl)
        if ok.all():
            cv2.polylines(img, [pts.astype(np.int32)], False, line, thickness, cv2.LINE_AA)
    return img


def render_players(img: np.ndarray, H: np.ndarray, players: list[Player]) -> list[dict]:
    """Draw players as ellipses standing on their foot point; returns ground truth per player."""
    gt = []
    Hci = np.linalg.inv(H)
    for i, p in enumerate(players):
        foot, ok = project_to_image(H, np.array([[p.x, p.y]]))
        head_ft = np.array([[p.x, p.y - 0.001]])  # on-plane proxy; height drawn in pixels using local scale
        # local scale: pixels per foot at the foot point (use a 1 ft offset in y)
        off, _ = project_to_image(H, np.array([[p.x, p.y + 1.0]]))
        ppf = float(np.linalg.norm(off[0] - foot[0]))
        h_px = max(8.0, p.height_ft * ppf * 0.9)
        w_px = max(4.0, h_px * 0.35)
        fx, fy = float(foot[0, 0]), float(foot[0, 1])
        cx, cy = fx, fy - h_px / 2
        color = TEAM_COLORS_BGR[p.team]
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), (int(w_px / 2), int(h_px / 2)), 0, 0, 360, color, -1, cv2.LINE_AA)
        gt.append({"id": i, "team": p.team, "court_xy": (p.x, p.y), "foot_px": (fx, fy), "box": (cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2)})
    return gt


def make_scene_video(path: Path, n_frames: int = 50, width: int = 640, height: int = 360, n_players: int = 8, variant: int = 0, rate: str = "30", gop: int = 15, bframes: int = 2, moving: bool = False):
    """Encode a synthetic scene; returns (H_list, gt_per_frame, frames)."""
    players = default_players(n_players)
    frames = []
    gts = []
    Hs = []
    for f in range(n_frames):
        H = camera_homography(width, height, variant)
        if moving:
            # slow pan: shift the image by a few pixels per frame via a translation on the image side
            T = np.array([[1, 0, 1.5 * f], [0, 1, 0.4 * f], [0, 0, 1]], dtype=float)
            H = H @ np.linalg.inv(T)  # image_t -> image_0 -> court
            H /= H[2, 2]
        img = render_court(H, width, height)
        gt = render_players(img, H, players)
        frames.append(img)
        gts.append(gt)
        Hs.append(H)
        step_players(players)
    ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", rate, "-i", "-",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-bf", str(bframes), "-g", str(gop), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    return Hs, gts, frames
