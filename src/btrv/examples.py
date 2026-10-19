"""Generate small example interchange files (used by docs and tests)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from btrv.core.types import BoxRecord, CourtPoint, FrameBoxes, TeamAssignment
from btrv.io.ball import BallRecord, write_ball_file
from btrv.io.boxes import write_boxes_file
from btrv.io.heatmap import HeatmapMeta, HeatmapWriter
from btrv.io.sportvu import SportVUExportOptions, export_sportvu, write_sportvu
from btrv.storage import BoxesTable


def write_examples(out_dir: Path, source_id: str = "example_game", n_frames: int = 10, image_size=(1920, 1080)) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = image_size
    frames = []
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        boxes = []
        for t in range(6):
            cx = 300 + 220 * t + 4 * i
            boxes.append(BoxRecord((cx - 30, 500.0, cx + 30, 680.0), "players", 0, float(0.6 + 0.04 * t), track_id=t + 1,
                                   team=TeamAssignment(t % 2, 0.9), court=CourtPoint(20 + 9 * t + 0.2 * i, 15 + 4 * t, True, 0.95)))
        frames.append(FrameBoxes(source_id, i, round(i / 25.0, 6), W, H, boxes, producer="example"))
    paths = {}
    paths["boxes"] = out_dir / "external_boxes.example.jsonl"
    write_boxes_file(paths["boxes"], frames)
    ball = [BallRecord(source_id, i, round(i / 25.0, 6), 40.0 + i * 0.5, 25.0, 6.0 if i % 3 else None, 0.8) for i in range(n_frames)]
    paths["ball"] = out_dir / "ball.example.jsonl"
    write_ball_file(paths["ball"], ball)
    # image-space heat map, 96x54 cells over the full frame, gaussian around the ball's image position
    meta = HeatmapMeta(source_id, "image_pixels", [0, 0, W, H], 96, 54, "probability", "sum_to_one", W, H, "example", "float16")
    for suffix in (".h5", ".npz"):
        p = out_dir / f"ball_heatmap.example{suffix}"
        with HeatmapWriter(p, meta) as w:
            ys, xs = np.mgrid[0:54, 0:96]
            for i in range(n_frames):
                cx, cy = 40 + 2 * i, 27
                g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 4.0**2)).astype(np.float32)
                g /= g.sum()
                w.append(i, round(i / 25.0, 6), g)
        paths["heatmap" + suffix] = p
    # court-space heat map
    meta_c = HeatmapMeta(source_id, "court_feet", [0, 0, 94, 50], 94, 50, "probability", "max_one", producer="example")
    p = out_dir / "ball_heatmap_court.example.npz"
    with HeatmapWriter(p, meta_c) as w:
        ys, xs = np.mgrid[0:50, 0:94]
        for i in range(n_frames):
            g = np.exp(-((xs - (40 + i * 0.5)) ** 2 + (ys - 25) ** 2) / (2 * 3.0**2)).astype(np.float32)
            g /= g.max()
            w.append(i, round(i / 25.0, 6), g)
    paths["heatmap_court"] = p
    table = BoxesTable().extend(frames)
    doc, m = export_sportvu(table, ball, SportVUExportOptions(game_id="example"))
    paths["sportvu"], paths["sportvu_meta"] = write_sportvu(out_dir / "sportvu.example.json", doc, m)
    doc, m = export_sportvu(table, None, SportVUExportOptions(game_id="example"), player_only_debug=True)
    paths["sportvu_debug"], paths["sportvu_debug_meta"] = write_sportvu(out_dir / "sportvu_player_only_debug.example.json", doc, m)
    return paths
