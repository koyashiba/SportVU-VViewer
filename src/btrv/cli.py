"""Command-line entry points (processing, exports, calibration, examples, viewer)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_info(a):
    from btrv.source import FileSource

    src = FileSource(a.video, source_id=a.source_id)
    info = src.info.to_json()
    idx = src.index
    info["first_timestamps_s"] = [round(idx.timestamp_s(i), 6) for i in range(min(5, idx.frame_count))]
    info["keyframes"] = int(idx.keyframe.sum())
    print(json.dumps(info, indent=2))
    src.close()


def _cmd_process(a):
    import threading

    from btrv.detect.base import DetectorConfig
    from btrv.pipeline import ProcessingConfig, process_video

    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    cfg = ProcessingConfig(
        video_path=a.video,
        run_dir=a.run_dir,
        source_id=a.source_id,
        camera_mode=a.camera_mode,
        roi=roi,
        frame_start=a.start,
        frame_stop=a.stop,
        frame_step=a.step,
        detector=DetectorConfig(backend=a.detector, model_id=a.model_id, confidence=a.confidence, compute=a.compute, max_inference_side=a.max_side, extra={"path": a.boxes_file} if a.boxes_file else {}),
        calibration_path=a.calibration,
        auto_calibrate=a.auto_calibrate,
    )

    def prog(i, n, msg):
        sys.stderr.write(f"\r{i}/{n} {msg}      ")
        sys.stderr.flush()

    res = process_video(cfg, progress=prog, cancel=threading.Event())
    sys.stderr.write("\n")
    print(json.dumps(res.summary, indent=2))


def _cmd_export_sportvu(a):
    from btrv.io.ball import read_ball_file
    from btrv.io.sportvu import SportVUExportOptions, export_sportvu, write_sportvu
    from btrv.storage import BoxesTable, RunStore

    store = RunStore(a.run_dir)
    table = BoxesTable().extend(store.iter_boxes("tracked_boxes.jsonl"))
    ball = read_ball_file(a.ball) if a.ball else None
    opts = SportVUExportOptions(game_id=a.game_id, epoch_ms=a.epoch_ms, ball_z_default_ft=a.ball_z_default, frame_start=a.start, frame_stop=a.stop)
    doc, meta = export_sportvu(table, ball, opts, player_only_debug=a.player_only_debug)
    p, mp = write_sportvu(a.out, doc, meta)
    print(json.dumps({"written": str(p), "meta": str(mp), "kind": meta["kind"], "stats": meta["stats"]}, indent=2))


def _cmd_export_boxes(a):
    from btrv.io.boxes import write_boxes_file
    from btrv.storage import RunStore

    store = RunStore(a.run_dir)
    n = write_boxes_file(a.out, store.iter_boxes("raw_boxes.jsonl" if a.raw else "tracked_boxes.jsonl"))
    print(json.dumps({"written": a.out, "frames": n}))


def _cmd_calibrate_auto(a):
    from btrv.court.auto_calibrate import auto_calibrate
    from btrv.pipeline.calibration_file import calibration_document, save_calibration
    from btrv.source import FileSource

    src = FileSource(a.video, source_id=a.source_id)
    info, frame = src.read(a.frame)
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    cal = auto_calibrate(frame, roi, frame_index=a.frame)
    doc = calibration_document(src.info.source_id, (src.info.width, src.info.height), "fixed" if roi else a.camera_mode, [cal], roi, cal.notes)
    save_calibration(a.out, doc)
    print(json.dumps({"valid": cal.valid, "confidence": cal.confidence, "notes": cal.notes, "written": a.out}, indent=2))
    src.close()


def _cmd_make_examples(a):
    from btrv.examples import write_examples

    paths = write_examples(Path(a.out_dir))
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def _cmd_viewer(a):
    from btrv.viewer.app import main as viewer_main

    viewer_main(a.rest)


def main(argv=None):
    p = argparse.ArgumentParser(prog="btrv", description="Basketball Tracking & Research Viewer")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="Print the frame index summary of a video (pts-based timestamps)")
    s.add_argument("video")
    s.add_argument("--source-id")
    s.set_defaults(fn=_cmd_info)

    s = sub.add_parser("process", help="Precompute detections, tracks, teams and court positions into a run directory")
    s.add_argument("video")
    s.add_argument("run_dir")
    s.add_argument("--source-id")
    s.add_argument("--camera-mode", choices=["fixed", "moving"], default="fixed")
    s.add_argument("--roi", help="x0,y0,x1,y1 in source pixels (fixed mode)")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int)
    s.add_argument("--step", type=int, default=1)
    s.add_argument("--detector", choices=["roboflow_local", "roboflow_hosted", "fake", "boxes_file"], default="roboflow_local")
    s.add_argument("--model-id", default="bachelor-nozvk/basketball-player-detection-jwk5o-t33zb-2-rfdetr-small-t1")
    s.add_argument("--confidence", type=float, default=0.3)
    s.add_argument("--compute", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    s.add_argument("--max-side", type=int, help="Downscale the inference image so its longest side is at most this many pixels")
    s.add_argument("--boxes-file", help="for --detector boxes_file: btrv.boxes file to replay as detections")
    s.add_argument("--calibration", help="btrv.calibration JSON (manual/auto reference calibration)")
    s.add_argument("--auto-calibrate", action="store_true", help="fixed mode: estimate the calibration from court lines on the first frame")
    s.set_defaults(fn=_cmd_process)

    s = sub.add_parser("export-sportvu", help="Export SportVU moments from a run (needs a ball file unless --player-only-debug)")
    s.add_argument("run_dir")
    s.add_argument("out")
    s.add_argument("--ball", help="btrv.ball file with external ball coordinates (court feet)")
    s.add_argument("--player-only-debug", action="store_true", help="NONSTANDARD export without a ball entry")
    s.add_argument("--game-id", default="btrv")
    s.add_argument("--epoch-ms", type=int, default=0)
    s.add_argument("--ball-z-default", type=float, default=0.0)
    s.add_argument("--start", type=int)
    s.add_argument("--stop", type=int)
    s.set_defaults(fn=_cmd_export_sportvu)

    s = sub.add_parser("export-boxes", help="Export per-frame boxes (tracked by default, --raw for detector output)")
    s.add_argument("run_dir")
    s.add_argument("out")
    s.add_argument("--raw", action="store_true")
    s.set_defaults(fn=_cmd_export_boxes)

    s = sub.add_parser("calibrate-auto", help="Automatic court calibration on one frame")
    s.add_argument("video")
    s.add_argument("out")
    s.add_argument("--frame", type=int, default=0)
    s.add_argument("--roi")
    s.add_argument("--camera-mode", default="moving")
    s.add_argument("--source-id")
    s.set_defaults(fn=_cmd_calibrate_auto)

    s = sub.add_parser("make-examples", help="Write example interchange files (boxes, heat map, ball, SportVU)")
    s.add_argument("out_dir")
    s.set_defaults(fn=_cmd_make_examples)

    s = sub.add_parser("viewer", help="Launch the desktop viewer")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=_cmd_viewer)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
