"""Smoke test of the real pinned model through the full pipeline.

1. Obtain real basketball frames: a local image/video (--image / --video), else public dataset
   images of the Roboflow project (direct URLs, then the project search API with the key).
2. Build a short clip that pans a crop window across the image (simulated camera motion), so the
   tracker has to keep ids across frames.
3. Run ``process_video`` with the requested detector backend and print the run summary.
4. Fail (exit 1) if the model returned no players or ids did not persist.

Usage:  python scripts/model_smoke.py --backend roboflow_hosted --out smoke_run
Needs ROBOFLOW_API_KEY (CI maps the repository secret RF_API_KEY to it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests

WORKSPACE = "bachelor-nozvk"
PROJECT = "basketball-player-detection-jwk5o-t33zb"
OWNER = "MKZwWZNXVkeyXCPFIfyqUN6wjNH2"
KNOWN_IMAGE_IDS = ["c1d3Syr0oSHZvcKygc1x", "hSY9NA47aZPgpzzeZLog", "xE3Ulcx2M6aQm0hqPhki"]


def source_url(image_id: str) -> str:
    return f"https://source.roboflow.com/{OWNER}/{image_id}/original.jpg"


def download(url: str, dest: Path, api_key: str | None = None) -> bool:
    try:
        r = requests.get(url, timeout=60, params={"api_key": api_key} if api_key else None)
        if r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff":
            dest.write_bytes(r.content)
            return True
        print(f"  {url}: HTTP {r.status_code}, {len(r.content)} bytes", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"  {url}: {e}", file=sys.stderr)
    return False


def fetch_dataset_image(dest: Path, api_key: str | None) -> bool:
    for iid in KNOWN_IMAGE_IDS:
        if download(source_url(iid), dest) or (api_key and download(source_url(iid), dest, api_key)):
            return True
    if api_key:
        try:
            r = requests.post(f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search", params={"api_key": api_key},
                              json={"prompt": "", "limit": 5, "fields": ["id", "name", "owner"]}, timeout=60)
            r.raise_for_status()
            for res in r.json().get("results", []):
                if download(f"https://source.roboflow.com/{res.get('owner', OWNER)}/{res['id']}/original.jpg", dest) or download(f"https://source.roboflow.com/{res.get('owner', OWNER)}/{res['id']}/original.jpg", dest, api_key):
                    return True
        except Exception as e:  # pragma: no cover
            print(f"  search API failed: {e}", file=sys.stderr)
    return False


def make_pan_clip(image: np.ndarray, path: Path, n_frames: int = 24, crop_frac: float = 0.85, rate: int = 25) -> None:
    h, w = image.shape[:2]
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    cw -= cw % 2
    ch -= ch % 2
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        x0 = int((w - cw) * t)
        y0 = int((h - ch) * (0.5 + 0.5 * np.sin(t * np.pi)))
        frames.append(np.ascontiguousarray(image[y0 : y0 + ch, x0 : x0 + cw]))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}", "-r", str(rate), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "12", "-bf", "2", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="roboflow_hosted", choices=["roboflow_hosted", "roboflow_local", "fake"])
    ap.add_argument("--out", default="smoke_run")
    ap.add_argument("--image", help="local basketball image to build the clip from")
    ap.add_argument("--video", help="local basketball video (first --frames frames are processed)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--compute", default="auto")
    ap.add_argument("--max-side", type=int, default=None)
    a = ap.parse_args()

    from btrv.detect.base import DetectorConfig
    from btrv.pipeline import ProcessingConfig, process_video

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if a.backend != "fake" and not api_key:
        print("ROBOFLOW_API_KEY is not set (CI maps the RF_API_KEY secret to it)", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="btrv_smoke_"))
    if a.video:
        video = Path(a.video)
    else:
        if a.image:
            img_path = Path(a.image)
        else:
            img_path = work / "frame.jpg"
            print("fetching a public dataset image of the project ...")
            if not fetch_dataset_image(img_path, api_key):
                print("could not obtain a basketball image; pass --image or --video", file=sys.stderr)
                return 3
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"cannot read image {img_path}", file=sys.stderr)
            return 3
        print(f"image {image.shape[1]}x{image.shape[0]}; building a {a.frames}-frame panning clip")
        video = work / "clip.mp4"
        make_pan_clip(image, video, a.frames)
    cfg = ProcessingConfig(
        video_path=str(video),
        run_dir=a.out,
        source_id="smoke",
        camera_mode="moving",
        frame_stop=a.frames,
        detector=DetectorConfig(backend=a.backend, confidence=a.confidence, compute=a.compute, max_inference_side=a.max_side,
                                extra={"fallback_model_id": f"{PROJECT}/2"} if a.backend == "roboflow_hosted" else {}),
    )
    res = process_video(cfg, progress=lambda i, n, m: print(f"  {i}/{n} {m}", flush=True) if i % 4 == 0 or i == n else None)
    s = res.summary
    print(json.dumps(s, indent=2))
    manifest = json.loads((Path(a.out) / "manifest.json").read_text())
    print("detector:", json.dumps(manifest["detector"]))
    print("compute:", json.dumps(manifest["compute"]))
    ok = True
    if s["detections_total"] == 0:
        print("FAIL: the model returned no detections", file=sys.stderr)
        ok = False
    tr = s["tracking"]
    if tr["tracks_total"] == 0 or tr["tracks_spanning_whole_range"] == 0:
        print("FAIL: no track persisted across the clip", file=sys.stderr)
        ok = False
    print("players per frame: %.1f, tracks: %d, spanning clip: %d, reappearances after gaps: %d" % (s["detections_per_frame"], tr["tracks_total"], tr["tracks_spanning_whole_range"], tr["reappearances_after_gap"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
