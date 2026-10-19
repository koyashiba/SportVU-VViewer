import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("BTRV_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache"))

FFMPEG = shutil.which("ffmpeg")


def make_marker_video(path: Path, n_frames: int = 60, width: int = 320, height: int = 240, rate: str = "25", bframes: int = 3, gop: int = 12, extra: list[str] | None = None) -> Path:
    """Encode a video whose frame i carries a white 8px block at x = 8*i (lossy-safe marker)."""
    frames = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        frames[i, :, :, :] = (20, 60, 20)
        x = (8 * i) % (width - 8)
        row = (i * 8 // (width - 8)) * 16 % (height - 16)
        frames[i, row : row + 16, x : x + 8, :] = 255
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", rate, "-i", "-",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-bf", str(bframes), "-g", str(gop), "-pix_fmt", "yuv420p"] + (extra or []) + [str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path


def marker_index(frame: np.ndarray, width: int = 320, height: int = 240) -> int:
    """Recover the frame index from the marker block position."""
    mask = frame.min(axis=2) > 180
    ys, xs = np.nonzero(mask)
    x = int(np.median(xs)) // 8 * 8
    row = (int(np.median(ys)) // 16) * 16
    per_row = (width - 8) // 8
    return (row // 16) * per_row + x // 8


@pytest.fixture(scope="session")
def marker_video(tmp_path_factory) -> Path:
    if not FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("video")
    return make_marker_video(d / "marker.mp4")
