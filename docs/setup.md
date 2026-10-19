# Setup and platform notes

## Requirements

* Python 3.10–3.13, pip. The PyAV wheel bundles FFmpeg; `ffmpeg` on PATH is only needed to run the
  test suite (it synthesises test videos).
* Linux: Qt runtime libraries — Debian/Ubuntu: `sudo apt install libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1`.
  Wayland sessions may need `QT_QPA_PLATFORM=xcb`.
* macOS 12+: nothing extra; Apple Silicon uses the MPS backend automatically when torch is installed.
* Windows 10/11: nothing extra (PySide6 wheels ship the Qt runtime). Use `py -3 -m venv .venv`.

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[local-inference]"      # optional, for the local RF-DETR backend
```

### Compute backend

| backend | how to get it | select |
|---|---|---|
| CUDA | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` before `[local-inference]` | `--compute cuda` or auto |
| MPS | default torch wheel on Apple Silicon | `--compute mps` or auto |
| CPU | default | `--compute cpu` or auto fallback |

`btrv.core.compute.resolve_backend` logs why a device was chosen; the result is stored in the run
manifest (`compute.requested/resolved/reason`). The behaviour on each accelerator has **not** been
measured in this repository's environment (CPU-only container); see `verification.md`.

### Roboflow key

`ROBOFLOW_API_KEY` in the environment or in a git-ignored `.env` (load it with your shell or
`python-dotenv`). Never commit it. The local backend downloads the pinned weights once into the
inference-models cache and runs offline afterwards.

## Running

* `btrv-viewer [video] [--run RUN_DIR] [--port 8765] [--no-adapter]`
* `btrv process ...`, `btrv export-sportvu ...`, `btrv export-boxes ...`, `btrv calibrate-auto ...`,
  `btrv info ...`, `btrv make-examples ...`

Caches: the frame index is cached under `$BTRV_CACHE_DIR/frame_index` (default `~/.cache/btrv` or
`%LOCALAPPDATA%\btrv`), keyed by path, size and mtime.

## Tests

```bash
pytest -q            # ~2-3 minutes on a 4-core CPU; Qt tests run offscreen automatically
```
