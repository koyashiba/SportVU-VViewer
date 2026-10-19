# Interchange schemas

All schemas are versioned (`schema`, `schema_version`), aligned by `source_id`, `frame_index` and
`timestamp_s` (see `coordinates.md`), and used identically for files and for the localhost adapter
(`api.md`). JSON schemas live in `src/btrv/schemas/` and are served by `GET /api/v1/schemas/{name}`.
Example files: `examples/` (regenerate with `btrv make-examples examples`).

## `btrv.boxes` 1.0 — per-frame bounding boxes

One JSON object per **frame**. Files are JSON Lines (`.jsonl`, one object per line, streamable) or a
JSON array of the same objects. The same object is a WebSocket/REST payload.

```json
{"schema": "btrv.boxes", "schema_version": "1.0",
 "source_id": "game01", "frame_index": 1234, "timestamp_s": 41.1333,
 "image_width": 3840, "image_height": 2160,
 "coordinate_convention": "image_pixels_xyxy_topleft",
 "producer": "my-research-tool 0.3",
 "calibration": {"valid": true, "confidence": 0.91, "method": "manual"},
 "boxes": [
   {"xyxy": [1510.2, 820.0, 1588.7, 1010.4], "class_name": "players", "class_id": 0,
    "confidence": 0.87, "track_id": 12,
    "team": {"id": 0, "confidence": 0.93, "method": "jersey_color_kmeans"},
    "court": {"x": 61.2, "y": 18.7, "valid": true, "confidence": 0.91},
    "foot_xy": [1549.4, 1010.4]}
 ]}
```

| field | required | notes |
|---|---|---|
| `source_id` | yes | must equal the loaded video's id (file stem by default, or `--source-id`) |
| `frame_index`, `timestamp_s` | yes | presentation-order index and pts-derived seconds |
| `image_width/height` | yes | full source frame size; must match the video |
| `coordinate_convention` | yes | always `image_pixels_xyxy_topleft` |
| `boxes[].xyxy` | yes | `[x1,y1,x2,y2]` source pixels |
| `boxes[].class_name` | yes | `players` for the pinned model; external tools may send `referee`, `ball`, ... |
| `boxes[].confidence` | yes | `[0,1]` |
| `boxes[].track_id` | no | integer, `null` when untracked |
| `boxes[].team` | no | `id` 0/1 or `null` (undecided) with explicit `confidence` |
| `boxes[].court` | no | ground-contact point in `sportvu_feet_v1`; `valid:false` = unmapped/off-court |
| `boxes[].foot_xy` | no | explicit ground-contact pixel; default bottom-centre |
| `calibration` | no | frame-level calibration state written by btrv |

btrv writes two such files per run: `raw_boxes.jsonl` (detector output, no ids) and
`tracked_boxes.jsonl` (ids, teams, court points). Import validation rejects: wrong schema/version,
wrong convention, missing fields, confidence outside `[0,1]`, non-positive boxes, `source_id` or
image-size mismatch, duplicate frames and frames beyond the video. Timestamp mismatches
(> 20 ms) are counted and reported.

## `btrv.heatmap` 1.0 — per-frame ball-location heat map

Array-backed container with a metadata block and three aligned arrays:

| member | type | meaning |
|---|---|---|
| metadata | attrs / JSON | see below |
| `frame_index` | int64 `[N]` | strictly increasing |
| `timestamp_s` | float64 `[N]` | source timestamps |
| `grid` | float16/float32 `[N, grid_height, grid_width]` | row 0 = `y_min` edge, column 0 = `x_min` edge |

Metadata (`heatmap_meta.schema.json`):

```json
{"schema": "btrv.heatmap", "schema_version": "1.0", "source_id": "game01",
 "space": "image_pixels",            // or "court_feet"
 "extent": [0, 0, 3840, 2160],        // [x_min, y_min, x_max, y_max] in the declared space
 "grid_width": 192, "grid_height": 108,   // independent of the video resolution
 "value_kind": "probability",         // or "confidence"
 "normalization": "sum_to_one",      // or "max_one" | "unit_range"
 "image_width": 3840, "image_height": 2160,   // required when space == image_pixels
 "producer": "ball-predictor 1.2", "dtype": "float16"}
```

Containers:

* **HDF5 `.h5`** (recommended for long videos): root attributes = metadata (`extent` as a float array),
  datasets `frame_index`, `timestamp_s`, `grid` chunked `(1, H, W)` with gzip. Only the current frame's
  chunk is read when rendering. 100k frames × 108×192 float16 ≈ 4 GB uncompressed on disk, nothing in RAM.
* **NumPy `.npz`** (short clips): members `meta` (JSON string), `frame_index`, `timestamp_s`, `grid`.

Writing from Python research code:

```python
import h5py, json, numpy as np
meta = {"schema": "btrv.heatmap", "schema_version": "1.0", "source_id": "game01", "space": "image_pixels",
        "extent": [0, 0, 3840, 2160], "grid_width": 192, "grid_height": 108,
        "value_kind": "probability", "normalization": "sum_to_one", "image_width": 3840, "image_height": 2160}
with h5py.File("ball_heatmap.h5", "w") as f:
    for k, v in meta.items():
        f.attrs[k] = np.asarray(v, dtype=np.float64) if k == "extent" else v
    g = f.create_dataset("grid", shape=(0, 108, 192), maxshape=(None, 108, 192), dtype="float16", chunks=(1, 108, 192), compression="gzip")
    fi = f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="int64")
    ts = f.create_dataset("timestamp_s", shape=(0,), maxshape=(None,), dtype="float64")
    for frame_index, timestamp_s, grid in my_predictions():   # grid: (108, 192), sums to 1
        n = g.shape[0]
        g.resize(n + 1, axis=0); fi.resize(n + 1, axis=0); ts.resize(n + 1, axis=0)
        g[n], fi[n], ts[n] = grid, frame_index, timestamp_s
```

or simply use `btrv.io.heatmap.HeatmapWriter` (same layout, `.h5` or `.npz`).
Validation on import: member presence, `[N,H,W]` shape vs. declared grid size, valid extent, sorted
unique frame indices, values in `[0,1]`, per-frame normalisation (sampled), `source_id`, image size and
frame range against the loaded video. Rendering: the grid is warped to its extent (image space) or
through the frame's calibration (court space) and alpha-blended with the chosen opacity; only the
grid of the **displayed** frame is drawn, frames without a grid show nothing.

Live streaming of the same data: `POST /api/v1/heatmap/meta` once, then `POST /api/v1/heatmap/frame`
with `{"frame_index": i, "timestamp_s": t, "grid": [[...]]}` per frame (or the WebSocket equivalents).

## `btrv.ball` 1.0 — externally supplied ball coordinates (court feet)

```json
{"schema": "btrv.ball", "schema_version": "1.0", "source_id": "game01", "frame_index": 1234,
 "timestamp_s": 41.1333, "coordinate_convention": "sportvu_feet_v1", "x": 47.3, "y": 24.1, "z": 7.2, "confidence": 0.8}
```
`z` may be `null` (the export then uses the documented default, see `sportvu.md`). JSON Lines or JSON
array. Required for a conforming SportVU export; also drawn in the top-down view.

## `btrv.calibration` 1.0

```json
{"schema": "btrv.calibration", "schema_version": "1.0", "source_id": "game01",
 "image_width": 3840, "image_height": 2160, "camera_mode": "fixed", "roi": [120, 400, 3700, 2100],
 "court": {"name": "nba", "convention": "sportvu_feet_v1", "length_ft": 94, "width_ft": 50, ...},
 "homography_direction": "image_pixels -> court_feet (x' = H x, homogeneous)",
 "keyframes": [{"H": [[...],[...],[...]], "method": "manual", "confidence": 0.92, "valid": true,
                "frame_index": 0, "reprojection_error_px": 2.1, "inlier_ratio": 1.0,
                "landmarks": [{"name": "corner_top_left", "image_xy": [..], "court_xy": [0, 0], "inlier": true, "error_px": 1.4}]}]}
```
For moving cameras several keyframes may be given; the worker propagates between them.

## `btrv.run_manifest` 1.0 (`runs/<id>/manifest.json`)

Stores, for reproducibility: exact `detector.model_id`, runtime and device, tracker library and
parameters, team-assignment parameters, `camera_mode`, `roi`, `inference_region` (crop + scale),
court model, calibration (file + keyframes + any automatic result), the full processing config,
compute backend (requested/resolved/reason), coordinate conventions, software versions
(btrv, Python, numpy, OpenCV, PyAV, supervision, trackers, inference-models, torch, PySide6),
source info (uri, size, frame count, time base, first pts, codec) and the run summary.
