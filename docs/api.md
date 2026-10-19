# Localhost adapter (REST + WebSocket)

Started automatically by the viewer (`Adapter` menu, default `http://127.0.0.1:8765`,
`--port` to change, `--no-adapter` to disable). Bound to localhost only. Payloads are exactly the
file schemas in `schemas.md`; validation errors return HTTP 422 with `{"error": "validation_error",
"detail": "..."}` (or `{"type": "error", "detail": ...}` on the WebSocket).

| method | path | body / result |
|---|---|---|
| GET | `/api/v1/info` | source id, image size, frame count, loaded data counts |
| GET | `/api/v1/state` | `{"frame_index", "timestamp_s"}` currently displayed |
| GET | `/api/v1/schemas/{boxes\|ball\|heatmap_meta}` | JSON schema |
| POST | `/api/v1/boxes` | one `btrv.boxes` record or a list → shown as the *external boxes* layer |
| POST | `/api/v1/ball` | one `btrv.ball` record or a list |
| POST | `/api/v1/heatmap/meta` | `btrv.heatmap` metadata object (starts a live heat-map stream) |
| POST | `/api/v1/heatmap/frame` | `{"frame_index": i, "timestamp_s": t, "grid": [[...]]}` normalised values |
| POST | `/api/v1/sportvu` | a SportVU document → court view source |
| GET | `/api/v1/tracking/frames/{i}` | the app's tracked boxes for frame *i* (`btrv.boxes`, with court points) |
| GET | `/api/v1/tracking/frames?start=&stop=&limit=` | list of records |
| GET | `/api/v1/tracking/raw/{i}` | raw detector boxes for frame *i* |
| GET | `/api/v1/calibration` | `btrv.calibration` document of the loaded run |
| WS | `/ws` | JSON messages `{"type": "boxes"\|"ball"\|"heatmap_meta"\|"heatmap_frame"\|"sportvu", "data": ...}`; replies `{"type":"ack"}`; `{"type":"get_tracking","frame_index":i}` → `{"type":"tracking","data":...}`; `{"type":"get_state"}`; the server broadcasts `{"type":"state","frame_index","timestamp_s"}` whenever the displayed frame changes |

Data pushed while the app runs appears on the next repaint of the affected frame; the viewer keeps a
bounded in-memory buffer for live heat maps (last 2000 frames) and all pushed boxes/ball records.

Python example:

```python
import requests, json
base = "http://127.0.0.1:8765/api/v1"
info = requests.get(f"{base}/info").json()
rec = {"schema": "btrv.boxes", "schema_version": "1.0", "source_id": info["source_id"], "frame_index": 10,
       "timestamp_s": 0.4, "image_width": info["image_size"][0], "image_height": info["image_size"][1],
       "coordinate_convention": "image_pixels_xyxy_topleft",
       "boxes": [{"xyxy": [100, 200, 160, 360], "class_name": "players", "confidence": 0.9, "track_id": 3}]}
print(requests.post(f"{base}/boxes", json=rec).json())
print(requests.get(f"{base}/tracking/frames/10").json())
```
