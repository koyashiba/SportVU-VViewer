"""Localhost REST + WebSocket adapter.

External research software pushes data while the app is running and pulls the
app's own tracking, using exactly the file schemas (docs/schemas.md):

  GET  /api/v1/info                      -> source + loaded data summary
  GET  /api/v1/schemas/{name}            -> JSON schema (boxes, ball, heatmap_meta)
  POST /api/v1/boxes                     <- one btrv.boxes record or a list (external boxes layer)
  POST /api/v1/ball                      <- one btrv.ball record or a list
  POST /api/v1/heatmap/meta              <- btrv.heatmap metadata (declares a live heat-map stream)
  POST /api/v1/heatmap/frame             <- {"frame_index", "timestamp_s", "grid": [[...]]}
  POST /api/v1/sportvu                   <- a SportVU document (top-down view source)
  GET  /api/v1/tracking/frames/{i}       -> the app's tracked boxes for frame i (btrv.boxes)
  GET  /api/v1/tracking/frames?start=&stop= -> list of records
  GET  /api/v1/tracking/raw/{i}          -> raw detector boxes for frame i
  GET  /api/v1/calibration               -> calibration document
  GET  /api/v1/state                     -> current frame index / timestamp of the viewer
  WS   /ws                               <- same JSON messages as the POST bodies, tagged {"type": "boxes"|"ball"|"heatmap_frame"|"heatmap_meta"|"sportvu"};
                                         -> broadcasts {"type": "state", ...} on frame changes and {"type": "tracking", ...} records on request {"type": "get_tracking", "frame_index": i}

The server runs in a background thread; the viewer polls :class:`DataHub`
through Qt signals, so the GUI thread never blocks on network I/O.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from btrv.core.types import FrameBoxes
from btrv.io.ball import BallRecord, parse_ball_record
from btrv.io.boxes import parse_frame_record
from btrv.io.errors import ImportValidationError
from btrv.io.heatmap import HeatmapMeta
from btrv.io.schemas import load_schema
from btrv.storage.boxes_table import BoxesTable


class LiveHeatmap:
    """In-memory heat-map stream received over the adapter (bounded ring buffer)."""

    def __init__(self, meta: HeatmapMeta, max_frames: int = 2000):
        self.meta = meta
        self.max_frames = max_frames
        self._grids: dict[int, np.ndarray] = {}
        self._order: list[int] = []

    def add(self, frame_index: int, grid: np.ndarray) -> None:
        if grid.shape != (self.meta.grid_height, self.meta.grid_width):
            raise ImportValidationError(f"grid shape {grid.shape} != declared ({self.meta.grid_height}, {self.meta.grid_width})")
        if frame_index not in self._grids:
            self._order.append(frame_index)
        self._grids[frame_index] = grid.astype(np.float32)
        while len(self._order) > self.max_frames:
            self._grids.pop(self._order.pop(0), None)

    def grid_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        return self._grids.get(frame_index)

    def has_frame(self, frame_index: int) -> bool:
        return frame_index in self._grids

    def __len__(self) -> int:
        return len(self._grids)


@dataclass
class DataHub:
    """Thread-safe shared state between the adapter and the viewer."""

    source_id: Optional[str] = None
    image_size: Optional[tuple[int, int]] = None
    frame_count: Optional[int] = None
    external_boxes: BoxesTable = field(default_factory=BoxesTable)
    ball: dict[int, BallRecord] = field(default_factory=dict)
    live_heatmap: Optional[LiveHeatmap] = None
    sportvu_doc: Optional[dict[str, Any]] = None
    tracking: Optional[BoxesTable] = None
    raw: Optional[BoxesTable] = None
    calibration_doc: Optional[dict[str, Any]] = None
    current_frame: int = 0
    current_timestamp: float = 0.0
    strict_source_check: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _listeners: list[Callable[[str, dict[str, Any]], None]] = field(default_factory=list)
    _version: int = 0

    def subscribe(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, kind: str, info: dict[str, Any]) -> None:
        self._version += 1
        for fn in list(self._listeners):
            try:
                fn(kind, info)
            except Exception:
                pass

    @property
    def version(self) -> int:
        return self._version

    def _check_source(self, sid: str, w: Optional[int] = None, h: Optional[int] = None) -> None:
        if self.strict_source_check and self.source_id is not None and sid != self.source_id:
            raise ImportValidationError(f"source_id {sid!r} does not match the loaded video {self.source_id!r}")
        if w is not None and self.image_size is not None and (w, h) != tuple(self.image_size):
            raise ImportValidationError(f"image size {w}x{h} does not match the video {self.image_size[0]}x{self.image_size[1]}")

    # -- ingest ------------------------------------------------------------
    def add_boxes(self, records: list[dict[str, Any]]) -> int:
        frames = [parse_frame_record(r, f"record {i}") for i, r in enumerate(records)]
        with self._lock:
            for fb in frames:
                self._check_source(fb.source_id, fb.image_width, fb.image_height)
                if self.frame_count is not None and fb.frame_index >= self.frame_count:
                    raise ImportValidationError(f"frame_index {fb.frame_index} is beyond the video's {self.frame_count} frames")
            for fb in frames:
                self.external_boxes.add_frame(fb)
            self.external_boxes.flush()
        self._notify("boxes", {"frames": len(frames)})
        return len(frames)

    def add_ball(self, records: list[dict[str, Any]]) -> int:
        recs = [parse_ball_record(r, f"record {i}") for i, r in enumerate(records)]
        with self._lock:
            for r in recs:
                self._check_source(r.source_id)
                self.ball[r.frame_index] = r
        self._notify("ball", {"records": len(recs)})
        return len(recs)

    def set_heatmap_meta(self, meta_obj: dict[str, Any]) -> HeatmapMeta:
        meta = HeatmapMeta.from_json(meta_obj, "heatmap meta")
        with self._lock:
            self._check_source(meta.source_id, meta.image_width if meta.space == "image_pixels" else None, meta.image_height if meta.space == "image_pixels" else None)
            self.live_heatmap = LiveHeatmap(meta)
        self._notify("heatmap_meta", {"grid": [meta.grid_height, meta.grid_width]})
        return meta

    def add_heatmap_frame(self, obj: dict[str, Any]) -> None:
        with self._lock:
            if self.live_heatmap is None:
                raise ImportValidationError("send heat-map metadata first (POST /api/v1/heatmap/meta or a 'heatmap_meta' message)")
            try:
                fi = int(obj["frame_index"])
                grid = np.asarray(obj["grid"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as e:
                raise ImportValidationError(f"heat-map frame needs integer 'frame_index' and a 2-D numeric 'grid': {e}") from e
            if not np.isfinite(grid).all() or grid.min() < 0 or grid.max() > 1.0001:
                raise ImportValidationError("heat-map grid values must be finite and normalized to [0, 1]")
            self.live_heatmap.add(fi, grid)
        self._notify("heatmap_frame", {"frame_index": fi})

    def set_sportvu(self, doc: dict[str, Any]) -> int:
        if not isinstance(doc, dict) or not isinstance(doc.get("events"), list) or not doc["events"]:
            raise ImportValidationError("SportVU document must have a non-empty 'events' list")
        with self._lock:
            self.sportvu_doc = doc
        n = len(doc["events"][0].get("moments", []))
        self._notify("sportvu", {"moments": n})
        return n

    # -- serve -------------------------------------------------------------
    def tracking_frame(self, i: int, raw: bool = False) -> Optional[dict[str, Any]]:
        with self._lock:
            table = self.raw if raw else self.tracking
            if table is None:
                return None
            fb = table.frame(i)
            return None if fb is None else fb.to_json()

    def set_state(self, frame_index: int, timestamp_s: float) -> None:
        self.current_frame, self.current_timestamp = int(frame_index), float(timestamp_s)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_id": self.source_id,
                "image_size": self.image_size,
                "frame_count": self.frame_count,
                "tracking_frames": 0 if self.tracking is None else len(self.tracking.frame_indices),
                "external_box_frames": len(self.external_boxes.frame_indices),
                "ball_records": len(self.ball),
                "live_heatmap_frames": 0 if self.live_heatmap is None else len(self.live_heatmap),
                "sportvu_loaded": self.sportvu_doc is not None,
                "schemas": {"boxes": "btrv.boxes 1.0", "ball": "btrv.ball 1.0", "heatmap": "btrv.heatmap 1.0"},
            }


def _as_list(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, dict):
        return [body]
    if isinstance(body, list):
        return body
    raise ImportValidationError("body must be a JSON object or a list of objects")


def create_app(hub: DataHub) -> FastAPI:
    app = FastAPI(title="btrv adapter", version="1.0")
    clients: set[WebSocket] = set()
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    @app.exception_handler(ImportValidationError)
    async def _ive(request, exc: ImportValidationError):
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": str(exc)})

    @app.get("/api/v1/info")
    def info():
        return hub.info()

    @app.get("/api/v1/state")
    def state():
        return {"frame_index": hub.current_frame, "timestamp_s": hub.current_timestamp}

    @app.get("/api/v1/schemas/{name}")
    def schema(name: str):
        files = {"boxes": "boxes.schema.json", "ball": "ball.schema.json", "heatmap_meta": "heatmap_meta.schema.json"}
        if name not in files:
            raise HTTPException(404, f"unknown schema {name}; one of {sorted(files)}")
        return load_schema(files[name])

    @app.post("/api/v1/boxes")
    def post_boxes(body: Any = Body(...)):
        return {"accepted_frames": hub.add_boxes(_as_list(body))}

    @app.post("/api/v1/ball")
    def post_ball(body: Any = Body(...)):
        return {"accepted_records": hub.add_ball(_as_list(body))}

    @app.post("/api/v1/heatmap/meta")
    def post_heatmap_meta(body: dict[str, Any]):
        m = hub.set_heatmap_meta(body)
        return {"ok": True, "grid_height": m.grid_height, "grid_width": m.grid_width}

    @app.post("/api/v1/heatmap/frame")
    def post_heatmap_frame(body: dict[str, Any]):
        hub.add_heatmap_frame(body)
        return {"ok": True}

    @app.post("/api/v1/sportvu")
    def post_sportvu(body: dict[str, Any]):
        return {"moments": hub.set_sportvu(body)}

    @app.get("/api/v1/tracking/frames/{i}")
    def tracking_frame(i: int):
        rec = hub.tracking_frame(i)
        if rec is None:
            raise HTTPException(404, f"no tracking for frame {i}")
        return rec

    @app.get("/api/v1/tracking/raw/{i}")
    def raw_frame(i: int):
        rec = hub.tracking_frame(i, raw=True)
        if rec is None:
            raise HTTPException(404, f"no raw boxes for frame {i}")
        return rec

    @app.get("/api/v1/tracking/frames")
    def tracking_range(start: int = Query(0), stop: int = Query(1 << 31), limit: int = Query(500, le=5000)):
        with hub._lock:
            if hub.tracking is None:
                return []
            fi = hub.tracking.frame_indices
            sel = fi[(fi >= start) & (fi < stop)][:limit]
            return [hub.tracking.frame(int(i)).to_json() for i in sel]

    @app.get("/api/v1/calibration")
    def calibration():
        if hub.calibration_doc is None:
            raise HTTPException(404, "no calibration loaded")
        return hub.calibration_doc

    async def _broadcast(msg: dict[str, Any]) -> None:
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    def _on_state(frame_index: int, timestamp_s: float) -> None:
        loop = loop_holder.get("loop")
        if loop is not None and clients:
            asyncio.run_coroutine_threadsafe(_broadcast({"type": "state", "frame_index": frame_index, "timestamp_s": timestamp_s}), loop)

    app.state.on_state = _on_state

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        loop_holder["loop"] = asyncio.get_running_loop()
        clients.add(ws)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                    kind = msg.get("type")
                    if kind == "boxes":
                        n = hub.add_boxes(_as_list(msg.get("data", msg)))
                        await ws.send_text(json.dumps({"type": "ack", "kind": kind, "accepted": n}))
                    elif kind == "ball":
                        n = hub.add_ball(_as_list(msg.get("data", msg)))
                        await ws.send_text(json.dumps({"type": "ack", "kind": kind, "accepted": n}))
                    elif kind == "heatmap_meta":
                        hub.set_heatmap_meta(msg.get("data", msg))
                        await ws.send_text(json.dumps({"type": "ack", "kind": kind}))
                    elif kind == "heatmap_frame":
                        hub.add_heatmap_frame(msg.get("data", msg))
                        await ws.send_text(json.dumps({"type": "ack", "kind": kind}))
                    elif kind == "sportvu":
                        n = hub.set_sportvu(msg.get("data", msg))
                        await ws.send_text(json.dumps({"type": "ack", "kind": kind, "moments": n}))
                    elif kind == "get_tracking":
                        rec = hub.tracking_frame(int(msg.get("frame_index", hub.current_frame)))
                        await ws.send_text(json.dumps({"type": "tracking", "data": rec}))
                    elif kind == "get_state":
                        await ws.send_text(json.dumps({"type": "state", "frame_index": hub.current_frame, "timestamp_s": hub.current_timestamp}))
                    else:
                        await ws.send_text(json.dumps({"type": "error", "detail": f"unknown message type {kind!r}"}))
                except ImportValidationError as e:
                    await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    await ws.send_text(json.dumps({"type": "error", "detail": f"malformed message: {e}"}))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    return app


class AdapterServer:
    """Runs uvicorn in a daemon thread bound to localhost."""

    def __init__(self, hub: DataHub, host: str = "127.0.0.1", port: int = 8765):
        self.hub = hub
        self.host, self.port = host, port
        self.app = create_app(hub)
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", loop="asyncio")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="btrv-adapter", daemon=True)
        self._thread.start()
        for _ in range(100):
            if getattr(self._server, "started", False):
                return
            time.sleep(0.05)

    def broadcast_state(self, frame_index: int, timestamp_s: float) -> None:
        self.hub.set_state(frame_index, timestamp_s)
        try:
            self.app.state.on_state(frame_index, timestamp_s)
        except Exception:
            pass

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
