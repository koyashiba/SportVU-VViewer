"""btrv.heatmap 1.0: per-frame probability grids in an array-backed file.

Two containers share one logical layout (see docs/schemas.md):

* ``.h5`` (recommended, random access, chunked per frame, optional gzip) --
  root attributes hold the metadata; datasets ``frame_index`` [N] int64,
  ``timestamp_s`` [N] float64, ``grid`` [N, H, W] float16/float32.
* ``.npz`` (small clips) -- members ``meta`` (JSON string), ``frame_index``,
  ``timestamp_s``, ``grid``.

Only the current frame's grid is read when rendering; the file is never loaded
entirely for the HDF5 container.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from btrv.core.types import HEATMAP_SCHEMA, HEATMAP_SCHEMA_VERSION
from .errors import ImportValidationError
from .schemas import validate

SPACES = ("image_pixels", "court_feet")
NORMALIZATIONS = ("sum_to_one", "max_one", "unit_range")


@dataclass
class HeatmapMeta:
    source_id: str
    space: str
    extent: list[float]  # [x_min, y_min, x_max, y_max]
    grid_width: int
    grid_height: int
    value_kind: str = "probability"
    normalization: str = "sum_to_one"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    producer: Optional[str] = None
    dtype: str = "float32"
    schema: str = HEATMAP_SCHEMA
    schema_version: str = HEATMAP_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra") or {}
        d = {k: v for k, v in d.items() if v is not None}
        d.update(extra)
        return d

    @staticmethod
    def from_json(d: dict[str, Any], context: str = "") -> "HeatmapMeta":
        validate(d, "heatmap_meta.schema.json", context)
        known = {f for f in HeatmapMeta.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["extent"] = [float(v) for v in d["extent"]]
        extra = {k: v for k, v in d.items() if k not in known}
        return HeatmapMeta(extra=extra, **kwargs)


def _check_grid(grid: np.ndarray, meta: HeatmapMeta, context: str) -> None:
    if grid.ndim != 3:
        raise ImportValidationError(f"{context}: grid must be [N, H, W], got shape {grid.shape}")
    if grid.shape[1] != meta.grid_height or grid.shape[2] != meta.grid_width:
        raise ImportValidationError(f"{context}: grid shape {grid.shape[1:]} != (grid_height, grid_width) = ({meta.grid_height}, {meta.grid_width})")
    x0, y0, x1, y1 = meta.extent
    if not (x1 > x0 and y1 > y0):
        raise ImportValidationError(f"{context}: extent must satisfy x_max > x_min and y_max > y_min, got {meta.extent}")


def validate_normalization(grid: np.ndarray, meta: HeatmapMeta, context: str = "", tol: float = 1e-2, sample: int = 16) -> None:
    n = grid.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, sample)).astype(int)) if n else []
    for i in idx:
        g = np.asarray(grid[i], dtype=np.float64)
        if not np.isfinite(g).all():
            raise ImportValidationError(f"{context}: grid {i} contains NaN/inf")
        if g.min() < -tol or g.max() > 1 + tol:
            raise ImportValidationError(f"{context}: grid {i} has values outside [0, 1] (min {g.min():.3g}, max {g.max():.3g}); values must be normalized")
        if meta.normalization == "sum_to_one" and g.sum() > 0 and abs(g.sum() - 1) > tol * 10:
            raise ImportValidationError(f"{context}: grid {i} sums to {g.sum():.4f} but normalization is 'sum_to_one'")
        if meta.normalization == "max_one" and g.max() > 0 and abs(g.max() - 1) > tol:
            raise ImportValidationError(f"{context}: grid {i} has max {g.max():.4f} but normalization is 'max_one'")


class HeatmapWriter:
    """Streaming writer; call ``append`` per frame then ``close``."""

    def __init__(self, path: str | os.PathLike, meta: HeatmapMeta, compression: Optional[str] = "gzip", chunk_frames: int = 1):
        self.path = Path(path)
        self.meta = meta
        self._npz = self.path.suffix.lower() == ".npz"
        self._dtype = np.dtype(meta.dtype)
        self._fi: list[int] = []
        self._ts: list[float] = []
        if self._npz:
            self._grids: list[np.ndarray] = []
        else:
            import h5py

            self._f = h5py.File(self.path, "w")
            for k, v in meta.to_json().items():
                self._f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            self._f.attrs["extent"] = np.asarray(meta.extent, dtype=np.float64)
            H, W = meta.grid_height, meta.grid_width
            self._g = self._f.create_dataset("grid", shape=(0, H, W), maxshape=(None, H, W), dtype=self._dtype, chunks=(chunk_frames, H, W), compression=compression)
            self._dfi = self._f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
            self._dts = self._f.create_dataset("timestamp_s", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))

    def append(self, frame_index: int, timestamp_s: float, grid: np.ndarray) -> None:
        grid = np.asarray(grid, dtype=self._dtype)
        if grid.shape != (self.meta.grid_height, self.meta.grid_width):
            raise ValueError(f"grid shape {grid.shape} != ({self.meta.grid_height}, {self.meta.grid_width})")
        self._fi.append(int(frame_index))
        self._ts.append(float(timestamp_s))
        if self._npz:
            self._grids.append(grid)
        else:
            n = self._g.shape[0]
            self._g.resize(n + 1, axis=0)
            self._g[n] = grid
            self._dfi.resize(n + 1, axis=0)
            self._dfi[n] = frame_index
            self._dts.resize(n + 1, axis=0)
            self._dts[n] = timestamp_s

    def close(self) -> None:
        if self._npz:
            grid = np.stack(self._grids) if self._grids else np.zeros((0, self.meta.grid_height, self.meta.grid_width), dtype=self._dtype)
            np.savez_compressed(self.path, meta=json.dumps(self.meta.to_json()), frame_index=np.asarray(self._fi, dtype=np.int64), timestamp_s=np.asarray(self._ts, dtype=np.float64), grid=grid)
        else:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HeatmapReader:
    """Random access by frame index; HDF5 reads one grid at a time."""

    def __init__(self, path: str | os.PathLike, strict: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise ImportValidationError(f"heat-map file not found: {self.path}")
        ctx = self.path.name
        self._npz = self.path.suffix.lower() == ".npz"
        if self._npz:
            try:
                z = np.load(self.path, allow_pickle=False)
            except Exception as e:
                raise ImportValidationError(f"{ctx}: not a readable .npz file: {e}") from e
            for k in ("meta", "frame_index", "timestamp_s", "grid"):
                if k not in z.files:
                    raise ImportValidationError(f"{ctx}: missing member '{k}' (expected meta, frame_index, timestamp_s, grid)")
            try:
                meta_d = json.loads(str(z["meta"]))
            except Exception as e:
                raise ImportValidationError(f"{ctx}: 'meta' must be a JSON string: {e}") from e
            self.meta = HeatmapMeta.from_json(meta_d, ctx)
            self._fi = np.asarray(z["frame_index"], dtype=np.int64)
            self._ts = np.asarray(z["timestamp_s"], dtype=np.float64)
            self._grid = z["grid"]
        else:
            import h5py

            try:
                self._f = h5py.File(self.path, "r")
            except Exception as e:
                raise ImportValidationError(f"{ctx}: not a readable HDF5 file: {e}") from e
            for k in ("frame_index", "timestamp_s", "grid"):
                if k not in self._f:
                    raise ImportValidationError(f"{ctx}: missing dataset '{k}' (expected frame_index, timestamp_s, grid)")
            meta_d: dict[str, Any] = {}
            for k, v in self._f.attrs.items():
                if isinstance(v, bytes):
                    v = v.decode()
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, str) and v[:1] in "[{":
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                meta_d[k] = v
            self.meta = HeatmapMeta.from_json(meta_d, ctx)
            self._fi = np.asarray(self._f["frame_index"][...], dtype=np.int64)
            self._ts = np.asarray(self._f["timestamp_s"][...], dtype=np.float64)
            self._grid = self._f["grid"]
        if self._grid.shape[0] != self._fi.shape[0] or self._ts.shape[0] != self._fi.shape[0]:
            raise ImportValidationError(f"{ctx}: frame_index ({self._fi.shape[0]}), timestamp_s ({self._ts.shape[0]}) and grid ({self._grid.shape[0]}) lengths differ")
        _check_grid(self._grid, self.meta, ctx)
        if self._fi.size and (np.any(np.diff(self._fi) <= 0)):
            raise ImportValidationError(f"{ctx}: frame_index must be strictly increasing (sorted, unique)")
        if strict:
            validate_normalization(self._grid, self.meta, ctx)
        self._lookup = {int(f): i for i, f in enumerate(self._fi)}

    def __len__(self) -> int:
        return int(self._fi.shape[0])

    @property
    def frame_indices(self) -> np.ndarray:
        return self._fi

    @property
    def timestamps(self) -> np.ndarray:
        return self._ts

    def has_frame(self, frame_index: int) -> bool:
        return frame_index in self._lookup

    def grid_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Exactly this frame's grid as float32 [H, W], or None if absent."""
        i = self._lookup.get(int(frame_index))
        if i is None:
            return None
        return np.asarray(self._grid[i], dtype=np.float32)

    def check_alignment(self, source_id: str, timestamps: Optional[np.ndarray], frame_count: Optional[int], image_size: Optional[tuple[int, int]] = None, tol_s: float = 0.02) -> dict[str, Any]:
        if self.meta.source_id != source_id:
            raise ImportValidationError(f"source_id mismatch: heat map is for {self.meta.source_id!r}, loaded video is {source_id!r}")
        if self.meta.space == "image_pixels" and image_size is not None and self.meta.image_width and self.meta.image_height:
            if (self.meta.image_width, self.meta.image_height) != tuple(image_size):
                raise ImportValidationError(f"heat map declares image {self.meta.image_width}x{self.meta.image_height} but the video is {image_size[0]}x{image_size[1]}")
        if frame_count is not None and self._fi.size and int(self._fi.max()) >= frame_count:
            raise ImportValidationError(f"heat map references frame {int(self._fi.max())} but the video has {frame_count} frames")
        mism = 0
        if timestamps is not None and self._fi.size:
            mism = int(np.sum(np.abs(np.asarray(timestamps)[self._fi] - self._ts) > tol_s))
        return {"frames": int(self._fi.size), "timestamp_mismatches": mism}

    def close(self) -> None:
        if not self._npz:
            try:
                self._f.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_heatmap_overlay(grid: np.ndarray, meta: HeatmapMeta, out_size: tuple[int, int], H_court_to_image: Optional[np.ndarray] = None, colormap: int = 2) -> np.ndarray:
    """Return a BGRA uint8 image (out_size = (width, height) of the source frame) with the grid
    placed according to its extent; alpha follows the (max-normalised) value."""
    import cv2

    w, h = out_size
    g = np.asarray(grid, dtype=np.float32)
    m = float(g.max()) if g.size else 0.0
    norm = (g / m) if m > 0 else g
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    alpha = (norm * 255).astype(np.uint8)
    bgra = np.dstack([colour, alpha])
    x0, y0, x1, y1 = meta.extent
    if meta.space == "image_pixels":
        gw, gh = meta.grid_width, meta.grid_height
        src = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]])
        dst = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgra, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if H_court_to_image is None:
        raise ValueError("a court-space heat map needs a calibration to be drawn on the video")
    # grid pixel -> court feet -> image
    gw, gh = meta.grid_width, meta.grid_height
    A = np.array([[(x1 - x0) / gw, 0, x0], [0, (y1 - y0) / gh, y0], [0, 0, 1]], dtype=np.float64)
    M = np.asarray(H_court_to_image, dtype=np.float64) @ A
    return cv2.warpPerspective(bgra, M.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
