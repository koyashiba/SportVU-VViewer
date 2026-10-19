"""File-based frame source built on PyAV.

Design points:

* A :class:`FrameIndex` maps presentation-order frame indices to presentation
  timestamps (pts) and keyframe flags.  It is built by demuxing packets (no
  decoding), sorted by pts, and cached on disk keyed by path/size/mtime.
* ``read(i)`` is frame accurate: it seeks to the last keyframe at or before the
  target pts and decodes forward until the decoded frame's pts equals the
  target.  Sequential reads decode a single frame.
* Decoded frames are kept in a byte-budgeted LRU cache so that stepping
  backwards inside a GOP does not re-decode the GOP every time.  Frames are
  decoded on demand; the file is never loaded into memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
import numpy as np

from btrv.core.types import FrameInfo
from .base import FrameSource, SourceInfo

INDEX_VERSION = 1


@dataclass
class FrameIndex:
    pts: np.ndarray  # int64 [N], presentation order (sorted ascending)
    keyframe: np.ndarray  # bool [N]
    time_base: Fraction
    width: int
    height: int
    codec: str
    average_rate: Optional[float]

    @property
    def frame_count(self) -> int:
        return int(self.pts.shape[0])

    def timestamp_s(self, frame_index: int) -> float:
        return float(self.pts[frame_index] * self.time_base.numerator / self.time_base.denominator)

    def timestamps_s(self) -> np.ndarray:
        return self.pts.astype(np.float64) * (self.time_base.numerator / self.time_base.denominator)

    def frame_for_timestamp(self, t_s: float) -> int:
        """Index of the frame displayed at time t (last frame with timestamp <= t)."""
        ts = self.timestamps_s()
        i = int(np.searchsorted(ts, t_s, side="right") - 1)
        return max(0, min(i, self.frame_count - 1))

    def keyframe_at_or_before(self, frame_index: int) -> int:
        kf = np.flatnonzero(self.keyframe[: frame_index + 1])
        return int(kf[-1]) if kf.size else 0

    def to_json(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "pts": self.pts.tolist(),
            "keyframe": self.keyframe.astype(np.uint8).tolist(),
            "time_base": [self.time_base.numerator, self.time_base.denominator],
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "average_rate": self.average_rate,
        }

    @staticmethod
    def from_json(d: dict) -> "FrameIndex":
        return FrameIndex(
            pts=np.asarray(d["pts"], dtype=np.int64),
            keyframe=np.asarray(d["keyframe"], dtype=bool),
            time_base=Fraction(int(d["time_base"][0]), int(d["time_base"][1])),
            width=int(d["width"]),
            height=int(d["height"]),
            codec=str(d["codec"]),
            average_rate=d.get("average_rate"),
        )


def _cache_key(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{INDEX_VERSION}".encode()).hexdigest()
    return h


def default_index_cache_dir() -> Path:
    base = os.environ.get("BTRV_CACHE_DIR")
    if base:
        return Path(base) / "frame_index"
    home = Path.home()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    return root / "btrv" / "frame_index"


def build_frame_index(path: str | os.PathLike, cache_dir: Optional[Path] = None, use_cache: bool = True) -> FrameIndex:
    """Demux the file once (no decoding) and build the presentation-order index."""
    path = Path(path)
    cache_file = None
    if use_cache:
        cache_dir = cache_dir or default_index_cache_dir()
        cache_file = cache_dir / f"{_cache_key(path)}.json"
        if cache_file.exists():
            try:
                return FrameIndex.from_json(json.loads(cache_file.read_text()))
            except Exception:
                pass
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        pts_list: list[int] = []
        key_list: list[bool] = []
        for packet in container.demux(stream):
            if packet.size == 0 and packet.pts is None and packet.dts is None:
                continue  # flush packet
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                continue
            pts_list.append(int(ts))
            key_list.append(bool(packet.is_keyframe))
        codec = stream.codec_context.name
        width, height = stream.codec_context.width, stream.codec_context.height
        avg = float(stream.average_rate) if stream.average_rate else None
    pts = np.asarray(pts_list, dtype=np.int64)
    keyframe = np.asarray(key_list, dtype=bool)
    order = np.argsort(pts, kind="stable")
    pts, keyframe = pts[order], keyframe[order]
    # Duplicate pts (broken files) would make seeking ambiguous; keep the first.
    if pts.size > 1:
        keep = np.concatenate([[True], np.diff(pts) > 0])
        pts, keyframe = pts[keep], keyframe[keep]
    index = FrameIndex(pts, keyframe, Fraction(time_base), width, height, codec, avg)
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(index.to_json()))
        except OSError:
            pass
    return index


class _FrameCache:
    """Byte-budgeted LRU cache of decoded frames."""

    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(self, i: int) -> Optional[np.ndarray]:
        f = self._d.get(i)
        if f is not None:
            self._d.move_to_end(i)
        return f

    def put(self, i: int, frame: np.ndarray) -> None:
        if i in self._d:
            self._d.move_to_end(i)
            return
        nb = frame.nbytes
        if nb > self.budget:
            return
        while self._bytes + nb > self.budget and self._d:
            _, old = self._d.popitem(last=False)
            self._bytes -= old.nbytes
        self._d[i] = frame
        self._bytes += nb

    def clear(self) -> None:
        self._d.clear()
        self._bytes = 0


class FileSource(FrameSource):
    """Frame-accurate random access to a video file."""

    def __init__(
        self,
        path: str | os.PathLike,
        source_id: Optional[str] = None,
        cache_budget_bytes: int = 1_500_000_000,
        index_cache_dir: Optional[Path] = None,
        use_index_cache: bool = True,
        thread_type: str = "AUTO",
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self.index = build_frame_index(self.path, index_cache_dir, use_index_cache)
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = thread_type
        self._decoder = None  # generator over decoded frames
        self._next_index: Optional[int] = None  # presentation index the decoder will yield next (if known)
        self._cache = _FrameCache(cache_budget_bytes)
        self._lock = threading.RLock()
        self._pts_to_index = {int(p): i for i, p in enumerate(self.index.pts)}
        sid = source_id or self.path.stem
        tb = self.index.time_base
        dur = None
        if self.index.frame_count:
            dur = self.index.timestamp_s(self.index.frame_count - 1) - self.index.timestamp_s(0)
        self._info = SourceInfo(
            source_id=sid,
            uri=str(self.path.resolve()),
            width=self.index.width,
            height=self.index.height,
            frame_count=self.index.frame_count,
            time_base=tb.numerator / tb.denominator,
            average_rate=self.index.average_rate,
            codec=self.index.codec,
            duration_s=dur,
            seekable=True,
            extra={"time_base_fraction": [tb.numerator, tb.denominator], "first_pts": int(self.index.pts[0]) if self.index.frame_count else None},
        )

    # -- FrameSource API -------------------------------------------------
    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame_info(self, frame_index: int) -> FrameInfo:
        self._check(frame_index)
        return FrameInfo(frame_index, self.index.timestamp_s(frame_index), self.index.width, self.index.height)

    def read(self, frame_index: int) -> tuple[FrameInfo, np.ndarray]:
        self._check(frame_index)
        with self._lock:
            cached = self._cache.get(frame_index)
            if cached is None:
                cached = self._decode_to(frame_index)
            return self.frame_info(frame_index), cached

    def next(self) -> Optional[tuple[FrameInfo, np.ndarray]]:
        with self._lock:
            i = self._next_index if self._next_index is not None else 0
            if i >= self.index.frame_count:
                return None
            return self.read(i)

    def close(self) -> None:
        with self._lock:
            self._decoder = None
            try:
                self._container.close()
            except Exception:
                pass
            self._cache.clear()

    # -- internals ---------------------------------------------------------
    def _check(self, i: int) -> None:
        if not (0 <= i < self.index.frame_count):
            raise IndexError(f"frame index {i} out of range [0, {self.index.frame_count})")

    def _frame_pts(self, frame) -> Optional[int]:
        pts = frame.pts
        if pts is None:
            pts = getattr(frame, "dts", None)
        return None if pts is None else int(pts)

    def _restart_decoder_at(self, frame_index: int) -> None:
        kf = self.index.keyframe_at_or_before(frame_index)
        target_pts = int(self.index.pts[kf])
        self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
        self._decoder = self._container.decode(self._stream)
        self._next_index = None  # unknown until the first frame is decoded

    def _decode_to(self, frame_index: int) -> np.ndarray:
        target_pts = int(self.index.pts[frame_index])
        need_seek = self._decoder is None or self._next_index is None or frame_index < self._next_index or frame_index - self._next_index > 400
        if need_seek:
            self._restart_decoder_at(frame_index)
        while True:
            try:
                frame = next(self._decoder)  # type: ignore[arg-type]
            except StopIteration:
                if not need_seek:
                    need_seek = True
                    self._restart_decoder_at(frame_index)
                    continue
                raise RuntimeError(f"could not decode frame {frame_index} (pts {target_pts}) from {self.path}")
            pts = self._frame_pts(frame)
            if pts is None:
                continue
            idx = self._pts_to_index.get(pts)
            if idx is None:
                continue  # frame not in the index (should not happen); skip
            self._next_index = idx + 1
            if pts < target_pts:
                # decoding forward from a keyframe: cache intermediate frames for cheap back-stepping
                self._cache.put(idx, frame.to_ndarray(format="bgr24"))
                continue
            if pts > target_pts:
                # overshoot: the keyframe seek landed after the target; seek further back and retry
                if need_seek and self.index.keyframe_at_or_before(frame_index) > 0:
                    kf = self.index.keyframe_at_or_before(frame_index)
                    prev = self.index.keyframe_at_or_before(kf - 1)
                    self._container.seek(int(self.index.pts[prev]), stream=self._stream, backward=True, any_frame=False)
                    self._decoder = self._container.decode(self._stream)
                    need_seek = False
                    continue
                raise RuntimeError(f"decoder skipped frame {frame_index} (pts {target_pts}); got pts {pts}")
            arr = frame.to_ndarray(format="bgr24")
            self._cache.put(idx, arr)
            return arr
