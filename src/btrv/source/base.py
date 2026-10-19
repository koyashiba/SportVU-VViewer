"""Frame source interface.

v1 ships a file-based source.  A live source (camera / network stream) can be
added later by implementing :class:`FrameSource`; consumers only rely on
``read``/``next`` and on the frame identity (index + presentation timestamp).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Iterator, Optional

import numpy as np

from btrv.core.types import FrameInfo


@dataclass
class SourceInfo:
    source_id: str
    uri: str
    width: int
    height: int
    frame_count: Optional[int]  # None for live sources
    time_base: Optional[float]  # seconds per pts tick, None if unknown
    average_rate: Optional[float]  # informational only, never used for sync
    codec: Optional[str] = None
    duration_s: Optional[float] = None
    seekable: bool = True
    extra: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class FrameSource(ABC):
    """Produces BGR uint8 frames identified by (frame_index, timestamp_s)."""

    @property
    @abstractmethod
    def info(self) -> SourceInfo: ...

    @abstractmethod
    def frame_info(self, frame_index: int) -> FrameInfo:
        """Identity of a frame without decoding it (seekable sources only)."""

    @abstractmethod
    def read(self, frame_index: int) -> tuple[FrameInfo, np.ndarray]:
        """Decode exactly the frame with the given presentation-order index."""

    @abstractmethod
    def next(self) -> Optional[tuple[FrameInfo, np.ndarray]]:
        """Decode the next frame in presentation order, or None at end of stream."""

    def iter_frames(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> Iterator[tuple[FrameInfo, np.ndarray]]:
        n = self.info.frame_count
        if stop is None:
            stop = n if n is not None else 2**62
        i = start
        while i < stop:
            yield self.read(i)
            i += step

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
