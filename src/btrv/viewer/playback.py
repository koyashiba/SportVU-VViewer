"""Presentation-timestamp-driven playback clock.

The displayed frame is the last frame whose decoded timestamp is <= the media
time; media time advances as wall time * speed.  Frame stepping while paused
moves exactly one source frame.  No constant FPS is assumed anywhere.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

SPEEDS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)


class PlaybackController(QObject):
    frame_changed = Signal(int)
    playing_changed = Signal(bool)
    speed_changed = Signal(float)

    def __init__(self, parent=None, clock: Callable[[], float] = time.perf_counter, use_timer: bool = True):
        super().__init__(parent)
        self._clock = clock
        self._timestamps: Optional[np.ndarray] = None
        self._current = 0
        self._speed = 1.0
        self._playing = False
        self._wall0 = 0.0
        self._media0 = 0.0
        self._timer = QTimer(self) if use_timer else None
        if self._timer is not None:
            self._timer.setTimerType(self._timer.timerType().PreciseTimer)
            self._timer.setInterval(4)
            self._timer.timeout.connect(self.tick)

    # -- setup ----------------------------------------------------------------
    def set_timestamps(self, timestamps: Optional[np.ndarray]) -> None:
        self.pause()
        self._timestamps = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        self._current = 0
        self.frame_changed.emit(0)

    @property
    def frame_count(self) -> int:
        return 0 if self._timestamps is None else int(self._timestamps.shape[0])

    @property
    def current(self) -> int:
        return self._current

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def playing(self) -> bool:
        return self._playing

    def media_time(self) -> float:
        if self._timestamps is None:
            return 0.0
        if not self._playing:
            return float(self._timestamps[self._current])
        return self._media0 + (self._clock() - self._wall0) * self._speed

    # -- control ------------------------------------------------------------------
    def _anchor(self) -> None:
        self._wall0 = self._clock()
        self._media0 = float(self._timestamps[self._current]) if self._timestamps is not None else 0.0

    def play(self) -> None:
        if self._playing or self.frame_count == 0:
            return
        if self._current >= self.frame_count - 1:
            self.seek(0)
        self._playing = True
        self._anchor()
        if self._timer is not None:
            self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        if self._timer is not None:
            self._timer.stop()
        self.playing_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def set_speed(self, speed: float) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        if self._playing:
            self._media0 = self.media_time()
            self._wall0 = self._clock()
        self._speed = speed
        self.speed_changed.emit(speed)

    def seek(self, frame_index: int) -> None:
        if self.frame_count == 0:
            return
        i = max(0, min(int(frame_index), self.frame_count - 1))
        changed = i != self._current
        self._current = i
        if self._playing:
            self._anchor()
        if changed:
            self.frame_changed.emit(i)

    def step(self, delta: int) -> None:
        self.pause()
        self.seek(self._current + delta)

    def tick(self) -> None:
        if not self._playing or self._timestamps is None:
            return
        t = self.media_time()
        ts = self._timestamps
        if t >= ts[-1]:
            self._current = self.frame_count - 1
            self.pause()
            self.frame_changed.emit(self._current)
            return
        i = int(np.searchsorted(ts, t, side="right") - 1)
        i = max(0, i)
        if i != self._current:
            self._current = i
            self.frame_changed.emit(i)
