"""Background decoder: always decodes the most recently requested frame and
prefetches the next ones, so the GUI thread never blocks on video decoding."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QMutex, QThread, QWaitCondition, Signal

from btrv.source.file_source import FileSource


class DecodeWorker(QThread):
    decoded = Signal(int, object)  # frame_index, BGR ndarray
    failed = Signal(int, str)

    def __init__(self, source: FileSource, prefetch: int = 2, parent=None):
        super().__init__(parent)
        self.source = source
        self.prefetch = prefetch
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._wanted: Optional[int] = None
        self._stop = False

    def request(self, frame_index: int) -> None:
        self._mutex.lock()
        self._wanted = int(frame_index)
        self._cond.wakeAll()
        self._mutex.unlock()

    def stop(self) -> None:
        self._mutex.lock()
        self._stop = True
        self._cond.wakeAll()
        self._mutex.unlock()
        self.wait(3000)

    def run(self) -> None:
        last: Optional[int] = None
        while True:
            self._mutex.lock()
            while self._wanted is None and not self._stop:
                # idle: prefetch upcoming frames into the source cache
                if last is not None and self.prefetch > 0:
                    nxt = [last + k for k in range(1, self.prefetch + 1) if last + k < self.source.info.frame_count]
                    self._mutex.unlock()
                    for i in nxt:
                        try:
                            self.source.read(i)
                        except Exception:
                            break
                    self._mutex.lock()
                    last = None  # prefetch once per request
                    continue
                self._cond.wait(self._mutex)
            if self._stop:
                self._mutex.unlock()
                return
            wanted = self._wanted
            self._wanted = None
            self._mutex.unlock()
            try:
                info, frame = self.source.read(wanted)
                self.decoded.emit(info.frame_index, frame)
                last = wanted
            except Exception as e:  # pragma: no cover - corrupt files
                self.failed.emit(wanted, str(e))
