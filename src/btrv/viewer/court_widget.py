"""Top-down court view synchronised by frame index with the video.

Renders the 94x50 ft court in the reference viewer's orientation (origin
top-left, y down) and positions from one or more sources for the frame
currently *displayed* in the video widget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from btrv.court.model import NBA_COURT

from .qt_utils import BALL_COLOR, EXTERNAL_COLOR, team_color

SPORTVU_PALETTE = [QColor(235, 64, 52), QColor(52, 120, 235), QColor(60, 200, 90), QColor(200, 60, 200)]


@dataclass
class CourtEntity:
    x: float
    y: float
    label: str
    color: QColor
    kind: str = "player"  # player | ball | external
    low_confidence: bool = False


@dataclass
class CourtFrameData:
    frame_index: int = -1
    timestamp_s: float = 0.0
    entities: list[CourtEntity] = field(default_factory=list)
    calibration_valid: Optional[bool] = None
    calibration_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


class CourtView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(376, 220)
        self.data = CourtFrameData()
        self.show_app = True
        self.show_sportvu = True
        self.show_external = True
        self.show_ball = True
        self.show_labels = True

    def set_data(self, data: CourtFrameData) -> None:
        self.data = data
        self.update()

    def _transform(self):
        margin = 18
        L, W = NBA_COURT.length, NBA_COURT.width
        s = min((self.width() - 2 * margin) / L, (self.height() - 2 * margin - 18) / W)
        ox = (self.width() - L * s) / 2
        oy = (self.height() - 18 - W * s) / 2 + 18
        return s, ox, oy

    def to_widget(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._transform()
        return QPointF(ox + x * s, oy + y * s)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(28, 28, 28))
        s, ox, oy = self._transform()
        L, W = NBA_COURT.length, NBA_COURT.width
        p.setBrush(QBrush(QColor(196, 150, 96)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(ox, oy, L * s, W * s))
        pen = QPen(QColor(250, 250, 250), 1.2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for pl in NBA_COURT.polylines():
            path = QPainterPath()
            pts = [self.to_widget(x, y) for x, y in pl]
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.drawPath(path)
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        d = self.data
        header = f"frame {d.frame_index}  t={d.timestamp_s:.3f}s" if d.frame_index >= 0 else "no frame"
        if d.calibration_valid is not None:
            header += "   calibration: %s (%.2f)" % ("valid" if d.calibration_valid else "INVALID", d.calibration_confidence or 0.0)
        p.setPen(QColor(230, 230, 230))
        p.drawText(QPointF(6, 13), header)
        for i, n in enumerate(d.notes[:3]):
            p.setPen(QColor(255, 160, 80))
            p.drawText(QPointF(6, self.height() - 6 - 12 * i), n)
        r = max(3.0, 1.1 * s)
        for e in d.entities:
            if e.kind == "player" and not self.show_app:
                continue
            if e.kind == "sportvu" and not self.show_sportvu:
                continue
            if e.kind == "external" and not self.show_external:
                continue
            if e.kind == "ball" and not self.show_ball:
                continue
            c = self.to_widget(e.x, e.y)
            if e.kind == "ball":
                p.setBrush(QBrush(BALL_COLOR))
                p.setPen(QPen(QColor(0, 0, 0), 1))
                p.drawEllipse(c, r * 0.6, r * 0.6)
                continue
            p.setBrush(QBrush(e.color if not e.low_confidence else QColor(e.color.red(), e.color.green(), e.color.blue(), 110)))
            p.setPen(QPen(QColor(0, 0, 0) if e.kind == "player" else QColor(255, 255, 255), 1.5, Qt.PenStyle.SolidLine if e.kind != "sportvu" else Qt.PenStyle.DashLine))
            if e.kind == "external":
                p.drawRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
            else:
                p.drawEllipse(c, r, r)
            if self.show_labels and e.label:
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(c.x() + r + 2, c.y() + 4), e.label)
        # axis hint
        p.setPen(QColor(200, 200, 200))
        p.drawText(QPointF(ox, oy + W * s + 13), "x: 0 -> 94 ft (left basket at x=5.25), y: 0 (top) -> 50 ft (bottom)")
