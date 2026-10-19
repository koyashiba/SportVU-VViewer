"""Video display with overlay layers and interaction modes (pan/zoom, ROI, landmark picking).

All overlay geometry is kept in *source pixels* and mapped to the widget with a
single scale+offset transform, so overlays stay aligned at any zoom and for any
video resolution (HD/4K/8K).  The frame itself is downscaled once per
(frame, zoom) with OpenCV, which is far cheaper than letting Qt rescale a 33 MP
image on every repaint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from btrv.core.types import FrameBoxes
from btrv.court.homography import Calibration, project_court_lines, project_to_image
from btrv.court.model import NBA_COURT
from btrv.io.heatmap import HeatmapMeta, render_heatmap_overlay

from .qt_utils import COURT_LINE_COLOR, EXTERNAL_COLOR, ndarray_to_qimage, team_color


@dataclass
class LayerState:
    local_boxes: bool = True
    raw_boxes: bool = False
    external_boxes: bool = True
    heatmap: bool = True
    heatmap_opacity: float = 0.6
    court_lines: bool = True
    roi: bool = True
    foot_points: bool = True
    labels: bool = True
    landmarks: bool = True


@dataclass
class FrameOverlays:
    """Everything the view needs to draw for one displayed frame (built by the main window)."""

    frame_index: int = -1
    local: Optional[FrameBoxes] = None
    raw: Optional[FrameBoxes] = None
    external: Optional[FrameBoxes] = None
    heatmap_grid: Optional[np.ndarray] = None
    heatmap_meta: Optional[HeatmapMeta] = None
    calibration: Optional[Calibration] = None
    roi: Optional[tuple[int, int, int, int]] = None
    landmarks: list[tuple[str, float, float]] = field(default_factory=list)  # (name, x, y) in source px


class VideoView(QWidget):
    point_picked = Signal(float, float)  # source pixels
    roi_selected = Signal(int, int, int, int)
    mouse_moved = Signal(float, float)

    MODE_PAN, MODE_ROI, MODE_PICK = "pan", "roi", "pick"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.layers = LayerState()
        self.overlays = FrameOverlays()
        self._frame: Optional[np.ndarray] = None
        self._frame_index = -1
        self._zoom = 1.0  # multiplier on fit scale
        self._pan = QPointF(0, 0)
        self._scaled: Optional[QImage] = None
        self._scaled_key = None
        self._heat_cache = None
        self._heat_key = None
        self.mode = self.MODE_PAN
        self._drag_start: Optional[QPointF] = None
        self._drag_cur: Optional[QPointF] = None
        self._pan_start: Optional[QPointF] = None
        self._pan_origin = QPointF(0, 0)
        self.setStyleSheet("background: #202020;")

    # -- data ---------------------------------------------------------------
    def set_frame(self, frame_index: int, frame: np.ndarray, overlays: FrameOverlays) -> None:
        self._frame = frame
        self._frame_index = frame_index
        self.overlays = overlays
        self.update()

    def set_overlays(self, overlays: FrameOverlays) -> None:
        self.overlays = overlays
        self.update()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self._frame

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode != self.MODE_PAN else Qt.CursorShape.OpenHandCursor)

    # -- transform ----------------------------------------------------------
    def _fit_scale(self) -> float:
        if self._frame is None:
            return 1.0
        h, w = self._frame.shape[:2]
        return min(self.width() / w, self.height() / h)

    def scale(self) -> float:
        return self._fit_scale() * self._zoom

    def offset(self) -> QPointF:
        if self._frame is None:
            return QPointF(0, 0)
        h, w = self._frame.shape[:2]
        s = self.scale()
        return QPointF((self.width() - w * s) / 2, (self.height() - h * s) / 2) + self._pan

    def to_widget(self, x: float, y: float) -> QPointF:
        s, o = self.scale(), self.offset()
        return QPointF(o.x() + x * s, o.y() + y * s)

    def to_source(self, p: QPointF) -> tuple[float, float]:
        s, o = self.scale(), self.offset()
        return ((p.x() - o.x()) / s, (p.y() - o.y()) / s)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    # -- painting ---------------------------------------------------------------
    def _scaled_image(self) -> Optional[QImage]:
        if self._frame is None:
            return None
        s = self.scale()
        h, w = self._frame.shape[:2]
        key = (id(self._frame), round(s, 5))
        if self._scaled_key == key and self._scaled is not None:
            return self._scaled
        tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
        if tw * th > 16_000_000:  # zoomed in on a huge frame: let Qt draw the sub-region instead
            self._scaled, self._scaled_key = None, None
            return None
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        small = cv2.resize(self._frame, (tw, th), interpolation=interp) if (tw, th) != (w, h) else self._frame
        self._scaled = ndarray_to_qimage(np.ascontiguousarray(small))
        self._scaled_key = key
        return self._scaled

    def _heat_image(self) -> Optional[QImage]:
        ov = self.overlays
        if ov.heatmap_grid is None or ov.heatmap_meta is None or self._frame is None:
            return None
        s = self.scale()
        h, w = self._frame.shape[:2]
        key = (ov.frame_index, id(ov.heatmap_grid), round(s, 5), id(ov.calibration))
        if self._heat_key == key:
            return self._heat_cache
        tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
        if tw * th > 16_000_000:
            return None
        meta = ov.heatmap_meta
        try:
            if meta.space == "image_pixels":
                m2 = HeatmapMeta(**{**meta.__dict__, "extent": [v * s for v in meta.extent]})
                bgra = render_heatmap_overlay(ov.heatmap_grid, m2, (tw, th))
            else:
                if ov.calibration is None or not ov.calibration.valid:
                    self._heat_cache, self._heat_key = None, key
                    return None
                S = np.diag([s, s, 1.0])
                bgra = render_heatmap_overlay(ov.heatmap_grid, meta, (tw, th), S @ np.linalg.inv(ov.calibration.H))
        except Exception:
            bgra = None
        self._heat_cache = None if bgra is None else ndarray_to_qimage(np.ascontiguousarray(bgra))
        self._heat_key = key
        return self._heat_cache

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 32))
        if self._frame is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video (File > Open video...)")
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        s, o = self.scale(), self.offset()
        h, w = self._frame.shape[:2]
        img = self._scaled_image()
        target = QRectF(o.x(), o.y(), w * s, h * s)
        if img is not None:
            p.drawImage(target.topLeft(), img)
        else:
            p.drawImage(target, ndarray_to_qimage(self._frame))
        L = self.layers
        ov = self.overlays
        if L.heatmap:
            hi = self._heat_image()
            if hi is not None:
                p.setOpacity(L.heatmap_opacity)
                p.drawImage(target.topLeft(), hi)
                p.setOpacity(1.0)
        if L.court_lines and ov.calibration is not None and ov.calibration.valid:
            self._draw_court_lines(p, ov.calibration)
        if L.roi and ov.roi is not None:
            x0, y0, x1, y1 = ov.roi
            p.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self.to_widget(x0, y0), self.to_widget(x1, y1)))
        if L.raw_boxes and ov.raw is not None:
            self._draw_boxes(p, ov.raw, kind="raw")
        if L.external_boxes and ov.external is not None:
            self._draw_boxes(p, ov.external, kind="external")
        if L.local_boxes and ov.local is not None:
            self._draw_boxes(p, ov.local, kind="local")
        if L.landmarks and ov.landmarks:
            for name, x, y in ov.landmarks:
                c = self.to_widget(x, y)
                p.setPen(QPen(QColor(255, 0, 255), 2))
                p.drawLine(QPointF(c.x() - 8, c.y()), QPointF(c.x() + 8, c.y()))
                p.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() + 8))
                p.setPen(QColor(255, 0, 255))
                p.drawText(QPointF(c.x() + 10, c.y() - 4), name)
        if self.mode == self.MODE_ROI and self._drag_start is not None and self._drag_cur is not None:
            p.setPen(QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(255, 255, 0, 40)))
            p.drawRect(QRectF(self._drag_start, self._drag_cur))
        if ov.calibration is not None and not ov.calibration.valid and L.court_lines:
            p.setPen(QColor(255, 80, 80))
            p.drawText(QPointF(10, 20), "calibration invalid for this frame (confidence %.2f)" % ov.calibration.confidence)

    def _draw_court_lines(self, p: QPainter, cal: Calibration) -> None:
        h, w = self._frame.shape[:2]
        pen = QPen(COURT_LINE_COLOR, 1.5)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for pl in project_court_lines(cal.H, NBA_COURT, (w, h)):
            path = QPainterPath()
            pts = [self.to_widget(x, y) for x, y in pl]
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.drawPath(path)

    def _draw_boxes(self, p: QPainter, fb: FrameBoxes, kind: str) -> None:
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)
        for b in fb.boxes:
            x1, y1, x2, y2 = b.xyxy
            tl, br = self.to_widget(x1, y1), self.to_widget(x2, y2)
            if kind == "external":
                color = EXTERNAL_COLOR
                style = Qt.PenStyle.DashLine
            elif kind == "raw":
                color = QColor(180, 180, 180)
                style = Qt.PenStyle.DotLine
            else:
                color = team_color(None if b.team is None else b.team.team_id, b.team is not None)
                style = Qt.PenStyle.SolidLine if b.track_id is not None else Qt.PenStyle.DashLine
            pen = QPen(color, 2 if kind == "local" else 1.5, style)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(tl, br))
            if self.layers.foot_points and kind != "raw":
                fx, fy = b.foot_point
                c = self.to_widget(fx, fy)
                p.setBrush(QBrush(color))
                p.drawEllipse(c, 3, 3)
                p.setBrush(Qt.BrushStyle.NoBrush)
            if self.layers.labels:
                parts = []
                if b.track_id is not None:
                    parts.append(f"#{b.track_id}")
                parts.append(b.class_name)
                parts.append(f"{b.confidence:.2f}")
                if b.team is not None:
                    parts.append("T%s%s" % ("?" if b.team.team_id is None else b.team.team_id, "" if b.team.confidence >= 0.5 else "(low)"))
                if kind == "external":
                    parts.insert(0, "ext")
                if b.court is not None and not b.court.valid and kind == "local":
                    parts.append("off-court/unmapped")
                text = " ".join(parts)
                p.setPen(QColor(0, 0, 0, 160))
                p.setBrush(QBrush(QColor(0, 0, 0, 160)))
                tw = p.fontMetrics().horizontalAdvance(text) + 6
                p.drawRect(QRectF(tl.x(), tl.y() - 16, tw, 16))
                p.setPen(color)
                p.drawText(QPointF(tl.x() + 3, tl.y() - 4), text)

    # -- interaction ------------------------------------------------------------
    def wheelEvent(self, event):  # noqa: N802
        if self._frame is None:
            return
        before = self.to_source(QPointF(event.position()))
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom = float(np.clip(self._zoom * factor, 0.25, 40.0))
        after_w = self.to_widget(*before)
        self._pan += QPointF(event.position()) - after_w
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if self._frame is None:
            return
        pos = QPointF(event.position())
        if self.mode == self.MODE_PICK and event.button() == Qt.MouseButton.LeftButton:
            x, y = self.to_source(pos)
            self.point_picked.emit(x, y)
        elif self.mode == self.MODE_ROI and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = pos
            self._drag_cur = pos
        else:
            self._pan_start = pos
            self._pan_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):  # noqa: N802
        pos = QPointF(event.position())
        if self._frame is not None:
            x, y = self.to_source(pos)
            self.mouse_moved.emit(x, y)
        if self.mode == self.MODE_ROI and self._drag_start is not None:
            self._drag_cur = pos
            self.update()
        elif self._pan_start is not None:
            self._pan = self._pan_origin + (pos - self._pan_start)
            self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self.mode == self.MODE_ROI and self._drag_start is not None and self._frame is not None:
            pos = QPointF(event.position())
            x0, y0 = self.to_source(self._drag_start)
            x1, y1 = self.to_source(pos)
            h, w = self._frame.shape[:2]
            xa, xb = sorted((int(round(x0)), int(round(x1))))
            ya, yb = sorted((int(round(y0)), int(round(y1))))
            xa, ya = max(0, xa), max(0, ya)
            xb, yb = min(w, xb), min(h, yb)
            self._drag_start = self._drag_cur = None
            if xb - xa > 4 and yb - ya > 4:
                self.roi_selected.emit(xa, ya, xb, yb)
            self.update()
        self._pan_start = None
        if self.mode == self.MODE_PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.reset_view()
