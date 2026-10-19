from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QImage

TEAM_COLORS = {0: QColor(235, 64, 52), 1: QColor(52, 120, 235), None: QColor(240, 200, 40)}
NO_TEAM_COLOR = QColor(60, 220, 90)
EXTERNAL_COLOR = QColor(40, 230, 230)
COURT_LINE_COLOR = QColor(255, 255, 0)
BALL_COLOR = QColor(255, 140, 0)


def ndarray_to_qimage(bgr: np.ndarray) -> QImage:
    """Zero-copy BGR888 QImage; the caller must keep ``bgr`` alive while the image is used."""
    if not bgr.flags["C_CONTIGUOUS"]:
        bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        img = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_ARGB32)
    else:
        img = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)
    img._keep = bgr  # type: ignore[attr-defined]
    return img


def team_color(team_id, has_team_info: bool = True) -> QColor:
    if not has_team_info:
        return NO_TEAM_COLOR
    return TEAM_COLORS.get(team_id, TEAM_COLORS[None])
