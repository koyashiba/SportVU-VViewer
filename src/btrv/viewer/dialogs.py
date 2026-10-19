"""Processing, calibration and export UI."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from btrv.core.compute import resolve_backend
from btrv.court.homography import Calibration, apply_symmetry, fit_homography
from btrv.court.model import NBA_COURT
from btrv.detect.base import DetectorConfig, PINNED_MODEL_ID
from btrv.pipeline import ProcessingConfig, process_video
from btrv.pipeline.calibration_file import calibration_document, save_calibration


class ProcessingThread(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: ProcessingConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            res = process_video(self.cfg, progress=lambda i, n, m: self.progress.emit(i, n, m), cancel=self.cancel)
            self.finished_ok.emit(res)
        except Exception as e:  # pragma: no cover - UI path
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class ProcessingDialog(QDialog):
    """Configure and run the precompute worker; emits run_ready(run_dir) when done."""

    run_ready = Signal(str)

    def __init__(self, video_path: str, source_id: str, frame_count: int, roi: Optional[tuple[int, int, int, int]], calibration_path: Optional[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Process video (precompute tracking)")
        self.video_path = video_path
        self.source_id = source_id
        self._thread: Optional[ProcessingThread] = None
        form = QFormLayout()
        self.run_dir = QLineEdit(str(Path(video_path).with_suffix("")) + "_run")
        b = QPushButton("...")
        b.clicked.connect(self._pick_run_dir)
        row = QHBoxLayout()
        row.addWidget(self.run_dir)
        row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        form.addRow("Run directory", w)
        self.camera_mode = QComboBox()
        self.camera_mode.addItems(["fixed", "moving"])
        form.addRow("Camera mode", self.camera_mode)
        self.roi_label = QLabel("none (full frame)" if roi is None else f"{roi} (set in the ROI panel)")
        self._roi = roi
        form.addRow("ROI (fixed mode)", self.roi_label)
        self.detector = QComboBox()
        self.detector.addItems(["roboflow_local", "roboflow_hosted", "fake"])
        form.addRow("Detector backend", self.detector)
        self.model_id = QLineEdit(PINNED_MODEL_ID)
        form.addRow("Model id (pinned)", self.model_id)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 1.0)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.3)
        form.addRow("Confidence", self.confidence)
        self.compute = QComboBox()
        self.compute.addItems(["auto", "cuda", "mps", "cpu"])
        be = resolve_backend("auto")
        form.addRow(f"Compute backend (auto -> {be.resolved})", self.compute)
        self.max_side = QSpinBox()
        self.max_side.setRange(0, 16000)
        self.max_side.setValue(0)
        self.max_side.setSpecialValueText("native")
        form.addRow("Max inference side (px)", self.max_side)
        self.start = QSpinBox()
        self.start.setRange(0, max(0, frame_count - 1))
        self.stop = QSpinBox()
        self.stop.setRange(1, max(1, frame_count))
        self.stop.setValue(frame_count)
        self.step = QSpinBox()
        self.step.setRange(1, 1000)
        rng = QHBoxLayout()
        rng.addWidget(QLabel("start"))
        rng.addWidget(self.start)
        rng.addWidget(QLabel("stop"))
        rng.addWidget(self.stop)
        rng.addWidget(QLabel("step"))
        rng.addWidget(self.step)
        w2 = QWidget()
        w2.setLayout(rng)
        form.addRow("Frame range", w2)
        self.calibration = QLineEdit(calibration_path or "")
        form.addRow("Calibration file", self.calibration)
        self.auto_cal = QCheckBox("Auto-calibrate on first frame if no calibration file")
        form.addRow("", self.auto_cal)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)
        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        btns.addWidget(close)
        layout.addLayout(btns)

    def _pick_run_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Run directory", self.run_dir.text())
        if d:
            self.run_dir.setText(d)

    def config(self) -> ProcessingConfig:
        return ProcessingConfig(
            video_path=self.video_path,
            run_dir=self.run_dir.text(),
            source_id=self.source_id,
            camera_mode=self.camera_mode.currentText(),
            roi=self._roi if self.camera_mode.currentText() == "fixed" else None,
            frame_start=self.start.value(),
            frame_stop=self.stop.value(),
            frame_step=self.step.value(),
            detector=DetectorConfig(backend=self.detector.currentText(), model_id=self.model_id.text().strip(), confidence=self.confidence.value(), compute=self.compute.currentText(), max_inference_side=self.max_side.value() or None),
            calibration_path=self.calibration.text().strip() or None,
            auto_calibrate=self.auto_cal.isChecked(),
        )

    def _run(self):
        cfg = self.config()
        self.log.append(f"starting: {cfg.detector.backend} {cfg.detector.model_id} frames [{cfg.frame_start}, {cfg.frame_stop}) step {cfg.frame_step}")
        self._thread = ProcessingThread(cfg, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._thread.start()

    def _cancel(self):
        if self._thread is not None:
            self._thread.cancel.set()

    def _on_progress(self, i, n, msg):
        self.progress.setMaximum(n)
        self.progress.setValue(i)

    def _on_done(self, res):
        s = res.summary
        self.log.append(f"done: {s['frames_processed']} frames, {s['detections_total']} detections, tracks {s['tracking']['tracks_total']}, status {s['status']}")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.run_ready.emit(str(res.run_dir))

    def _on_failed(self, msg):
        self.log.append("FAILED: " + msg)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)


class CalibrationPanel(QWidget):
    """Manual landmark calibration, automatic calibration, ROI selection."""

    pick_requested = Signal(str)  # landmark name -> main window switches the video view to pick mode
    roi_mode_requested = Signal(bool)
    calibration_changed = Signal(object)  # Calibration or None
    landmarks_changed = Signal(list)  # [(name, x, y)]
    roi_changed = Signal(object)  # tuple or None
    auto_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: dict[str, tuple[float, float]] = {}
        self.calibration: Optional[Calibration] = None
        self.image_size = (0, 0)
        self.frame_index = 0
        self.source_id = ""
        self.roi: Optional[tuple[int, int, int, int]] = None
        layout = QVBoxLayout(self)
        g = QGroupBox("ROI (fixed camera)")
        gl = QHBoxLayout(g)
        self.roi_btn = QPushButton("Select ROI by dragging")
        self.roi_btn.setCheckable(True)
        self.roi_btn.toggled.connect(self.roi_mode_requested.emit)
        self.roi_lbl = QLabel("none")
        clr = QPushButton("Clear")
        clr.clicked.connect(self.clear_roi)
        gl.addWidget(self.roi_btn)
        gl.addWidget(self.roi_lbl, 1)
        gl.addWidget(clr)
        layout.addWidget(g)
        g2 = QGroupBox("Manual landmarks (select a landmark, then click it in the video)")
        g2l = QVBoxLayout(g2)
        self.list = QListWidget()
        for lm in NBA_COURT.landmarks:
            it = QListWidgetItem(f"{lm.name}  ({lm.x:g}, {lm.y:g} ft)  - {lm.description}")
            it.setData(Qt.ItemDataRole.UserRole, lm.name)
            self.list.addItem(it)
        self.list.itemClicked.connect(lambda it: self.pick_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        g2l.addWidget(self.list)
        row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit homography")
        self.fit_btn.clicked.connect(self.fit)
        rm = QPushButton("Remove selected point")
        rm.clicked.connect(self.remove_selected)
        clear = QPushButton("Clear points")
        clear.clicked.connect(self.clear_points)
        row.addWidget(self.fit_btn)
        row.addWidget(rm)
        row.addWidget(clear)
        g2l.addLayout(row)
        layout.addWidget(g2)
        g3 = QGroupBox("Automatic / orientation")
        g3l = QHBoxLayout(g3)
        auto = QPushButton("Auto-calibrate this frame")
        auto.clicked.connect(self.auto_requested.emit)
        fe = QPushButton("Flip ends (x)")
        fe.clicked.connect(lambda: self.flip("flip_ends"))
        fs = QPushButton("Flip sides (y)")
        fs.clicked.connect(lambda: self.flip("flip_sides"))
        g3l.addWidget(auto)
        g3l.addWidget(fe)
        g3l.addWidget(fs)
        layout.addWidget(g3)
        row2 = QHBoxLayout()
        save = QPushButton("Save calibration...")
        save.clicked.connect(self.save)
        clearcal = QPushButton("Discard calibration")
        clearcal.clicked.connect(self.discard)
        row2.addWidget(save)
        row2.addWidget(clearcal)
        layout.addLayout(row2)
        self.status = QLabel("no calibration")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    # -- state ----------------------------------------------------------------
    def set_context(self, source_id: str, image_size: tuple[int, int], frame_index: int) -> None:
        self.source_id, self.image_size, self.frame_index = source_id, image_size, frame_index

    def add_point(self, name: str, x: float, y: float) -> None:
        self.points[name] = (x, y)
        self.landmarks_changed.emit([(n, px, py) for n, (px, py) in self.points.items()])
        self.status.setText(f"{len(self.points)} landmark(s); need >= 4 to fit")

    def remove_selected(self) -> None:
        it = self.list.currentItem()
        if it is not None:
            self.points.pop(it.data(Qt.ItemDataRole.UserRole), None)
            self.landmarks_changed.emit([(n, px, py) for n, (px, py) in self.points.items()])

    def clear_points(self) -> None:
        self.points.clear()
        self.landmarks_changed.emit([])

    def set_roi(self, roi: Optional[tuple[int, int, int, int]]) -> None:
        self.roi = roi
        self.roi_lbl.setText("none" if roi is None else str(roi))
        self.roi_btn.setChecked(False)
        self.roi_changed.emit(roi)

    def clear_roi(self) -> None:
        self.set_roi(None)

    def set_calibration(self, cal: Optional[Calibration], note: str = "") -> None:
        self.calibration = cal
        if cal is None:
            self.status.setText("no calibration")
        else:
            self.status.setText(f"{cal.method}: valid={cal.valid} confidence={cal.confidence:.2f} reproj_err={'' if cal.reprojection_error_px is None else '%.1f px' % cal.reprojection_error_px} {cal.notes} {note}")
        self.calibration_changed.emit(cal)

    def fit(self) -> None:
        if len(self.points) < 4:
            QMessageBox.warning(self, "Calibration", "Click at least 4 landmarks first (5 or more lets RANSAC reject a bad click).")
            return
        names = list(self.points)
        img = np.array([self.points[n] for n in names])
        court = np.array([[NBA_COURT.landmark(n).x, NBA_COURT.landmark(n).y] for n in names])
        try:
            cal = fit_homography(img, court, self.image_size, method="manual", frame_index=self.frame_index, landmark_names=names)
        except ValueError as e:
            QMessageBox.warning(self, "Calibration", str(e))
            return
        bad = [l["name"] for l in cal.landmarks if not l["inlier"]]
        self.set_calibration(cal, f"outliers: {bad}" if bad else "")

    def flip(self, name: str) -> None:
        if self.calibration is None:
            return
        c = self.calibration
        self.set_calibration(Calibration(apply_symmetry(c.H, name), c.method, c.confidence, c.valid, c.frame_index, c.reprojection_error_px, c.inlier_ratio, [], c.notes + f"; {name}"))

    def discard(self) -> None:
        self.set_calibration(None)

    def save(self) -> None:
        if self.calibration is None:
            QMessageBox.information(self, "Calibration", "Nothing to save yet.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save calibration", "calibration.json", "JSON (*.json)")
        if p:
            save_calibration(p, calibration_document(self.source_id, self.image_size, "fixed" if self.roi else "fixed", [self.calibration], self.roi))
            self.status.setText(self.status.text() + f"\nsaved to {p}")


class ExportDialog(QDialog):
    def __init__(self, has_ball: bool, frame_range: tuple[int, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export SportVU")
        form = QFormLayout(self)
        self.kind = QComboBox()
        self.kind.addItems(["SportVU conforming (ball first; needs external ball coordinates)", "Player-only debug export (NONSTANDARD, no ball)"])
        if not has_ball:
            self.kind.setCurrentIndex(1)
        form.addRow("Kind", self.kind)
        self.game_id = QLineEdit("btrv")
        form.addRow("gameid", self.game_id)
        self.epoch = QSpinBox()
        self.epoch.setRange(0, 2_000_000_000)
        form.addRow("epoch_ms (added to source time)", self.epoch)
        self.ball_z = QDoubleSpinBox()
        self.ball_z.setRange(0, 20)
        form.addRow("ball z default (ft) when unknown", self.ball_z)
        self.start = QSpinBox()
        self.start.setRange(frame_range[0], max(frame_range))
        self.start.setValue(frame_range[0])
        self.stop = QSpinBox()
        self.stop.setRange(frame_range[0] + 1, max(frame_range) + 1)
        self.stop.setValue(max(frame_range) + 1)
        form.addRow("frame start", self.start)
        form.addRow("frame stop (exclusive)", self.stop)
        info = QLabel("Ball coordinates: %s" % ("loaded" if has_ball else "NOT loaded - conforming export will fail with an actionable error"))
        info.setWordWrap(True)
        form.addRow(info)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    @property
    def player_only_debug(self) -> bool:
        return self.kind.currentIndex() == 1
