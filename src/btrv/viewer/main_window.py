"""Main window: video + overlays, playback controls, synchronised court view,
calibration/ROI panel, imports/exports and the localhost adapter."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from btrv.adapter.server import AdapterServer
from btrv.court.auto_calibrate import auto_calibrate
from btrv.court.homography import Calibration
from btrv.io.ball import BallRecord
from btrv.io.errors import ExportValidationError, ImportValidationError
from btrv.io.sportvu import SportVUExportOptions, export_sportvu, write_sportvu
from btrv.io.boxes import write_boxes_file

from .court_widget import CourtEntity, CourtFrameData, CourtView, SPORTVU_PALETTE
from .decode_worker import DecodeWorker
from .dialogs import CalibrationPanel, ExportDialog, ProcessingDialog
from .playback import SPEEDS, PlaybackController
from .qt_utils import EXTERNAL_COLOR, team_color
from .session import ViewerSession
from .video_widget import FrameOverlays, VideoView


class MainWindow(QMainWindow):
    hub_changed = Signal(str, object)  # relayed from the adapter thread into the GUI thread

    def __init__(self, adapter_port: int = 8765, start_adapter: bool = True):
        super().__init__()
        self.setWindowTitle("Basketball Tracking & Research Viewer")
        self.resize(1400, 860)
        self.session = ViewerSession()
        self.playback = PlaybackController(self)
        self.worker: Optional[DecodeWorker] = None
        self.adapter: Optional[AdapterServer] = None
        self.adapter_port = adapter_port
        self._landmarks: list[tuple[str, float, float]] = []
        self._pick_name: Optional[str] = None
        self._build_ui()
        self.playback.frame_changed.connect(self._on_frame_changed)
        self.playback.playing_changed.connect(lambda on: self.play_btn.setText("Pause" if on else "Play"))
        self.session.hub.subscribe(lambda kind, info: self.hub_changed.emit(kind, info))
        self.hub_changed.connect(self._on_hub_changed)
        if start_adapter:
            self.start_adapter()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        self.video = VideoView()
        self.video.point_picked.connect(self._on_point_picked)
        self.video.roi_selected.connect(self._on_roi_selected)
        self.video.mouse_moved.connect(self._on_mouse_moved)
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.video, 1)
        # transport bar
        bar = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.playback.toggle)
        back = QPushButton("<| frame")
        back.clicked.connect(lambda: self.playback.step(-1))
        fwd = QPushButton("frame |>")
        fwd.clicked.connect(lambda: self.playback.step(+1))
        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem(f"{s:g}x", s)
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_box.currentIndexChanged.connect(lambda i: self.playback.set_speed(self.speed_box.itemData(i)))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.playback.seek)
        self.frame_lbl = QLabel("frame - / -   t = -")
        self.frame_lbl.setMinimumWidth(320)
        for w in (self.play_btn, back, fwd, QLabel("speed"), self.speed_box):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.frame_lbl)
        v.addLayout(bar)
        # layer bar
        lb = QHBoxLayout()
        self.chk = {}
        for key, label, default in (("local_boxes", "Local boxes", True), ("raw_boxes", "Raw detections", False), ("external_boxes", "External boxes", True), ("heatmap", "Ball heat map", True), ("court_lines", "Court lines", True), ("foot_points", "Foot points", True), ("labels", "Labels", True)):
            c = QCheckBox(label)
            c.setChecked(default)
            c.toggled.connect(lambda on, k=key: self._set_layer(k, on))
            self.chk[key] = c
            lb.addWidget(c)
        lb.addWidget(QLabel("heat-map opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.setMaximumWidth(140)
        self.opacity.valueChanged.connect(lambda val: self._set_layer("heatmap_opacity", val / 100.0))
        lb.addWidget(self.opacity)
        lb.addStretch(1)
        v.addLayout(lb)
        self.setCentralWidget(central)
        # court dock (floatable -> separate window)
        self.court = CourtView()
        self.court_dock = QDockWidget("Top-down court view (synchronised)", self)
        self.court_dock.setWidget(self._court_panel())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.court_dock)
        # calibration dock
        self.calib = CalibrationPanel()
        self.calib.pick_requested.connect(self._start_pick)
        self.calib.roi_mode_requested.connect(lambda on: self.video.set_mode(VideoView.MODE_ROI if on else VideoView.MODE_PAN))
        self.calib.calibration_changed.connect(self._on_manual_calibration)
        self.calib.landmarks_changed.connect(self._on_landmarks)
        self.calib.roi_changed.connect(self._on_roi_changed)
        self.calib.auto_requested.connect(self._auto_calibrate)
        self.calib_dock = QDockWidget("Calibration & ROI", self)
        self.calib_dock.setWidget(self.calib)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.calib_dock)
        self.tabifyDockWidget(self.court_dock, self.calib_dock)
        self.court_dock.raise_()
        self.setStatusBar(QStatusBar())
        self.status_data = QLabel("")
        self.statusBar().addPermanentWidget(self.status_data)
        self._build_menus()

    def _court_panel(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        for attr, label in (("show_app", "App tracking"), ("show_sportvu", "Imported SportVU"), ("show_external", "External boxes"), ("show_ball", "Ball"), ("show_labels", "IDs")):
            c = QCheckBox(label)
            c.setChecked(True)
            c.toggled.connect(lambda on, a=attr: (setattr(self.court, a, on), self.court.update()))
            row.addWidget(c)
        row.addStretch(1)
        l.addLayout(row)
        l.addWidget(self.court, 1)
        return w

    def _build_menus(self):
        m = self.menuBar()
        f = m.addMenu("&File")
        self._act(f, "Open video...", self.open_video, "Ctrl+O")
        self._act(f, "Load run directory...", self.load_run, "Ctrl+R")
        f.addSeparator()
        self._act(f, "Import external boxes (btrv.boxes)...", self.import_boxes)
        self._act(f, "Import ball heat map (btrv.heatmap .h5/.npz)...", self.import_heatmap)
        self._act(f, "Import ball coordinates (btrv.ball)...", self.import_ball)
        self._act(f, "Import SportVU JSON for court view...", self.import_sportvu)
        self._act(f, "Load calibration file...", self.import_calibration)
        f.addSeparator()
        self._act(f, "Export tracked boxes (btrv.boxes)...", lambda: self.export_boxes(False))
        self._act(f, "Export raw detections (btrv.boxes)...", lambda: self.export_boxes(True))
        self._act(f, "Export SportVU...", self.export_sportvu)
        f.addSeparator()
        self._act(f, "Quit", self.close, "Ctrl+Q")
        p = m.addMenu("&Process")
        self._act(p, "Run processing on this video...", self.open_processing, "Ctrl+P")
        vw = m.addMenu("&View")
        self._act(vw, "Reset zoom", self.video.reset_view, "0")
        vw.addAction(self.court_dock.toggleViewAction())
        vw.addAction(self.calib_dock.toggleViewAction())
        self._act(vw, "Use live (adapter) heat map", lambda: (self.session.use_live_heatmap(), self._refresh_overlays()))
        self._act(vw, "Set SportVU time offset (epoch ms)...", self._set_sportvu_epoch)
        a = m.addMenu("&Adapter")
        self._act(a, "Start localhost adapter", self.start_adapter)
        self._act(a, "Stop adapter", self.stop_adapter)
        self._act(a, "Show adapter info", self.show_adapter_info)
        # keyboard
        self._act(None, "play/pause", self.playback.toggle, "Space")
        self._act(None, "step back", lambda: self.playback.step(-1), "Left")
        self._act(None, "step fwd", lambda: self.playback.step(+1), "Right")
        self._act(None, "first", lambda: self.playback.seek(0), "Home")
        self._act(None, "last", lambda: self.playback.seek(self.playback.frame_count - 1), "End")

    def _act(self, menu, text, fn, shortcut=None):
        act = QAction(text, self)
        act.triggered.connect(fn)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if menu is not None:
            menu.addAction(act)
        else:
            self.addAction(act)
        return act

    # ------------------------------------------------------------ helpers --
    def _error(self, title: str, e: Exception):
        QMessageBox.critical(self, title, f"{e}")
        self.statusBar().showMessage(f"{title}: {e}", 10000)

    def _set_layer(self, key: str, value):
        setattr(self.video.layers, key, value)
        self.video.update()

    # -------------------------------------------------------------- video --
    def open_video(self, path: Optional[str] = None, source_id: Optional[str] = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.ts *.webm);;All files (*)")
            if not path:
                return
        try:
            self._stop_worker()
            self.session.open_video(path, source_id)
        except Exception as e:
            self._error("Open video failed", e)
            return
        src = self.session.source
        self.playback.set_timestamps(self.session.timestamps)
        self.slider.setRange(0, max(0, src.info.frame_count - 1))
        self.worker = DecodeWorker(src, parent=self)
        self.worker.decoded.connect(self._on_decoded)
        self.worker.failed.connect(lambda i, m: self.statusBar().showMessage(f"decode error frame {i}: {m}", 8000))
        self.worker.start()
        self.calib.set_context(src.info.source_id, (src.info.width, src.info.height), 0)
        self.setWindowTitle(f"BTRV - {Path(path).name} [{src.info.source_id}] {src.info.width}x{src.info.height}, {src.info.frame_count} frames")
        self._on_frame_changed(0)
        self._update_status()

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def _on_frame_changed(self, i: int):
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        if self.worker is not None:
            self.worker.request(i)
        if self.adapter is not None:
            self.adapter.broadcast_state(i, self.session.timestamp(i))

    def _on_decoded(self, i: int, frame: np.ndarray):
        self.session.current_frame = i
        self.video.set_frame(i, frame, self._overlays_for(i))
        self.court.set_data(self._court_data_for(i))
        self.calib.frame_index = i
        pts = self.session.pts(i)
        self.frame_lbl.setText(f"frame {i} / {self.session.frame_count - 1}   t = {self.session.timestamp(i):.4f} s   pts = {pts}   {self.playback.speed:g}x")

    def _refresh_overlays(self):
        i = self.video.frame_index
        if i >= 0:
            self.video.set_overlays(self._overlays_for(i))
            self.court.set_data(self._court_data_for(i))
        self._update_status()

    def _overlays_for(self, i: int) -> FrameOverlays:
        s = self.session
        grid, meta = s.heatmap_grid(i)
        return FrameOverlays(
            frame_index=i,
            local=s.local_boxes(i),
            raw=s.local_boxes(i, raw=True) if self.video.layers.raw_boxes else None,
            external=s.external_boxes(i),
            heatmap_grid=grid,
            heatmap_meta=meta,
            calibration=s.calibration_for_frame(i),
            roi=s.roi,
            landmarks=list(self._landmarks),
        )

    def _court_data_for(self, i: int) -> CourtFrameData:
        s = self.session
        d = CourtFrameData(frame_index=i, timestamp_s=s.timestamp(i))
        cal = s.calibration_for_frame(i)
        if cal is not None:
            d.calibration_valid, d.calibration_confidence = cal.valid, cal.confidence
        fb = s.local_boxes(i)
        if fb is not None:
            for b in fb.boxes:
                if b.court is not None and b.court.valid:
                    tid = None if b.team is None else b.team.team_id
                    d.entities.append(CourtEntity(b.court.x, b.court.y, "" if b.track_id is None else str(b.track_id), team_color(tid, b.team is not None), "player", bool(b.team is not None and b.team.confidence < 0.5)))
            if fb.calibration is not None and not fb.calibration.get("valid", True):
                d.notes.append("no valid calibration for this frame: positions withheld")
        ext = s.external_boxes(i)
        if ext is not None:
            for b in ext.boxes:
                if b.court is not None and b.court.valid:
                    d.entities.append(CourtEntity(b.court.x, b.court.y, "" if b.track_id is None else f"e{b.track_id}", EXTERNAL_COLOR, "external"))
        ball = s.ball_for_frame(i)
        if ball is not None:
            d.entities.append(CourtEntity(ball.x, ball.y, "", None, "ball"))
        m = s.sportvu_moment_for_frame(i)
        if m is not None:
            teams = {}
            for team_id, pid, x, y, z in m.players:
                if team_id not in teams:
                    teams[team_id] = SPORTVU_PALETTE[len(teams) % len(SPORTVU_PALETTE)]
                d.entities.append(CourtEntity(x, y, str(pid), teams[team_id], "sportvu"))
            if m.ball is not None and ball is None:
                d.entities.append(CourtEntity(m.ball[0], m.ball[1], "", None, "ball"))
        elif s.sportvu is not None:
            d.notes.append("no SportVU moment within tolerance of this frame time (View > Set SportVU time offset)")
        return d

    def _update_status(self):
        s = self.session
        parts = []
        if s.hub.tracking is not None:
            parts.append(f"run: {s.hub.tracking.stats()['frames']} frames")
        if len(s.hub.external_boxes.frame_indices):
            parts.append(f"ext boxes: {len(s.hub.external_boxes.frame_indices)} frames")
        if s.heatmap_source != "none":
            parts.append(f"heat map: {s.heatmap_source}")
        if s.hub.ball:
            parts.append(f"ball: {len(s.hub.ball)} frames")
        if s.sportvu is not None:
            parts.append(f"sportvu: {len(s.sportvu)} moments")
        if self.adapter is not None:
            parts.append(f"adapter {self.adapter.url}")
        self.status_data.setText("  |  ".join(parts))

    def _on_mouse_moved(self, x: float, y: float):
        cal = self.session.calibration_for_frame(self.video.frame_index) if self.video.frame_index >= 0 else None
        txt = f"px ({x:.0f}, {y:.0f})"
        if cal is not None and cal.valid:
            from btrv.court.homography import project_to_court

            xy, ok = project_to_court(cal.H, np.array([[x, y]]))
            if ok.all():
                txt += f"   court ({xy[0, 0]:.1f}, {xy[0, 1]:.1f}) ft"
        self.statusBar().showMessage(txt)

    # ---------------------------------------------------------------- run --
    def load_run(self, run_dir: Optional[str] = None):
        if not run_dir:
            run_dir = QFileDialog.getExistingDirectory(self, "Run directory")
            if not run_dir:
                return
        try:
            had_video = self.session.source is not None
            rep = self.session.load_run(run_dir)
            if not had_video and self.session.source is not None:
                # load_run opened the video recorded in the manifest
                path = self.session.source.path
                sid = self.session.source.info.source_id
                self.session.close_video()
                self.open_video(str(path), sid)
        except Exception as e:
            self._error("Load run failed", e)
            return
        self.statusBar().showMessage(f"run loaded: {rep['tracked']}", 8000)
        self._refresh_overlays()

    def open_processing(self):
        if self.session.source is None:
            QMessageBox.information(self, "Process", "Open a video first.")
            return
        src = self.session.source
        cal_path = None
        if self.calib.calibration is not None:
            p, _ = QFileDialog.getSaveFileName(self, "Save current calibration for processing", str(Path(src.path).with_suffix(".calibration.json")), "JSON (*.json)")
            if p:
                from btrv.pipeline.calibration_file import calibration_document, save_calibration

                save_calibration(p, calibration_document(src.info.source_id, (src.info.width, src.info.height), "fixed", [self.calib.calibration], self.session.roi))
                cal_path = p
        dlg = ProcessingDialog(str(src.path), src.info.source_id, src.info.frame_count, self.session.roi, cal_path, self)
        dlg.run_ready.connect(lambda d: self.load_run(d))
        dlg.exec()

    # ------------------------------------------------------------ imports --
    def _import(self, title: str, filt: str, fn):
        p, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if not p:
            return
        try:
            rep = fn(p)
        except ImportValidationError as e:
            self._error(title + " rejected", e)
            return
        except Exception as e:
            self._error(title + " failed", e)
            return
        self.statusBar().showMessage(f"{title}: {rep}", 8000)
        self._refresh_overlays()

    def import_boxes(self):
        self._import("Import external boxes", "btrv.boxes (*.jsonl *.json)", self.session.import_external_boxes)

    def import_heatmap(self):
        self._import("Import heat map", "btrv.heatmap (*.h5 *.hdf5 *.npz)", self.session.import_heatmap)

    def import_ball(self):
        self._import("Import ball coordinates", "btrv.ball (*.jsonl *.json)", self.session.import_ball)

    def import_sportvu(self):
        self._import("Import SportVU", "SportVU JSON (*.json)", self.session.import_sportvu)

    def import_calibration(self):
        def fn(p):
            cal = self.session.import_calibration(p)
            self.calib.set_calibration(cal, "(loaded)")
            self.calib.set_roi(self.session.roi)
            return f"valid={cal.valid} confidence={cal.confidence:.2f}"

        self._import("Load calibration", "btrv.calibration (*.json)", fn)

    def _set_sportvu_epoch(self):
        v, ok = QInputDialog.getInt(self, "SportVU time offset", "moment unix_ms = epoch_ms + round(frame time * 1000)\nepoch_ms:", self.session.sportvu_epoch_ms, 0, 2_000_000_000)
        if ok:
            self.session.sportvu_epoch_ms = v
            self._refresh_overlays()

    # ------------------------------------------------------------ exports --
    def export_boxes(self, raw: bool):
        table = self.session.hub.raw if raw else self.session.hub.tracking
        if table is None:
            QMessageBox.information(self, "Export", "Load or compute a run first.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export boxes", "raw_boxes.jsonl" if raw else "tracked_boxes.jsonl", "JSON Lines (*.jsonl)")
        if p:
            n = write_boxes_file(p, table.iter_frames())
            self.statusBar().showMessage(f"wrote {n} frames to {p}", 8000)

    def export_sportvu(self):
        table = self.session.hub.tracking
        if table is None:
            QMessageBox.information(self, "Export", "Load or compute a run first.")
            return
        dlg = ExportDialog(bool(self.session.hub.ball), table.frame_range(), self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        opts = SportVUExportOptions(game_id=dlg.game_id.text(), epoch_ms=dlg.epoch.value(), ball_z_default_ft=dlg.ball_z.value(), frame_start=dlg.start.value(), frame_stop=dlg.stop.value())
        try:
            doc, meta = export_sportvu(table, list(self.session.hub.ball.values()), opts, player_only_debug=dlg.player_only_debug)
        except ExportValidationError as e:
            self._error("SportVU export not possible", e)
            return
        default = "sportvu_player_only_debug.json" if dlg.player_only_debug else "sportvu.json"
        p, _ = QFileDialog.getSaveFileName(self, "Export SportVU", default, "JSON (*.json)")
        if p:
            path, mp = write_sportvu(p, doc, meta)
            self.statusBar().showMessage(f"wrote {path} ({meta['kind']}, {meta['stats']['frames']} moments) + {mp.name}", 10000)

    # -------------------------------------------------------- calibration --
    def _start_pick(self, name: str):
        self._pick_name = name
        self.video.set_mode(VideoView.MODE_PICK)
        self.statusBar().showMessage(f"click the landmark '{name}' in the video (double-click resets zoom)")

    def _on_point_picked(self, x: float, y: float):
        if self._pick_name is None:
            return
        self.calib.add_point(self._pick_name, x, y)
        self._pick_name = None
        self.video.set_mode(VideoView.MODE_PAN)

    def _on_landmarks(self, lms):
        self._landmarks = list(lms)
        self._refresh_overlays()

    def _on_roi_selected(self, x0, y0, x1, y1):
        self.calib.set_roi((x0, y0, x1, y1))
        self.video.set_mode(VideoView.MODE_PAN)

    def _on_roi_changed(self, roi):
        self.session.roi = roi
        self._refresh_overlays()

    def _on_manual_calibration(self, cal: Optional[Calibration]):
        self.session.manual_calibration = cal
        self._refresh_overlays()

    def _auto_calibrate(self):
        frame = self.video.frame
        if frame is None:
            return
        boxes = None
        fb = self.session.local_boxes(self.video.frame_index)
        if fb is not None and fb.boxes:
            boxes = np.array([b.xyxy for b in fb.boxes])
        hint = self.calib.calibration.H if self.calib.calibration is not None else None
        self.statusBar().showMessage("auto-calibrating (a few seconds)...")
        QTimer.singleShot(0, lambda: self._run_auto(frame, boxes, hint))

    def _run_auto(self, frame, boxes, hint):
        try:
            cal = auto_calibrate(frame, self.session.roi, boxes, frame_index=self.video.frame_index, orientation_hint_H=hint)
        except Exception as e:
            self._error("Auto-calibration failed", e)
            return
        self.calib.set_calibration(cal if cal.valid else cal, "" if cal.valid else "(INVALID - not applied for mapping)")

    # ------------------------------------------------------------ adapter --
    def start_adapter(self):
        if self.adapter is not None:
            return
        try:
            self.adapter = AdapterServer(self.session.hub, port=self.adapter_port)
            self.adapter.start()
        except Exception as e:
            self.adapter = None
            self._error("Adapter failed to start", e)
            return
        self._update_status()

    def stop_adapter(self):
        if self.adapter is not None:
            self.adapter.stop()
            self.adapter = None
            self._update_status()

    def show_adapter_info(self):
        url = self.adapter.url if self.adapter else "(stopped)"
        QMessageBox.information(self, "Adapter", f"REST base: {url}/api/v1   WebSocket: ws://127.0.0.1:{self.adapter_port}/ws\nSee docs/api.md for endpoints and schemas.")

    def _on_hub_changed(self, kind: str, info):
        if kind in ("heatmap_meta", "heatmap_frame") and self.session.heatmap_source != "file":
            self.session.heatmap_source = "live"
        if kind == "sportvu":
            self.session.sportvu = None  # re-adopt the new live document lazily
        self._refresh_overlays()

    def closeEvent(self, event):  # noqa: N802
        self.playback.pause()
        self._stop_worker()
        self.stop_adapter()
        self.session.close()
        super().closeEvent(event)
