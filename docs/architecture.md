# Architecture

```
            ┌──────────────── processing worker (btrv process / GUI) ────────────────┐
 video ──▶ FileSource ──▶ camera mode ──▶ Detector ──▶ Tracker ──▶ TeamAssigner ──▶ court mapping ──▶ RunStore
            (PyAV,         fixed: ROI       pinned        ByteTrack    jersey k-means    homography       manifest.json
             pts index)    moving: full     RF-DETR       (trackers)   over tracks       per frame        raw_boxes.jsonl
                                            local/hosted                                                 tracked_boxes.jsonl
                                                                                                         homographies.npz
                                                                                                         teams.json, summary.json
            └────────────────────────────────────────────────────────────────────────┘
                                                      │
 external tools ──(files / REST / WebSocket, same schemas)──▶ DataHub ◀──▶ Viewer (PySide6)
                                                                  │          video + overlays, playback,
                                                                  └────────▶ synchronised court view
```

## Replaceable modules

| role | interface | shipped implementations |
|---|---|---|
| source | `btrv.source.FrameSource` (`read(i)`, `next()`, frame identity) | `FileSource` (PyAV). A live source implements the same interface; consumers only use frame index + timestamp. |
| detector | `btrv.detect.Detector.detect(image) -> [Detection]` + `describe()` | `RoboflowLocalDetector`, `RoboflowHostedDetector`, `FakeDetector`, `BoxesFileDetector` |
| tracker | `btrv.track.Tracker.update(dets, frame_index, timestamp_s)` | `ByteTrackTracker` (`trackers`), `NullTracker` |
| team assignment | `btrv.team.TeamAssigner` | jersey-colour k-means |
| court mapper | `btrv.court` (`fit_homography`, `auto_calibrate`, `MovingCalibrator`) | manual landmarks, automatic line-based, frame-to-frame propagation |
| storage | `btrv.storage.RunStore`, `BoxesTable` | JSONL run directory + columnar in-memory table |
| visualisation | `btrv.viewer.video_widget.VideoView` layers, `court_widget.CourtView` | Qt widgets; overlays receive a `FrameOverlays` bundle per displayed frame |

## Key design decisions

* **Synchronisation** uses source frame index + decoded pts everywhere (records, heat maps, SportVU
  moment times, playback clock). The playback controller advances *media time* with wall time × speed
  and shows the last frame whose pts ≤ media time, so all six speeds and VFR files stay in sync; slow
  decoding drops frames instead of drifting.
* **Memory**: frames are decoded on demand (byte-budgeted LRU cache, default 1.5 GB) — never the whole
  file; heat maps are read per frame from HDF5; boxes live in numpy columns, not Python objects.
* **Coordinates**: inference may run on an ROI crop and a downscaled image; `InferenceRegion` maps
  results back to full-frame pixels before anything is stored. Foot points (bottom-centre) are projected,
  not box centres; invalid/behind-camera/off-court points are flagged, never guessed.
* **Reproducibility**: `manifest.json` records the exact model id, runtime and device, tracker and team
  parameters, ROI and inference region, calibration, conventions and library versions.
* **Precompute before playback**: the worker writes everything to disk; the viewer only decodes video
  and looks up per-frame data, so playback stays responsive at any speed.
* **Threads**: decoding runs in a `DecodeWorker` QThread (coalescing requests, prefetching); the adapter
  runs uvicorn in a daemon thread and hands data to the GUI through a Qt signal; processing runs in a
  `ProcessingThread` with progress and cancellation.

## Moving cameras

`MovingCalibrator` chains frame-to-frame homographies (ORB features on the court surface with player
boxes masked out) from the last valid calibration, decays confidence per frame, and every
`revalidate_every` frames refines against court-line evidence; on failure it re-runs automatic
calibration with the previous homography as orientation hint. Frames that cannot be calibrated are
stored with `calibration.valid = false` and no court positions.
