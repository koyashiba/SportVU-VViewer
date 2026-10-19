# Roboflow asset status (inspected 2026-10-19 via the Roboflow connector)

| asset | finding | consequence |
|---|---|---|
| Model `bachelor-nozvk/basketball-player-detection-jwk5o-t33zb-2-rfdetr-small-t1` | RF-DETR small, version 2 of project *Basketball Player Detection* (800 images, single class `players`), training finished, mAP50 98.87 / P 97.6 / R 98.2 | pinned as `DetectorConfig.model_id`; used directly by both the local and hosted backends |
| Hosted inference of the pinned model on a 1920×1080 dataset image, confidence 0.3 | 10 `players` boxes (class_id 0), centre-based `x, y, width, height`, served from an ONNX fp32 package | parser verified in `tests/test_hosted_parser.py` |
| Workflow `Basketball Player Model Comparison` (`8N8ClrbW7q72knk2RCdb`) | `config` (saved) = detector @0.3 + box/label visualisation, outputs `output_image`, `predictions`. `lastVersionConfig` (draft) adds `trackers_bytetrack@v1` (`minimum_consecutive_frames: 1`) and outputs `tracked_players`, `tracked_image`. The two differ, confirming the tracking additions are not published. | the app does not depend on the Workflow; tracking runs locally with the same ByteTrack implementation (`trackers`). If the Workflow is published later, a `roboflow_workflow` detector backend could be added behind the same `Detector` interface. |
| Project *Basketball Tracking Detection* | live endpoint serves pretrained `rfdetr-nano`, `selectedModelId: null` | untouched |
| Project *Basketball Player Detection* deployment | base workflow, fine-tuned model `basketball-player-detection-jwk5o-t33zb/2` | untouched |
