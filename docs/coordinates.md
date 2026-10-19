# Coordinate conventions

## Synchronisation keys

Every record btrv stores or exchanges carries two keys taken from the **source video**:

| key | meaning |
|---|---|
| `frame_index` | 0-based index of the frame in *presentation order* (after B-frame reordering). |
| `timestamp_s` | decoded presentation timestamp in seconds: `pts * time_base` of the video stream. |

A constant frame rate is never assumed. The frame index is built by demuxing the file once
(no decoding), sorting packets by pts, and caching the result (`~/.cache/btrv/frame_index`,
`%LOCALAPPDATA%\btrv` on Windows, or `$BTRV_CACHE_DIR`). The run manifest records
`time_base`, the first pts and the frame count, so external software can reproduce the mapping.
Timestamps are raw stream time (not shifted so that the first frame is 0); the first timestamp is
in the manifest under `source.extra.first_pts`.

## Image space: `image_pixels_xyxy_topleft`

* Origin at the top-left corner of the **full decoded frame**, x to the right, y down, units: pixels (floats allowed).
* Boxes are `[x1, y1, x2, y2]` with `x2 > x1`, `y2 > y1`.
* If inference ran on an ROI crop and/or a downscaled image, detections are transformed back with
  `source = offset + inference_pixel / scale` **before** they are stored or displayed. The exact
  `inference_region` (crop and scale) is stored in `manifest.json`.
* The **ground-contact point** of a player is the bottom-centre of the box
  (`((x1+x2)/2, y2)`) unless an explicit `foot_xy` is present. It is the point projected to the court.

## Court space: `sportvu_feet_v1`

* 94 × 50 ft court, units feet.
* Origin `(0, 0)` is the **top-left corner of the court as drawn by the reference viewer**
  ([linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements)):
  `x` runs along the 94 ft length (left baseline `x = 0`, right baseline `x = 94`),
  `y` runs across the 50 ft width and **increases downwards** in that drawing (top sideline `y = 0`,
  bottom sideline `y = 50`). The reference viewer draws its court image with
  `extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN] = [0, 94, 50, 0]`, which is exactly this frame, so
  **no conversion is applied** when exporting to or importing from that format.
* `z` is height in feet (players `0.0`; the ball's `z` is whatever the external ball data supplies).
* Left basket centre `(5.25, 25)`, right basket `(88.75, 25)`; the free-throw line is 19 ft from the
  baseline, the key is 16 ft wide, the three-point arc radius is 23.75 ft with corner lines 3 ft from
  the sidelines (NBA dimensions, `btrv.court.model.CourtModel`).
* **Team orientation / which end is which**: btrv cannot know from pixels which real end of the
  arena is `x = 0`. The user decides during calibration by the landmark names they click
  (e.g. `corner_top_left`). Automatic calibration from court lines is only determined up to the
  court's symmetry (ends flipped `x -> 94 - x`, sides flipped `y -> 50 - y`); the result is flagged
  `orientation_ambiguous` and the panel offers *Flip ends* / *Flip sides*. A manual calibration or the
  previous frame is used as an orientation hint whenever one exists.
* Points whose homogeneous `w <= 0` (behind the camera / above the horizon) are **invalid**; points more
  than `court_margin_ft` (default 2 ft) outside the court are marked `valid: false` and are excluded
  from the top-down view and from the SportVU export. Positions are never invented for frames without a
  valid calibration; such frames carry `calibration.valid = false`.

## Homography

`H` (3×3) maps image pixels to court feet: `[x_ft, y_ft, 1]^T ~ H [u_px, v_px, 1]^T`.
The inverse maps court to image (used for the court-line overlay). Calibration files store `H`,
the method (`manual`, `auto`, `propagated`, `refined`), the landmarks used, the court->image
reprojection RMSE in pixels, the inlier ratio and a confidence in `[0, 1]`.
