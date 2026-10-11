# Tailgating Detection

Detect tailgating at an access-controlled door by counting entries per access grant.
The system raises a `tailgating` alert when more than one person enters on a grant and an
`unauthorised_entry` alert when someone enters outside all grant windows.
Built with YOLO11m and ByteTrack, using Python 3.10 and offline CPU inference.

## Setup and Usage

```bash
python -m pip install -r requirements.txt
python detect_tailgating.py --video clips/1791534560797_2212977288714148921_808826234038601398.mp4 --grants clips/1791534560797_2212977288714148921_808826234038601398_grants.csv --config configs/1791534560797_2212977288714148921_808826234038601398.yaml --out results/1791534560797_2212977288714148921_808826234038601398
python -m unittest discover -s tests -v
```

Add `--gt <clip>_gt.csv` to write `metrics.json`. The scenario mapping
`<clip>_scenarios.json` is loaded automatically when it is located beside the GT file.
Add `--overwrite` if the output directory already contains results.
Model weights are included in `models/`; inference does not require network access.

## Repository Structure

| Path | Contents |
|---|---|
| `src/tailgating/` | Detection, tracking, crossing logic, grant association, rendering, and evaluation |
| `detect_tailgating.py` | CLI matching the assignment specification |
| `run_dataset.py`, `evaluate.py` | Run and evaluate multiple clips; evaluate existing `events.json` files |
| `calibrate_door.py` | Click-based tool for selecting the threshold and entry side |
| `configs/` | One `<clip>.yaml` per video, containing threshold geometry, model path, and processing parameters |
| `clips/` | Videos and labels for 9 clips, plus annotation-tool project files; download videos using the link below |
| `results/` | Per-clip results, metrics tables, and failure images (`failures/`) |
| `tests/` | Unit tests and synthetic scenario tests |

## Camera Configuration

```bash
python calibrate_door.py --video clips/<clip>.mp4 --time 5 --out configs/<clip>.yaml
```

Click the two threshold endpoints, then a point on the inside of the door. Press Enter to
save. The configuration is rejected if `reference_size` differs from the video resolution.

| Parameter | Default | Description |
|---|---|---|
| `threshold.start/end`, `in_side` | — | Threshold segment in pixels and entry side (+1/-1) |
| `threshold.margin_px` | Tool: 1.25% of image height | Dead band to reduce bounding-box jitter (pixels) |
| `margin_rel` | 0.03 | Minimum dead band relative to bounding-box height |
| `threshold.mode` | `footprint` | `footprint`: the entire bottom edge must cross; `point`: bottom center |
| `grant_grace_s` / `grant_lead_s` | 0.5 / 0.1 s | Extend the window after relock / before unlock |
| `grant_assignment` | `unfilled_first` | For overlapping grants, prioritize the oldest grant with no assigned entry |
| `stitch_radius_rel`, `stitch_min_iou` | 0.35 / 0.5 | Repair tracker ID switches; radius relative to bounding-box height; 0 disables stitching |
| `min_box_height_frac` | 0 | Discard boxes shorter than this fraction of image height (overhead camera) |
| `max_observation_gap_s`, `tracker_buffer_s` | 1.0 s | Do not infer crossings across longer observation gaps |
| `model`, `imgsz`, `confidence`, `cpu_threads` | `yolo11m.pt`, 640, 0.1, 4 | Local model, input size, detector threshold, and CPU thread count |
| `track_high_thresh`, `new_track_thresh` | 0.25 / 0.25 | ByteTrack: lower-score detections only associate with existing tracks |

The five `configs/179153*.yaml` files use `point` because feet are cut off at the image boundary, together with
`min_box_height_frac: 0.25`.
The four `configs/179169*.yaml` files use `footprint` and their individually calibrated thresholds.

To run all 9 clips and generate aggregate metrics and scenario confusion tables:

```bash
python run_dataset.py --videos clips --labels clips --config-dir configs --out results/final
```

Video, label, and configuration filenames share the same clip stem. Each video must have
exactly one matching configuration; there is no fallback to another camera's threshold.

Reuse a configuration for a new video only when the camera, viewpoint, crop, and resolution
are unchanged. Matching resolution alone does not imply matching door geometry. For a new
camera, run the calibration tool, start with `min_box_height_frac: 0`, and choose a view
that shows people's feet when using `footprint`. The `footprint` mode remains a bounding-box bottom-edge
approximation, not a test of full-body passage in 3D.

The batch runner defaults to `clips/` for videos and labels and `configs/` for per-clip
configurations. It also accepts `.json` and `.yml`. Use `--config <path>` only when deliberately
sharing one camera configuration across all selected clips; it cannot be combined with `--config-dir`.
FPS is read from the video. Timestamps use OpenCV POS_MSEC, with frame index/FPS as a fallback.
The output MP4 uses a constant frame rate, so its playback timing may differ from a variable-frame-rate
source. Constant-frame-rate video is preferable for evaluation and annotation alignment.

## Outputs (`--out`)

- `events.json`: one record per grant, including `entries: 0`, and one record per unauthorised entry.
- `annotated.mp4`: track IDs, threshold and entry-direction arrow, per-grant entry counts, and an alert banner (MPEG-4; VLC is recommended for playback).
- `metrics.json` (with `--gt`): tailgating P/R/F1, count accuracy, confusion by scenario, unauthorised-entry metrics, FPS, and hardware details.
- Additional files: `performance.json`, `crossings.json`, `track_stitches.json`, `config.used.json`, and `snapshots/`.

The pipeline decodes the entire exported video and checks its frame count and dimensions
before finalizing the output. `processing_fps` excludes model loading and output validation;
`total_fps` includes both. Check copied or downloaded outputs before submission:

```bash
python validate_output.py --video clips/<clip>.mp4 --grants clips/<clip>_grants.csv --out results/<clip>
```

This command checks the event schema, crossing counts, and grant IDs, and decodes both the
source and annotated videos. It does not visually assess detection quality. `evaluate.py`
also rejects events that do not conform to the schema.

Scenario 6 has no crossings, so GT contains only the header. If a grant exists, map it to
scenario `6`. Clip …9613 has no grants: its grants file also contains only the header, and
the expected output is `[]`. Walk-past events are documented separately in DATASET and
excluded from grant-level metrics.
For scenario 9, an `in` crossing has an empty `grant_id`; even if the entire clip has no
grants, the grants CSV still requires its header.

## Models and Libraries

| Component | Source | License |
|---|---|---|
| YOLO11m pretrained on COCO | [ultralytics/assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt), SHA256 `d5ffc1a6…f5305b95` | AGPL-3.0 |
| Ultralytics (detector, ByteTrack) | [v8.3.70](https://github.com/ultralytics/ultralytics/tree/v8.3.70) | AGPL-3.0 |
| ByteTrack (algorithm) | [Zhang et al., 2022](https://arxiv.org/abs/2110.06864) | — |

Dependencies are pinned in `requirements.txt`. No ReID model, pose model, or cloud API is used.

## Videos

Original videos and `annotated.mp4` files (excluded from the repository because of their size): <link>

## Use of AI Coding Assistants

AI assistants (Codex and Claude Code) were used to help write code and check logic.
See section 6 of [REPORT.md](REPORT.md) for details.
