"""Validated per-camera configuration. Paths are relative to the config file."""

import json
import math
from pathlib import Path


def load_config(path, width, height):
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    size = config.get("reference_size")
    if size != [width, height]:
        raise ValueError(f"Config reference_size={size}; video is {[width, height]}. Recalibrate the threshold.")
    threshold = config.get("threshold", {})
    if not isinstance(threshold, dict):
        raise ValueError("threshold must be an object")
    for key in ("start", "end"):
        point = threshold.get(key)
        if not isinstance(point, list) or len(point) != 2 or any(
                type(v) not in (float, int) or not math.isfinite(v) for v in point):
            raise ValueError(f"threshold.{key} must be two finite pixel coordinates")
        # Boxes use continuous boundary coordinates: a bottom edge can equal height.
        if not (0 <= point[0] <= width and 0 <= point[1] <= height):
            raise ValueError(f"threshold.{key} is outside the image")
    if threshold["start"] == threshold["end"] or threshold.get("in_side") not in (-1, 1):
        raise ValueError("Nonzero threshold and in_side -1 or 1 are required")
    defaults = {"grant_grace_s": 0.5, "max_observation_gap_s": 1.0,
                "tracker_buffer_s": 1.0, "imgsz": 640, "confidence": 0.1,
                "track_high_thresh": 0.25, "new_track_thresh": 0.25,
                "iou": 0.7, "cpu_threads": 4, "alert_hold_s": 2.0,
                # Camera-independent defaults; min_box_height_frac is camera specific, off.
                "grant_lead_s": 0.1, "stitch_radius_rel": 0.35, "stitch_min_iou": 0.5, "margin_rel": 0.03,
                "min_box_height_frac": 0.0}
    for key, default in defaults.items():
        config.setdefault(key, default)
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    for key in ("grant_grace_s", "alert_hold_s", "grant_lead_s", "stitch_radius_rel", "margin_rel"):
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("max_observation_gap_s", "tracker_buffer_s"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("imgsz", "cpu_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    if not 0 < config["confidence"] <= config["track_high_thresh"] <= 1 or not 0 < config["iou"] <= 1:
        raise ValueError("Need 0 < confidence <= track_high_thresh <= 1 and iou in (0, 1]")
    if not 0 < config["new_track_thresh"] <= 1:
        raise ValueError("new_track_thresh must be in (0, 1]")
    if not 0 < config["stitch_min_iou"] <= 1 or not 0 <= config["min_box_height_frac"] < 1:
        raise ValueError("stitch_min_iou must be in (0, 1]; min_box_height_frac in [0, 1)")
    config.setdefault("grant_assignment", "unfilled_first")
    if config["grant_assignment"] not in ("latest", "unfilled_first"):
        raise ValueError("grant_assignment must be latest or unfilled_first")
    margin = threshold.setdefault("margin_px", 8.0)
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise ValueError("margin_px must be finite and nonnegative")
    mode = threshold.setdefault("mode", "footprint")
    if mode not in ("footprint", "point"):
        raise ValueError("threshold.mode must be footprint or point")
    model = config.get("model", "../models/yolo11m.pt")
    if not isinstance(model, str) or "://" in model:
        raise ValueError("model must be a local path")
    config["model"] = str((path.parent / model).resolve())
    config["device"] = str(config.get("device", "cpu"))
    return config
