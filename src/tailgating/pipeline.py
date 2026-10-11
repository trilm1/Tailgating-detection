"""Offline video pipeline. GT is loaded only for scoring, never for inference."""

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2

from .config import load_config
from .crossing import CrossingDetector
from .events import EventAccumulator
from .evaluation import evaluate, label_warnings
from .io import read_grants, read_ground_truth, write_json
from .render import annotate_frame
from .video import validate_video


def hardware_info():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        names = [line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                 if line.startswith("model name")]
        cpu = names[0] if names else cpu
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    versions = {}
    for name in ("ultralytics", "torch", "opencv-python", "numpy", "lap"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), os=platform.platform(),
                python=platform.python_version(), packages=versions)


def run(video_path, grants_path, config_path, out, gt_path=None, scenarios_path=None,
        overwrite=False, detector_factory=None, tracker_factory=None):
    start_wall = time.perf_counter()
    video_path, out = Path(video_path).resolve(), Path(out).resolve()
    if not video_path.is_file():
        raise ValueError(f"Video does not exist: {video_path}")
    grants = read_grants(grants_path)
    gt = read_ground_truth(gt_path, grants) if gt_path else None
    scenarios = {}
    if scenarios_path is None and gt_path:
        candidate = Path(gt_path).with_name(Path(gt_path).name.removesuffix("_gt.csv") + "_scenarios.json")
        if candidate.exists():
            scenarios_path = candidate
    if scenarios_path:
        scenarios = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
        if not isinstance(scenarios, dict) or any(k not in {g.grant_id for g in grants}
                or type(v) is not int or not 1 <= v <= 10 for k, v in scenarios.items()):
            raise ValueError("Invalid scenario mapping: expected known grant IDs and numbers 1..10")
    names = ("events.json", "annotated.mp4", "metrics.json", "performance.json",
             "crossings.json", "config.used.json", "annotated.partial.mp4", "track_stitches.json")
    if not overwrite and any((out / name).exists() for name in names):
        raise ValueError(f"Output already exists: {out}. Choose another directory or use --overwrite.")
    if any(out / name == video_path for name in names):
        raise ValueError("Output must not overwrite the input video")
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video has invalid FPS or dimensions")
        config = load_config(config_path, width, height)
        cv2.setNumThreads(config["cpu_threads"])
        if detector_factory is None:
            from .detector import PersonDetector
            detector_factory = PersonDetector
        if tracker_factory is None:
            from .tracker import PersonTracker
            tracker_factory = PersonTracker
        detector = detector_factory(config)
        tracker = tracker_factory(fps, config)
        crossing_detector = CrossingDetector(**config["threshold"], max_gap_s=config["max_observation_gap_s"],
                                             stitch_radius_rel=config["stitch_radius_rel"],
                                             stitch_min_iou=config["stitch_min_iou"],
                                             min_box_height_px=config["min_box_height_frac"] * height,
                                             margin_rel=config["margin_rel"])
        accumulator = EventAccumulator(grants, config["grant_grace_s"], config["grant_lead_s"],
                                       config["grant_assignment"])
        out.mkdir(parents=True, exist_ok=True)
        partial = out / "annotated.partial.mp4"
        writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Cannot open MP4 video writer")
        processing_start = time.perf_counter()
        frames = fallbacks = 0
        previous_time = -1.0
        banner, banner_until = "", -1.0
        crossings = []
        snapshot_alerts = set()
        snapshots = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(raw_time) or raw_time <= previous_time:
                t_s = max(frames / fps, previous_time + 1 / fps)
                fallbacks += 1
            else:
                t_s = raw_time
            previous_time = t_s
            observations = tracker.update(detector.detect(frame), t_s, frame)
            alert_now = False
            for crossing in crossing_detector.update_frame(observations):
                crossings.append(asdict(crossing))
                accumulator.add(crossing)
                if crossing.direction == "in":
                    # Raise the banner once per newly alerting grant or intruder.
                    for index, event in enumerate(accumulator.records()):
                        key = event["grant_id"] if event["grant_id"] else f"unauthorised-{index}"
                        if event["alert"] != "none" and key not in snapshot_alerts:
                            snapshot_alerts.add(key)
                            banner = ("TAILGATING " + event["grant_id"] if event["grant_id"] else "UNAUTHORISED ENTRY")
                            banner_until = t_s + config["alert_hold_s"]
                            alert_now = True
            # Draw the counting ID, so a stitched track keeps its original label on screen.
            observations = [replace(o, track_id=crossing_detector.keys.get(o.track_id, o.track_id))
                            for o in observations]
            annotated = annotate_frame(frame, observations, config, t_s, accumulator,
                                       banner if t_s <= banner_until else "")
            writer.write(annotated)
            if alert_now:
                snapshots.append((f"alert_{frames:06d}.jpg", annotated.copy()))
            frames += 1
            if frames % 250 == 0:
                speed = frames / (time.perf_counter() - processing_start)
                print(f"{video_path.name[:13]}: {frames}/{expected_frames} frames, {speed:.1f} FPS", flush=True)
        if frames == 0 or (expected_frames > 0 and frames < expected_frames - 1):
            raise ValueError(f"Incomplete decode: {frames}/{expected_frames} frames")
        writer.release()
        writer = None
        processing_seconds = time.perf_counter() - processing_start
        cap.release()
        validation_start = time.perf_counter()
        validate_video(partial, frames, (width, height))
        validation_seconds = time.perf_counter() - validation_start
        events = accumulator.records()
        performance = dict(frames=frames, input_fps=fps, width=width, height=height,
                           processing_seconds=processing_seconds, processing_fps=frames/processing_seconds,
                           output_validation_seconds=validation_seconds, output_validated=True,
                           timestamp_fallback_frames=fallbacks, timestamp_source="OpenCV POS_MSEC; index/FPS fallback",
                           device=config["device"], hardware=hardware_info(),
                           model_sha256=hashlib.sha256(Path(config["model"]).read_bytes()).hexdigest()
                           if Path(config["model"]).exists() else None,
                           fps_scope="decode, detect, track, count, render, write and finalize video; model loading and output validation excluded")
        warnings = label_warnings(grants, gt, scenarios) if gt is not None else []
        metrics = evaluate(events, grants, gt, scenarios) if gt is not None else None
        write_json(out / "events.json", events)
        write_json(out / "crossings.json", crossings)
        write_json(out / "track_stitches.json", crossing_detector.stitches)
        write_json(out / "config.used.json", config)
        partial.replace(out / "annotated.mp4")
        # Remove stale optional metrics on explicit overwrite without ground truth.
        if gt is None and overwrite:
            (out / "metrics.json").unlink(missing_ok=True)
        for name, frame in snapshots:
            (out / "snapshots").mkdir(exist_ok=True)
            cv2.imwrite(str(out / "snapshots" / name), frame)
        performance["total_seconds"] = time.perf_counter() - start_wall
        performance["total_fps"] = frames / performance["total_seconds"]
        write_json(out / "performance.json", performance)
        if metrics is not None:
            metrics.update(performance=performance, label_warnings=warnings)
            write_json(out / "metrics.json", metrics)
        return dict(out=str(out), performance=performance, metrics=metrics)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

