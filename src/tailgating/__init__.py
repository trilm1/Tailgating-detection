"""Offline tailgating detection: model-independent event processing."""

import importlib.metadata
import os
from pathlib import Path

# Keep Ultralytics settings per version inside the project (schemas differ by version).
try:
    _version = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    _version = "uninstalled"
os.environ.setdefault("YOLO_CONFIG_DIR", str(
    Path(__file__).resolve().parents[2] / ".cache" / f"ultralytics-{_version}"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
