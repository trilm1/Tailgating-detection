"""Person-only YOLO inference from a pre-downloaded local checkpoint."""

import os
from pathlib import Path

# Must precede Ultralytics imports: no dependency auto-install or online checks.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"


class PersonDetector:
    def __init__(self, config):
        model_path = Path(config["model"])
        if not model_path.is_file() or model_path.suffix != ".pt":
            raise ValueError(f"Local .pt weights required: {model_path}")
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(config["cpu_threads"])
        self.model = YOLO(str(model_path), task="detect")
        if self.model.names.get(0) != "person":
            raise ValueError("Expected COCO weights with person at class 0")
        self.options = dict(classes=[0], conf=config["confidence"], iou=config["iou"],
                            imgsz=config["imgsz"], device=config["device"],
                            verbose=False, save=False)

    def detect(self, frame):
        return self.model.predict(frame, **self.options)[0].boxes.cpu().numpy()
