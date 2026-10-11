import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from run_dataset import clip_config
from src.tailgating.domain import Crossing, Grant
from src.tailgating.events import EventAccumulator
from src.tailgating.evaluation import evaluate
from src.tailgating.video import validate_video


class SubmissionTests(unittest.TestCase):
    def test_invalid_exports_cannot_receive_valid_scores(self):
        grants = [Grant("g1", 0, 5)]
        accumulator = EventAccumulator(grants)
        accumulator.add(Crossing(1, 2, "in"))
        accumulator.add(Crossing(2, 9, "in"))
        valid = accumulator.records()
        evaluate(valid, grants, [])
        for field, value in [("entries", 2), ("t_unlock_s", 10), ("crossings", []), ("confidence", float("nan"))]:
            with self.subTest(field=field):
                events = copy.deepcopy(valid)
                events[0][field] = value
                with self.assertRaises(ValueError):
                    evaluate(events, grants, [])
        events = copy.deepcopy(valid)
        events[1]["alert"] = "none"
        with self.assertRaises(ValueError):
            evaluate(events, grants, [])

    def test_per_clip_config_does_not_silently_use_another_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = Path("camera_b.mp4")
            with self.assertRaises(ValueError):
                clip_config(video, None, root)
            (root / "camera_b.json").write_text("{}")
            self.assertEqual(clip_config(video, None, root), root / "camera_b.json")
            (root / "camera_b.yaml").write_text("{}")
            with self.assertRaises(ValueError):
                clip_config(video, None, root)

    def test_output_video_must_decode_all_expected_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            self.assertEqual(validate_video(path, 3, (64, 48)), 3)
            with self.assertRaises(ValueError):
                validate_video(path, 4)
            with self.assertRaises(ValueError):
                validate_video(path, 3, (128, 96))
            path.write_bytes(b"broken MP4")
            with self.assertRaises(ValueError):
                validate_video(path)
