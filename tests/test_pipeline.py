import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.tailgating.config import load_config
from src.tailgating.crossing import CrossingDetector
from src.tailgating.domain import Observation
from src.tailgating.evaluation import match_times
from src.tailgating.pipeline import run


class EvaluationMatchingTests(unittest.TestCase):
    def test_matching_is_one_to_one(self):
        result = match_times([1, 1.1, 10], [1.05, 20], .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 1))

    def test_footprint_must_clear_line(self):
        detector = CrossingDetector((50, 0), (50, 100), in_side=-1, margin_px=2, mode="footprint")
        self.assertIsNone(detector.update(Observation(1, 0, (20, 10, 40, 50))))
        self.assertIsNone(detector.update(Observation(1, .1, (45, 10, 65, 50))))
        self.assertEqual(detector.update(Observation(1, .2, (55, 10, 75, 50))).direction, "in")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "clip.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        for _ in range(20):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        self.grants = self.root / "clip_grants.csv"
        self.grants.write_text("grant_id,t_unlock_s,t_relock_s\ng01,0,0.9\n", encoding="utf-8")
        self.gt = self.root / "clip_gt.csv"
        self.gt.write_text("person_id,t_cross_s,direction,grant_id\np1,0.5,in,g01\np2,0.5,in,g01\np3,1.65,in,\n", encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(dict(reference_size=[160, 120], threshold=dict(
            start=[0, 60], end=[159, 60], in_side=1, margin_px=2), grant_grace_s=0)), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_output_contract_and_gt_not_needed_for_inference(self):
        class FakeDetector:
            def __init__(self, config):
                pass

            def detect(self, frame):
                return None

        class FakeTracker:
            def __init__(self, fps, config):
                self.index = 0

            def update(self, detections, t_s, frame):
                i = self.index
                self.index += 1
                if i < 10:
                    y = 40 if i < 5 else 80
                    return [Observation(track, t_s, (x, y - 20, x + 10, y)) for track, x in [(1, 20), (2, 80)]]
                y = 40 if i < 17 else 80
                return [Observation(3, t_s, (50, y - 20, 60, y))]

        out = self.root / "result"
        result = run(self.video, self.grants, self.config, out, self.gt,
                     detector_factory=FakeDetector, tracker_factory=FakeTracker)
        events = json.loads((out / "events.json").read_text())
        self.assertEqual(events[0]["entries"], 2)
        self.assertEqual(events[0]["alert"], "tailgating")
        self.assertEqual(events[1]["alert"], "unauthorised_entry")
        self.assertEqual(result["metrics"]["tailgating"]["f1"], 1)
        self.assertEqual(result["metrics"]["unauthorised_entry"]["f1"], 1)
        capture = cv2.VideoCapture(str(out / "annotated.mp4"))
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 20)
        capture.release()
        run(self.video, self.grants, self.config, out, overwrite=True,
            detector_factory=FakeDetector, tracker_factory=FakeTracker)
        self.assertFalse((out / "metrics.json").exists())
        self.assertEqual(json.loads((out / "events.json").read_text()), events)

    def test_config_resolution_mismatch(self):
        with self.assertRaises(ValueError):
            load_config(self.config, 854, 480)

    def test_config_nan_rejected(self):
        data = json.loads(self.config.read_text())
        data["threshold"]["margin_px"] = float("nan")
        self.config.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            load_config(self.config, 160, 120)

    def test_threshold_can_reach_continuous_image_boundary(self):
        data = json.loads(self.config.read_text())
        data["threshold"]["start"] = [50, 120]
        data["threshold"]["end"] = [50, 0]
        self.config.write_text(json.dumps(data))
        config = load_config(self.config, 160, 120)
        detector = CrossingDetector(**config["threshold"])
        detector.update(Observation(1, 0, (30, 80, 40, 120)))
        event = detector.update(Observation(1, .1, (60, 80, 70, 120)))
        self.assertEqual(event.direction, "in")


class ByteTrackTests(unittest.TestCase):
    def test_short_occlusion_keeps_id(self):
        from ultralytics.engine.results import Boxes
        from src.tailgating.tracker import PersonTracker

        tracker = PersonTracker(25, dict(tracker_buffer_s=1.0, track_high_thresh=0.25, new_track_thresh=0.25))
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        detected = Boxes(np.array([[30, 20, 60, 100, .9, 0]], dtype=np.float32), (120, 160))
        empty = Boxes(np.empty((0, 6), dtype=np.float32), (120, 160))
        first = tracker.update(detected, 0, frame)[0].track_id
        for i in range(1, 21):
            tracker.update(empty, i/25, frame)
        last = tracker.update(detected, 21/25, frame)[0].track_id
        self.assertEqual(first, last)


if __name__ == "__main__":
    unittest.main()
