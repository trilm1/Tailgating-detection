import tempfile
import unittest
from pathlib import Path

from src.tailgating.crossing import CrossingDetector
from src.tailgating.domain import Crossing, Grant, GroundTruth, Observation
from src.tailgating.events import EventAccumulator
from src.tailgating.evaluation import evaluate
from src.tailgating.io import read_grants, read_ground_truth


def observation(t, y, x=50, track=1):
    return Observation(track, t, (x - 5, y - 20, x + 5, y))


class CrossingTests(unittest.TestCase):
    def setUp(self):
        self.detector = CrossingDetector((0, 100), (100, 100), margin_px=5)

    def test_entry_and_exit(self):
        self.assertIsNone(self.detector.update(observation(0, 80)))
        entry = self.detector.update(observation(0.5, 120))
        self.assertEqual(entry.direction, "in")
        self.assertAlmostEqual(entry.t_cross_s, 0.25)
        self.assertEqual(self.detector.update(observation(1, 80)).direction, "out")

    def test_hesitation(self):
        for t, y in [(0, 80), (0.1, 99), (0.2, 102), (0.3, 80)]:
            self.assertIsNone(self.detector.update(observation(t, y)))
        self.assertEqual(self.detector.update(observation(0.5, 120)).direction, "in")

    def test_beyond_segment_not_counted(self):
        self.detector.update(observation(0, 80, x=150))
        self.assertIsNone(self.detector.update(observation(0.5, 120, x=150)))

    def test_short_occlusion(self):
        self.detector.update(observation(0, 80))
        self.assertIsNotNone(self.detector.update(observation(1, 120)))

    def test_long_gap_not_inferred(self):
        self.detector.update(observation(0, 80))
        self.assertIsNone(self.detector.update(observation(2, 120)))

    def test_reverse_in_side(self):
        detector = CrossingDetector((0, 100), (100, 100), in_side=-1)
        detector.update(observation(0, 120))
        self.assertEqual(detector.update(observation(1, 80)).direction, "in")

    def test_loitering_does_not_backdate_crossing(self):
        self.detector.update(observation(0, 80))
        for tick in range(1, 11):
            self.assertIsNone(self.detector.update(observation(tick / 2, 99)))
        self.assertIsNone(self.detector.update(observation(5.5, 101)))
        event = self.detector.update(observation(6, 120))
        self.assertAlmostEqual(event.t_cross_s, 5.25)

    def test_exact_zero_and_retreat_does_not_count(self):
        for t, y in [(0, 80), (.1, 100), (.2, 101), (.3, 99), (.4, 80)]:
            self.assertIsNone(self.detector.update(observation(t, y)))


class EventTests(unittest.TestCase):
    def test_unfilled_first_for_back_to_back_grants(self):
        engine = EventAccumulator([Grant("g1", 0, 5), Grant("g2", 1, 6)],
                                  assignment="unfilled_first")
        engine.add(Crossing(1, 2, "in"))
        engine.add(Crossing(2, 3, "in"))
        self.assertEqual([r["entries"] for r in engine.records()], [1, 1])

    def test_reentry_not_recounted_across_overlapping_grants(self):
        engine = EventAccumulator([Grant("g1", 0, 5), Grant("g2", 1, 6)],
                                  assignment="unfilled_first")
        for crossing in [Crossing(1, 2, "in"), Crossing(1, 3, "out"), Crossing(1, 4, "in")]:
            engine.add(crossing)
        self.assertEqual([r["entries"] for r in engine.records()], [1, 0])

    def test_lead_tolerance_before_unlock(self):
        engine = EventAccumulator([Grant("g1", 1, 5)], lead_s=0.2)
        engine.add(Crossing(1, 0.9, "in"))
        engine.add(Crossing(2, 0.7, "in"))
        self.assertEqual([r["alert"] for r in engine.records()], ["none", "unauthorised_entry"])

    def test_deduplication_and_unauthorised(self):
        engine = EventAccumulator([Grant("g1", 1, 5), Grant("g2", 10, 12)])
        for crossing in [Crossing(1, 2, "in"), Crossing(1, 3, "out"),
                         Crossing(1, 4, "in"), Crossing(2, 5.4, "in"),
                         Crossing(3, 8, "in")]:
            engine.add(crossing)
        records = engine.records()
        self.assertEqual([r["entries"] for r in records], [2, 0, 1])
        self.assertEqual([r["alert"] for r in records],
                         ["tailgating", "none", "unauthorised_entry"])

    def test_new_active_grant_beats_old_grace(self):
        engine = EventAccumulator([Grant("g1", 0, 3), Grant("g2", 3.1, 6)])
        engine.add(Crossing(1, 3.2, "in"))
        self.assertEqual([r["entries"] for r in engine.records()], [0, 1])

    def test_overlapping_grants_choose_latest(self):
        engine = EventAccumulator([Grant("g1", 0, 5), Grant("g2", 2, 6)])
        engine.add(Crossing(1, 3, "in"))
        self.assertEqual([r["entries"] for r in engine.records()], [0, 1])

    def test_entry_before_unlock_is_unauthorised(self):
        engine = EventAccumulator([Grant("g1", 1, 5)])
        engine.add(Crossing(1, 0.9, "in"))
        self.assertEqual(engine.records()[-1]["alert"], "unauthorised_entry")


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts_and_repeat_person(self):
        grants = [Grant(f"g{i}", i * 10, i * 10 + 5) for i in range(4)]
        gt = [GroundTruth(f"p{j}", i * 10 + j, "in", f"g{i}")
              for i, count in enumerate([2, 1, 2, 1]) for j in range(count)]
        gt.append(GroundTruth("p0", 4, "in", "g0"))
        engine = EventAccumulator(grants)
        for i, count in enumerate([2, 2, 1, 1]):
            for j in range(count):
                engine.add(Crossing(j, i * 10 + j, "in"))
        metrics = evaluate(engine.records(), grants, gt, {"g0": 2})
        self.assertEqual(metrics["tailgating"], dict(precision=0.5, recall=0.5, f1=0.5))
        self.assertEqual(metrics["count_accuracy"], 0.5)
        self.assertEqual(metrics["by_scenario"]["2"]["tp"], 1)

    def test_missing_prediction_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [Grant("g1", 0, 1)], [])

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grants.csv"
            path.write_text("grant_id,t_unlock_s,t_relock_s\ng1,0,5\n", encoding="utf-8")
            grants = read_grants(path)
            gt = Path(directory) / "gt.csv"
            gt.write_text("person_id,t_cross_s,direction,grant_id\np1,2,in,missing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_ground_truth(gt, grants)
            path.write_text("grant_id,t_unlock_s,t_relock_s\ng1,nan,5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_grants(path)


if __name__ == "__main__":
    unittest.main()
