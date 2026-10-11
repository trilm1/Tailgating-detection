"""Camera-independent scenario tests for crossing + grant logic.

Trajectories are written in door coordinates: u runs along the threshold (0..1 is the
door segment) and v is the signed distance from the line in person heights (+ = inside).
Each scenario runs on several cameras (line orientation, resolution, which side is in)
and frame rates, so the logic cannot depend on the recorded development clips.
"""

from math import hypot
import unittest

from src.tailgating.crossing import CrossingDetector
from src.tailgating.domain import Grant, Observation
from src.tailgating.events import EventAccumulator

CAMERAS = {
    # name: line start, line end, in_side, person height at the door (px), margin (px)
    "horizontal_640x480": ((120, 300), (520, 300), 1, 220, 6),
    "steep_854x480": ((350, 480), (400, 280), 1, 260, 6),
    "diagonal_1920x1080_in_left": ((700, 1000), (1300, 500), -1, 520, 14),
}
FPS = (10, 25, 30)
STITCH = dict(stitch_radius_rel=0.35, stitch_min_iou=0.5)


def lerp(points, t):
    """Piecewise-linear (t, u, v) keyframes; None outside the visible interval."""
    if t < points[0][0] or t > points[-1][0]:
        return None
    for (t0, u0, v0), (t1, u1, v1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return u0 + f * (u1 - u0), v0 + f * (v1 - v0)
    return None


class Camera:
    def __init__(self, name):
        self.name = name
        self.start, self.end, self.in_side, self.height, self.margin = CAMERAS[name]
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        length = hypot(dx, dy)
        # Unit normal with positive CrossingDetector distance, flipped to the in side.
        self.normal = (-dy / length * self.in_side, dx / length * self.in_side)

    def box(self, u, v, scale=1.0):
        h = self.height * scale
        x = self.start[0] + u * (self.end[0] - self.start[0]) + v * self.height * self.normal[0]
        y = self.start[1] + u * (self.end[1] - self.start[1]) + v * self.height * self.normal[1]
        return (x - 0.2 * h, y - h, x + 0.2 * h, y)

    def detector(self, **options):
        return CrossingDetector(self.start, self.end, self.in_side, self.margin, 1.0, "point", **options)


def simulate(camera, fps, people, grants, detector_options=None, duration=None):
    """people: list of (track_id, keyframes[, scale]); keyframes (t, u, v)."""
    detector = camera.detector(**(detector_options or {}))
    events = EventAccumulator([Grant(*g) for g in grants], grace_s=0.5)
    end = duration or max(p[1][-1][0] for p in people) + 0.5
    crossings = []
    for frame in range(int(end * fps) + 1):
        t = frame / fps
        observations = []
        for person in people:
            track_id, keys = person[0], person[1]
            scale = person[2] if len(person) > 2 else 1.0
            position = lerp(keys, t)
            if position is not None:
                observations.append(Observation(track_id, t, camera.box(*position, scale)))
        for crossing in detector.update_frame(observations):
            crossings.append(crossing)
            events.add(crossing)
    return events.records(), crossings, detector


def walk_in(t0, u=0.5, track=1, duration=2.0):
    return (track, [(t0, u, -1.0), (t0 + duration, u, 1.0)])


def walk_out(t0, u=0.5, track=1, duration=2.0):
    return (track, [(t0, u, 1.0), (t0 + duration, u, -1.0)])


class ScenarioTests(unittest.TestCase):
    @staticmethod
    def cases():
        return [(Camera(name), fps) for name in CAMERAS for fps in FPS]

    def counts(self, records):
        return [(r["grant_id"], r["entries"], r["alert"]) for r in records]

    def test_s1_single_entry(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                records, crossings, _ = simulate(camera, fps, [walk_in(0)], [("g1", 0, 3)])
                self.assertEqual(self.counts(records), [("g1", 1, "none")])
                # Crossing time is the interpolated zero-line time (here t = 1 s), not confirmation.
                self.assertAlmostEqual(crossings[0].t_cross_s, 1.0, delta=1 / fps)

    def test_s2_close_follower(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                people = [walk_in(0, track=1), walk_in(0.8, track=2)]
                records, _, _ = simulate(camera, fps, people, [("g1", 0, 3)])
                self.assertEqual(self.counts(records), [("g1", 2, "tailgating")])

    def test_s3_side_by_side(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                people = [walk_in(0, u=0.3, track=1), walk_in(0, u=0.7, track=2)]
                records, _, _ = simulate(camera, fps, people, [("g1", 0, 3)], STITCH)
                self.assertEqual(self.counts(records), [("g1", 2, "tailgating")])

    def test_s4_three_people(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                people = [walk_in(0, 0.3, 1), walk_in(0.3, 0.7, 2), walk_in(0.9, 0.5, 3)]
                records, _, _ = simulate(camera, fps, people, [("g1", 0, 4)], STITCH)
                self.assertEqual(self.counts(records), [("g1", 3, "tailgating")])

    def test_s5_exit_while_entering(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                people = [walk_in(0, 0.3, 1), walk_out(0, 0.7, 2)]
                records, crossings, _ = simulate(camera, fps, people, [("g1", 0, 3)], STITCH)
                self.assertEqual(self.counts(records), [("g1", 1, "none")])
                self.assertEqual(sorted(c.direction for c in crossings), ["in", "out"])

    def test_s6_walk_past_outside_and_beyond_segment(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                past = (1, [(0, -0.5, -0.6), (3, 1.5, -0.6)])
                # Crosses the infinite line, but beside the door segment (u = 1.6).
                beside = (2, [(0, 1.6, -1.0), (2, 1.6, 1.0)])
                records, crossings, _ = simulate(camera, fps, [past, beside], [("g1", 0, 3)], STITCH)
                self.assertEqual(self.counts(records), [("g1", 0, "none")])
                self.assertEqual(crossings, [])

    def test_s7_hesitate_then_enter(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                keys = [(0, 0.5, -1.0), (1, 0.5, -0.01), (2, 0.5, -0.8), (3, 0.5, 1.0)]
                records, _, _ = simulate(camera, fps, [(1, keys)], [("g1", 0, 4)])
                self.assertEqual(self.counts(records), [("g1", 1, "none")])

    def test_reentry_same_person_counts_once(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                keys = [(0, 0.5, -1), (1, 0.5, 1), (2, 0.5, -1), (3, 0.5, 1)]
                records, crossings, _ = simulate(camera, fps, [(1, keys)], [("g1", 0, 4)])
                self.assertEqual(self.counts(records), [("g1", 1, "none")])
                self.assertEqual([c.direction for c in crossings], ["in", "out", "in"])

    def test_s9_every_entry_without_grant_is_a_record(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                keys = [(0, 0.5, -1), (1, 0.5, 1), (2, 0.5, -1), (3, 0.5, 1)]
                records, _, _ = simulate(camera, fps, [(1, keys)], [("g1", 10, 12)], duration=13)
                self.assertEqual(self.counts(records), [("g1", 0, "none"), (None, 1, "unauthorised_entry"),
                                                        (None, 1, "unauthorised_entry")])

    def test_two_new_ids_cannot_take_the_same_lost_track(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Track 1 vanishes at the line; two new IDs appear in the same frame.
                # The nearer one (2) continues track 1, the other (3) stays separate.
                lost = (1, [(0, 0.5, -1), (0.9, 0.5, -0.1)])
                near = (2, [(1.2, 0.51, 0.2), (2.2, 0.51, 1)])
                far = (3, [(1.2, 0.6, 0.2), (2.2, 0.6, 1)])
                records, crossings, detector = simulate(camera, fps, [lost, near, far], [("g1", 0, 3)], STITCH)
                self.assertEqual([(s["new_track_id"], s["kept_id"]) for s in detector.stitches], [(2, 1)])
                self.assertEqual([c.track_id for c in crossings], [1])

    def test_fragments_do_not_bridge_a_long_gap(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Valid box outside, then 2.4 s of head-sized boxes only, then valid inside.
                person = (1, [(0, 0.5, -1), (0.5, 0.5, -0.6)])
                fragment = (1, [(0.5 + 1 / fps, 0.5, -0.5), (2.9, 0.5, -0.4)], 0.2)
                inside = (1, [(3.0, 0.5, 0.6), (3.5, 0.5, 1)])
                options = dict(min_box_height_px=0.5 * camera.height)
                records, crossings, _ = simulate(camera, fps, [person, fragment, inside], [("g1", 0, 4)], options)
                self.assertEqual(crossings, [])

    def test_short_occlusion_keeps_entry_and_long_gap_is_not_inferred(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Same tracker ID, unseen for 0.8 s across the line (tracker kept the ID).
                short = (1, [(0, 0.5, -1), (0.6, 0.5, -0.4)])
                short_after = (1, [(1.4, 0.5, 0.4), (2, 0.5, 1)])
                records, _, _ = simulate(camera, fps, [short, short_after], [("g1", 0, 3)])
                self.assertEqual(records[0]["entries"], 1)
                long_after = (1, [(2.0, 0.5, 0.4), (2.5, 0.5, 1)])
                records, _, _ = simulate(camera, fps, [short, long_after], [("g1", 0, 3)])
                self.assertEqual(records[0]["entries"], 0)

    def test_id_switch_at_line_is_repaired_only_when_enabled(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                before = (1, [(0, 0.5, -1), (0.9, 0.5, -0.1)])
                after = (2, [(1.2, 0.5, 0.2), (2, 0.5, 1)])
                records, _, _ = simulate(camera, fps, [before, after], [("g1", 0, 3)])
                self.assertEqual(records[0]["entries"], 0)
                records, _, detector = simulate(camera, fps, [before, after], [("g1", 0, 3)], STITCH)
                self.assertEqual(records[0]["entries"], 1)
                self.assertEqual(detector.stitches[0]["reason"], "new_id_after_gap")

    def test_overlapping_successor_takes_over(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Successor ID starts inside the dead band 0.2 s before the old ID is dropped
            # (as in clip ...6064 g04); alone it would never see the outside side.
                before = (1, [(0, 0.5, -1), (0.8, 0.5, -0.1), (1.0, 0.5, -0.02)])
                after = (2, [(0.8, 0.5, 0.0), (1.0, 0.5, 0.04), (2.2, 0.5, 1)])
                records, _, _ = simulate(camera, fps, [before, after], [("g1", 0, 3)])
                self.assertEqual(records[0]["entries"], 0)
                records, _, detector = simulate(camera, fps, [before, after], [("g1", 0, 3)], STITCH)
                self.assertEqual(records[0]["entries"], 1)
                self.assertEqual(detector.stitches[0]["reason"], "overlapping_successor")

    def test_tailgater_behind_lost_counted_person_is_still_counted(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Person 1 enters and is lost just inside; person 2 appears where 1 vanished.
                first = (1, [(0, 0.5, -1), (1.2, 0.5, 0.15)])
                second = (2, [(1.3, 0.5, -0.1), (2.5, 0.5, 1)])
                records, _, detector = simulate(camera, fps, [first, second], [("g1", 0, 3)], STITCH)
                self.assertEqual(self.counts(records), [("g1", 2, "tailgating")])
                self.assertEqual(detector.stitches, [])

    def test_side_by_side_new_id_is_not_absorbed(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Two different people, both visible: low overlap, no take-over.
                people = [walk_in(0, 0.35, 1), (2, [(0.5, 0.65, -0.75), (2.5, 0.65, 1.25)])]
                records, _, detector = simulate(camera, fps, people, [("g1", 0, 3)], STITCH)
                self.assertEqual(records[0]["entries"], 2)
                self.assertEqual(detector.stitches, [])

    def test_relative_dead_band_absorbs_jitter_of_person_standing_on_line(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                # Arrives from outside, then stands on the line with +-5% box-height jitter.
                keys = [(0, 0.5, -1)] + [(1 + 0.2 * k, 0.5, 0.05 if k % 2 else -0.05) for k in range(15)]
                _, crossings, _ = simulate(camera, fps, [(1, keys)], [("g1", 0, 5)])
                self.assertGreater(len(crossings), 0)
                _, crossings, _ = simulate(camera, fps, [(1, keys)], [("g1", 0, 5)], dict(margin_rel=0.08))
                self.assertEqual(crossings, [])

    def test_head_fragment_ignored_but_people_counted(self):
        for camera, fps in self.cases():
            with self.subTest(camera=camera.name, fps=fps):
                people = [walk_in(0, 0.5, 1), (2, [(0, 0.6, 0.3), (1, 0.6, -0.3), (2, 0.6, 0.3)], 0.25)]
                options = dict(min_box_height_px=0.5 * camera.height)
                records, crossings, _ = simulate(camera, fps, people, [("g1", 0, 3)], options)
                self.assertEqual(records[0]["entries"], 1)
                self.assertEqual({c.track_id for c in crossings}, {1})

    def test_scale_invariance(self):
        # Same scenario at 1x and 3x pixel scale must produce the same events.
        for fps in FPS:
            results = []
            for k in (1, 3):
                camera = Camera("horizontal_640x480")
                camera.start = tuple(k * c for c in camera.start)
                camera.end = tuple(k * c for c in camera.end)
                camera.height *= k
                camera.margin *= k
                before = (1, [(0, 0.5, -1), (0.9, 0.5, -0.1)])
                after = (2, [(1.2, 0.52, 0.2), (2, 0.52, 1)])
                records, crossings, _ = simulate(camera, fps, [before, after], [("g1", 0, 3)], STITCH)
                results.append((self.counts(records), [(c.track_id, round(c.t_cross_s, 6)) for c in crossings]))
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
