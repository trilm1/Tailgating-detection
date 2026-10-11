"""Directional crossing of a finite threshold segment with a dead band.

Tolerances that depend on apparent person size are relative to box height, so no
resolution, mounting height or line orientation is assumed.
"""

from math import hypot

from .domain import Crossing, Observation

# Keys for tracks that lost their ID to a stitched successor; above any tracker ID.
_FRESH_KEY_BASE = 1_000_000


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class CrossingDetector:
    def __init__(self, start, end, in_side=1, margin_px=8.0, max_gap_s=1.0, mode="point",
                 stitch_radius_rel=0.0, stitch_min_iou=0.5, min_box_height_px=0.0, margin_rel=0.0):
        self.start, self.end = tuple(start), tuple(end)
        self.dx, self.dy = end[0] - start[0], end[1] - start[1]
        self.length = hypot(self.dx, self.dy)
        if self.length == 0 or in_side not in (-1, 1):
            raise ValueError("Nonzero threshold and in_side = -1 or 1 required")
        if margin_px < 0 or max_gap_s <= 0:
            raise ValueError("Invalid margin or observation gap")
        if stitch_radius_rel < 0 or min_box_height_px < 0 or margin_rel < 0 or not 0 < stitch_min_iou <= 1:
            raise ValueError("stitch_radius_rel, min_box_height_px, margin_rel >= 0; stitch_min_iou in (0, 1]")
        if mode not in ("point", "footprint"):
            raise ValueError("Unknown crossing mode")
        self.in_side = in_side
        self.margin = margin_px
        self.margin_rel = margin_rel
        self.max_gap = max_gap_s
        self.mode = mode
        self.stitch_radius = stitch_radius_rel
        self.stitch_iou = stitch_min_iou
        self.min_height = min_box_height_px
        # Per state key; a key is a tracker ID unless stitching re-pointed it.
        self.states, self.pending = {}, {}
        self.last_seen, self.last_points, self.last_boxes = {}, {}, {}
        self.births, self.history, self.overlaps = {}, {}, {}
        self.emitted = set()
        # Tracker ID -> state key, and state key -> tracker ID that currently owns it.
        self.keys, self.owners = {}, {}
        self.stitches = []

    def update_frame(self, observations) -> list[Crossing]:
        """Process all observations of one frame; stitching needs the full visible set."""
        # Head or limb fragments (person right under the camera) have no foot point; they
        # are treated as not observed, so they neither keep a track alive nor get stitched.
        observations = [o for o in observations if o.bbox[3] - o.bbox[1] >= self.min_height]
        if not self.stitch_radius:
            return [c for c in (self.update(o) for o in observations) if c is not None]
        visible = {self._key(o.track_id) for o in observations if o.track_id in self.keys
                   or o.track_id in self.last_seen}
        self._stitch_new_ids([o for o in observations
                              if o.track_id not in self.keys and o.track_id not in self.last_seen], visible)
        crossings = [c for c in (self.update(o) for o in observations) if c is not None]
        current = {self._key(o.track_id): o for o in observations}
        self._record_overlaps(current)
        return crossings + self._handover(current)

    # --- ID-switch repair -------------------------------------------------------------
    # Only tracks that have not counted a crossing are joined, so a tailgater right
    # behind someone who already entered always keeps its own key. Joins are one-to-one.

    def _young(self, key, t_s):
        return key not in self.emitted and t_s - self.births[key] <= self.max_gap

    def _stitch_new_ids(self, new, visible):
        # New tracker IDs near where uncounted tracks vanished within max_gap. All pairs of
        # the frame are matched greedily by distance, so each lost track is taken at most once.
        pairs = []
        for obs in new:
            for key, seen in self.last_seen.items():
                if key in visible or key in self.emitted or not 0 < obs.t_s - seen <= self.max_gap:
                    continue
                old = self.last_boxes[key]
                if not self._similar_size(old, obs.bbox):
                    continue
                distance = hypot((old[0] + old[2] - obs.bbox[0] - obs.bbox[2]) / 2, old[3] - obs.bbox[3])
                if distance <= self.stitch_radius * (old[3] - old[1]):
                    pairs.append((distance, obs.track_id, key, obs.t_s))
        used_ids, used_keys = set(), set()
        for distance, track_id, key, t_s in sorted(pairs):
            if track_id not in used_ids and key not in used_keys:
                used_ids.add(track_id)
                used_keys.add(key)
                self._link(track_id, key, t_s, "new_id_after_gap", distance_px=round(distance, 1))

    def _record_overlaps(self, current):
        # IoU of each young track with each older visible track in their latest shared frame.
        for key, obs in current.items():
            if key not in self.births or not self._young(key, obs.t_s):
                continue
            for other, other_obs in current.items():
                if other != key and self.births.get(other, obs.t_s) < self.births[key]:
                    self.overlaps.setdefault(key, {})[other] = (box_iou(obs.bbox, other_obs.bbox), obs.t_s)

    def _handover(self, current):
        # ByteTrack may start a successor ID before dropping the old one. The successor
        # takes over if it overlapped the lost track by stitch_min_iou in its last frame.
        if not current:
            return []
        t_s = next(iter(current.values())).t_s
        crossings = []
        lost = [k for k, seen in self.last_seen.items()
                if k not in current and k in self.states and k not in self.emitted
                and 0 < t_s - seen <= self.max_gap]
        for old in sorted(lost, key=lambda k: -self.last_seen[k]):
            best = None
            for key, obs in current.items():
                if (key not in self.births or not self._young(key, t_s) or key in self.owners
                        or not self.births[old] < self.births[key]):
                    continue
                iou, shared_t = self.overlaps.get(key, {}).get(old, (0.0, None))
                if shared_t == self.last_seen[old] and iou >= self.stitch_iou and (best is None or iou > best[0]):
                    best = (iou, key, obs)
            if best is None:
                continue
            iou, key, obs = best
            replay = [o for o in self.history.get(key, []) if o.t_s > self.last_seen[old]]
            self._forget(key)
            self._link(obs.track_id, old, t_s, "overlapping_successor", iou=round(iou, 3))
            current.pop(key)
            crossings += [c for c in (self.update(o) for o in replay) if c is not None]
        return crossings

    @staticmethod
    def _similar_size(old, new):
        return 0.5 <= (new[3] - new[1]) / (old[3] - old[1]) <= 2

    def _link(self, track_id, key, t_s, reason, **evidence):
        self.keys[track_id] = key
        self.owners[key] = track_id
        self.stitches.append(dict(t_s=t_s, new_track_id=track_id, kept_id=key, reason=reason, **evidence))

    def _forget(self, key):
        for table in (self.last_seen, self.states, self.last_points, self.last_boxes,
                      self.pending, self.history, self.births, self.overlaps):
            table.pop(key, None)

    def _key(self, track_id):
        key = self.keys.get(track_id, track_id)
        if self.owners.get(key, track_id) != track_id:
            # The original ID came back after a successor took over: treat it as new.
            key = _FRESH_KEY_BASE + track_id
            self.keys[track_id] = key
        return key

    # --- Crossing ----------------------------------------------------------------------

    def update(self, obs: Observation) -> Crossing | None:
        x1, y1, x2, y2 = obs.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box")
        if y2 - y1 < self.min_height:
            return None
        key = self._key(obs.track_id)
        last = self.last_seen.get(key)
        if last is not None and obs.t_s < last:
            raise ValueError("Observations must be chronological per track")
        # Expire stale IDs without treating tracker gaps as confirmed crossings.
        for stale, seen in list(self.last_seen.items()):
            if obs.t_s - seen > self.max_gap:
                self._forget(stale)
        self.last_seen[key] = obs.t_s
        self.last_boxes[key] = obs.bbox
        self.births.setdefault(key, obs.t_s)
        if self.stitch_radius and self._young(key, obs.t_s):
            self.history.setdefault(key, []).append(obs)
        else:
            self.history.pop(key, None)
        # Bottom centre approximates the point where a person's feet touch ground.
        point = ((x1 + x2) / 2, y2)
        distance = self._distance(point)
        # Record the actual zero-line transition using adjacent nonzero samples.
        # Confirmation may happen much later if a person waits in the dead band.
        last_point = self.last_points.get(key)
        if distance != 0:
            if last_point is not None and last_point[2] * distance < 0:
                old_point, old_time, old_distance = last_point
                fraction = old_distance / (old_distance - distance)
                hit = tuple(a + fraction * (b - a) for a, b in zip(old_point, point))
                along = ((hit[0] - self.start[0]) * self.dx
                         + (hit[1] - self.start[1]) * self.dy) / self.length**2
                self.pending[key] = (
                    old_time + fraction * (obs.t_s - old_time), 0 <= along <= 1)
            self.last_points[key] = (point, obs.t_s, distance)
        # Box jitter grows with apparent size, so the dead band may scale with box height.
        margin = max(self.margin, self.margin_rel * (y2 - y1))
        if abs(distance) <= margin:
            return None
        if self.mode == "footprint":
            # Require the entire bottom edge to clear the line's dead band.
            # This is an image-space foot support proxy, not 3D body geometry.
            left, right = self._distance((x1, y2)), self._distance((x2, y2))
            if not (min(left, right) > margin or max(left, right) < -margin):
                return None
        side = 1 if distance > 0 else -1
        previous = self.states.get(key)
        self.states[key] = side
        if previous is None or previous == side:
            self.pending.pop(key, None)
            return None
        transition = self.pending.pop(key, None)
        if transition is None or not transition[1]:
            return None
        self.emitted.add(key)
        return Crossing(key, transition[0], "in" if side == self.in_side else "out")

    def _distance(self, point):
        return (self.dx * (point[1] - self.start[1])
                - self.dy * (point[0] - self.start[0])) / self.length
