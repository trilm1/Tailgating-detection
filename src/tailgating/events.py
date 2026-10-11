from dataclasses import asdict

from .domain import Crossing, Grant


class EventAccumulator:
    def __init__(self, grants: list[Grant], grace_s: float = 0.5, lead_s: float = 0.0,
                 assignment: str = "latest"):
        if grace_s < 0 or lead_s < 0:
            raise ValueError("grace_s and lead_s must be nonnegative")
        if assignment not in ("latest", "unfilled_first"):
            raise ValueError("assignment must be latest or unfilled_first")
        if len({g.grant_id for g in grants}) != len(grants):
            raise ValueError("Duplicate grant IDs")
        if any(g.t_unlock_s < 0 or g.t_relock_s < g.t_unlock_s for g in grants):
            raise ValueError("Invalid grant window")
        self.grants = sorted(grants, key=lambda g: (g.t_unlock_s, g.grant_id))
        self.grace_s = grace_s
        self.lead_s = lead_s
        self.assignment = assignment
        self.crossings = {g.grant_id: {} for g in grants}
        self.unauthorised = []

    def add(self, crossing: Crossing):
        if crossing.direction not in ("in", "out"):
            raise ValueError("direction must be in or out")
        if crossing.direction == "out":
            return
        # Prefer an actual active window over the grace period of an older grant.
        # lead_s absorbs timing error between an interpolated crossing and the unlock mark.
        active = [g for g in self.grants
                  if g.t_unlock_s - self.lead_s <= crossing.t_cross_s <= g.t_relock_s]
        eligible = active or [g for g in self.grants
                              if g.t_relock_s < crossing.t_cross_s <= g.t_relock_s + self.grace_s]
        if not eligible:
            # One record per entry: in-out-in without any grant is two unauthorised entries.
            self.unauthorised.append(crossing)
            return
        # Re-entry with the same tracker ID counts only once, also across overlapping grants.
        if any(crossing.track_id in self.crossings[g.grant_id] for g in eligible):
            return
        unfilled = [g for g in eligible if not self.crossings[g.grant_id]]
        if self.assignment == "unfilled_first" and unfilled:
            # Back-to-back credentials: the oldest grant still waiting for its person first.
            grant = min(unfilled, key=lambda g: (g.t_unlock_s, g.grant_id))
        else:
            grant = max(eligible, key=lambda g: (g.t_unlock_s, g.grant_id))
        self.crossings[grant.grant_id][crossing.track_id] = crossing

    def records(self) -> list[dict]:
        records = []
        for grant in self.grants:
            crossings = sorted(self.crossings[grant.grant_id].values(),
                               key=lambda c: c.t_cross_s)
            records.append({
                "grant_id": grant.grant_id, "t_unlock_s": grant.t_unlock_s,
                "entries": len(crossings),
                "alert": "tailgating" if len(crossings) > 1 else "none",
                "crossings": [asdict(c) for c in crossings],
            })
        records.extend({"grant_id": None, "entries": 1,
                        "alert": "unauthorised_entry", "crossings": [asdict(c)]}
                       for c in sorted(self.unauthorised, key=lambda c: c.t_cross_s))
        return records
