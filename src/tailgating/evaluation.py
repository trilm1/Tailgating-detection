"""Grant-level metrics. Ground-truth grant IDs are authoritative labels."""

from math import isfinite


def validate_events(events, grants):
    """Reject malformed exports instead of scoring only their entry totals."""
    if not isinstance(events, list):
        raise ValueError("events must be a JSON array")
    known = {g.grant_id: g for g in grants}
    seen = set()
    for event in events:
        if not isinstance(event, dict) or not {"grant_id", "entries", "alert", "crossings"} <= event.keys():
            raise ValueError("Event is missing required fields")
        grant_id = event["grant_id"]
        if grant_id is not None and not isinstance(grant_id, str):
            raise ValueError("grant_id must be a string or null")
        count, crossings = event["entries"], event["crossings"]
        if type(count) is not int or count < 0 or not isinstance(crossings, list) or count != len(crossings):
            raise ValueError("entries must equal the number of crossings")
        if grant_id is None:
            if count != 1 or event["alert"] != "unauthorised_entry":
                raise ValueError("Each unauthorised event must contain exactly one entry")
        else:
            if grant_id not in known or grant_id in seen:
                raise ValueError(f"Unknown or duplicate predicted grant: {grant_id}")
            seen.add(grant_id)
            unlock = event.get("t_unlock_s")
            if type(unlock) not in (int, float) or unlock != known[grant_id].t_unlock_s:
                raise ValueError(f"Incorrect or missing t_unlock_s: {grant_id}")
            if event["alert"] != ("tailgating" if count > 1 else "none"):
                raise ValueError("Alert disagrees with entry count")
        ids = set()
        for crossing in crossings:
            if not isinstance(crossing, dict) or not {"track_id", "t_cross_s", "direction"} <= crossing.keys():
                raise ValueError("Crossing is missing required fields")
            track_id, t_s = crossing["track_id"], crossing["t_cross_s"]
            if type(track_id) is not int or crossing["direction"] != "in":
                raise ValueError("Event crossings require integer track_id and direction in")
            if type(t_s) not in (int, float) or not isfinite(t_s) or t_s < 0:
                raise ValueError("Crossing timestamps must be finite and nonnegative")
            if track_id in ids:
                raise ValueError("Duplicate track within a grant")
            ids.add(track_id)
        if "confidence" in event:
            confidence = event["confidence"]
            if type(confidence) not in (int, float) or not isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0, 1]")
    if seen != known.keys():
        raise ValueError("Predictions must contain one record for every grant")


def evaluate(events, grants, ground_truth, scenarios=None, unauthorised_tolerance_s=1.0):
    if unauthorised_tolerance_s < 0:
        raise ValueError("Matching tolerance must be nonnegative")
    scenarios = scenarios or {}
    validate_events(events, grants)
    expected = {g.grant_id: set() for g in grants}
    for row in ground_truth:
        if row.grant_id is not None and row.grant_id not in expected:
            raise ValueError(f"Unknown GT grant: {row.grant_id}")
        if row.direction == "in" and row.grant_id is not None:
            expected[row.grant_id].add(row.person_id)
    predictions = {}
    for event in events:
        grant_id = event["grant_id"]
        if grant_id is None:
            continue
        if grant_id not in expected or grant_id in predictions:
            raise ValueError(f"Unknown or duplicate predicted grant: {grant_id}")
        count = event["entries"]
        if type(count) is not int or count < 0:
            raise ValueError("entries must be a nonnegative integer")
        if event["alert"] != ("tailgating" if count > 1 else "none"):
            raise ValueError("Alert disagrees with entry count")
        predictions[grant_id] = count
    if predictions.keys() != expected.keys():
        raise ValueError("Predictions must contain one record for every grant")
    total = dict(tp=0, fp=0, fn=0, tn=0, exact_count=0, grants=0)
    by_scenario = {}
    for grant_id, people in expected.items():
        actual, predicted = len(people), predictions[grant_id]
        label = ("tp" if actual > 1 else "fp") if predicted > 1 else (
            "fn" if actual > 1 else "tn")
        bucket = by_scenario.setdefault(str(scenarios.get(grant_id, "unassigned")),
                                        dict.fromkeys(total, 0))
        for counts in (total, bucket):
            counts[label] += 1
            counts["grants"] += 1
            counts["exact_count"] += int(actual == predicted)
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    unauthorised = match_times(
        [c["t_cross_s"] for e in events if e["grant_id"] is None for c in e["crossings"]
         if c["direction"] == "in"],
        [c.t_cross_s for c in ground_truth if c.grant_id is None and c.direction == "in"],
        unauthorised_tolerance_s)
    return {
        "tailgating": scores(tp, fp, fn),
        "count_accuracy": total["exact_count"] / total["grants"] if total["grants"] else None,
        "confusion": total,
        "by_scenario": by_scenario,
        "unauthorised_entry": unauthorised,
        "notes": "Tailgating is scored per grant; unauthorised entries by one-to-one time matching. Undefined ratios are null.",
    }


def scores(tp, fp, fn):
    # None marks an undefined ratio, e.g. recall on a clip without positive grants.
    return dict(precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def match_times(predicted, expected, tolerance_s):
    """Maximum-cardinality one-to-one matching on sorted timestamps."""
    predicted, expected = sorted(predicted), sorted(expected)
    i = j = tp = 0
    while i < len(predicted) and j < len(expected):
        if predicted[i] < expected[j] - tolerance_s:
            i += 1
        elif expected[j] < predicted[i] - tolerance_s:
            j += 1
        else:
            tp += 1
            i += 1
            j += 1
    fp, fn = len(predicted) - tp, len(expected) - tp
    return dict(tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn), tolerance_s=tolerance_s)


def label_warnings(grants, ground_truth, scenarios):
    warnings = []
    lookup = {g.grant_id: g for g in grants}
    for row in ground_truth:
        if row.grant_id is not None:
            grant = lookup[row.grant_id]
            if not grant.t_unlock_s <= row.t_cross_s <= grant.t_relock_s:
                warnings.append(f"{row.person_id} at {row.t_cross_s}s is outside {row.grant_id}'s original window")
    for grant in grants:
        count = len({c.person_id for c in ground_truth if c.direction == "in" and c.grant_id == grant.grant_id})
        scenario = scenarios.get(grant.grant_id)
        valid = (count == 1 if scenario in (1, 5, 7, 8) else count == 2 if scenario in (2, 3)
                 else count >= 3 if scenario == 4 else count == 0 if scenario == 6 else True)
        if not valid:
            warnings.append(f"{grant.grant_id}: scenario {scenario} conflicts with GT entry count {count}")
    return warnings
