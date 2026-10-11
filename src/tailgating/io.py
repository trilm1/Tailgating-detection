import csv
import json
from math import isfinite
from pathlib import Path

from .domain import Grant, GroundTruth


def _rows(path, required):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {sorted(required)}")
        return list(reader)


def _time(value):
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError("Timestamps must be finite and nonnegative")
    return result


def read_grants(path):
    rows = _rows(path, {"grant_id", "t_unlock_s", "t_relock_s"})
    grants = [Grant(r["grant_id"].strip(), _time(r["t_unlock_s"]),
                    _time(r["t_relock_s"])) for r in rows]
    if any(not g.grant_id or g.t_relock_s < g.t_unlock_s for g in grants):
        raise ValueError("Invalid grant ID or window")
    if len({g.grant_id for g in grants}) != len(grants):
        raise ValueError("Duplicate grant IDs")
    return grants


def read_ground_truth(path, grants):
    rows = _rows(path, {"person_id", "t_cross_s", "direction", "grant_id"})
    known = {g.grant_id for g in grants}
    result = []
    for row in rows:
        person = row["person_id"].strip()
        direction = row["direction"].strip()
        grant = row["grant_id"].strip() or None
        if not person or direction not in ("in", "out") or (grant and grant not in known):
            raise ValueError(f"Invalid ground truth row: {row}")
        result.append(GroundTruth(person, _time(row["t_cross_s"]), direction, grant))
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
