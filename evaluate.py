"""Evaluate already exported events without loading a detection model."""

import argparse
import json
from pathlib import Path

from src.tailgating.evaluation import evaluate, label_warnings
from src.tailgating.io import read_grants, read_ground_truth, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--grants", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--scenarios", help="JSON mapping grant_id to scenario number")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    grants = read_grants(args.grants)
    gt = read_ground_truth(args.gt, grants)
    with open(args.events, encoding="utf-8") as stream:
        events = json.load(stream)
    scenarios = {}
    if not args.scenarios:
        candidate = Path(args.gt).with_name(Path(args.gt).name.removesuffix("_gt.csv") + "_scenarios.json")
        if candidate.exists():
            args.scenarios = str(candidate)
    if args.scenarios:
        with open(args.scenarios, encoding="utf-8") as stream:
            scenarios = json.load(stream)
    result = evaluate(events, grants, gt, scenarios)
    result["label_warnings"] = label_warnings(grants, gt, scenarios)
    performance = Path(args.events).with_name("performance.json")
    if performance.exists():
        result["performance"] = json.loads(performance.read_text(encoding="utf-8"))
    write_json(args.out, result)


if __name__ == "__main__":
    main()
