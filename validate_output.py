"""Check a submitted clip's event schema and decode its entire annotated video."""

import argparse
import json
from pathlib import Path

import cv2

from src.tailgating.evaluation import validate_events
from src.tailgating.io import read_grants
from src.tailgating.video import validate_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--grants", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        events = json.loads((args.out / "events.json").read_text(encoding="utf-8"))
        validate_events(events, read_grants(args.grants))
        cap = cv2.VideoCapture(str(args.video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open source video: {args.video}")
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            cap.release()
        frames = validate_video(args.video)
        validate_video(args.out / "annotated.mp4", frames, size)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Invalid output: {error}\n")
    print(f"OK: {len(events)} events; {frames} annotated frames; {args.out}")


if __name__ == "__main__":
    main()
