"""Select two threshold endpoints and an inside point on a video frame."""

import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=0)
    parser.add_argument("--out", type=Path, required=True, help="Output YAML or JSON config")
    parser.add_argument("--points", type=float, nargs=6, metavar=("X1", "Y1", "X2", "Y2", "INSIDE_X", "INSIDE_Y"),
                        help="Noninteractive calibration")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.suffix.lower() not in (".json", ".yaml", ".yml"):
        parser.error("Config output must have a .yaml, .yml or .json extension")
    if args.out.exists() and not args.overwrite:
        parser.error("Config exists; use another path or --overwrite")
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        parser.error("Cannot read the requested video frame")
    height, width = frame.shape[:2]
    points = []
    if args.points:
        points = [args.points[i:i+2] for i in range(0, 6, 2)]
    else:
        title = "Click endpoint 1, endpoint 2, INSIDE. Enter=save, R=reset, Esc=cancel"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
                points.append([x, y])

        cv2.setMouseCallback(title, on_mouse)
        while True:
            preview = frame.copy()
            for i, p in enumerate(points):
                cv2.circle(preview, tuple(p), 5, (0, 255, 255), -1)
                cv2.putText(preview, str(i + 1), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
            if len(points) >= 2:
                cv2.line(preview, tuple(points[0]), tuple(points[1]), (0, 255, 255), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 3:
                break
        cv2.destroyAllWindows()
    a, b, inside = points
    if any(not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in points):
        parser.error("Points must be inside the frame")
    cross = (b[0] - a[0])*(inside[1] - a[1]) - (b[1] - a[1])*(inside[0] - a[0])
    if cross == 0:
        parser.error("Inside point must not lie on the threshold line")
    import os
    model = Path(__file__).resolve().parent / "models" / "yolo11m.pt"
    try:
        model_path = os.path.relpath(model, args.out.resolve().parent).replace("\\", "/")
    except ValueError:
        # Windows cannot express a relative path between different drives.
        model_path = model.as_posix()
    config = dict(camera_id=Path(args.video).stem, reference_size=[width, height],
                  threshold=dict(start=a, end=b, in_side=1 if cross > 0 else -1,
                                 # Dead band scales with resolution: 6 px at 480 lines.
                                 margin_px=max(2, round(0.0125 * height)), mode="footprint"),
                  model=model_path,
                  device="cpu", grant_grace_s=.5, tracker_buffer_s=1, max_observation_gap_s=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".json":
        contents = json.dumps(config, indent=2)
    else:
        import yaml
        contents = yaml.safe_dump(config, sort_keys=False)
    args.out.write_text(contents, encoding="utf-8")
    cv2.line(frame, tuple(round(v) for v in a), tuple(round(v) for v in b), (0, 255, 255), 2)
    cv2.circle(frame, tuple(round(v) for v in inside), 6, (0, 255, 0), -1)
    cv2.imwrite(str(args.out.with_suffix(".preview.jpg")), frame)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
