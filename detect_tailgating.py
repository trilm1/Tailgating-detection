"""Detect tailgating on one video using local weights and a per-camera config."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--grants", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gt", help="Optional ground-truth CSV; enables metrics.json")
    parser.add_argument("--scenarios", help="Optional scenario JSON; inferred beside GT if omitted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        from src.tailgating.pipeline import run
        result = run(args.video, args.grants, args.config, args.out, args.gt, args.scenarios, args.overwrite)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Done: {result['out']} | CPU/device={result['performance']['device']} | {result['performance']['processing_fps']:.2f} FPS")
    if result["metrics"]:
        for warning in result["metrics"]["label_warnings"]:
            print(f"GT warning: {warning}")


if __name__ == "__main__":
    main()
