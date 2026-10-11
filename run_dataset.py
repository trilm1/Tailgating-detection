"""Run every local MP4 with its labels and aggregate metrics across clips."""

import argparse
from pathlib import Path

from src.tailgating.evaluation import scores
from src.tailgating.io import write_json
from src.tailgating.pipeline import run


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def clip_config(video, shared_config, config_dir):
    if shared_config is not None:
        return Path(shared_config)
    config_dir = Path(config_dir) if config_dir is not None else Path("configs")
    matches = [Path(config_dir) / (video.stem + suffix) for suffix in (".yaml", ".yml", ".json")]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one config for {video.stem} in {config_dir}; found {len(matches)}")
    return matches[0]


def aggregate(results):
    total = dict(tp=0, fp=0, fn=0, tn=0, exact_count=0, grants=0)
    by_scenario, unauthorised = {}, dict(tp=0, fp=0, fn=0)
    frames = seconds = 0
    for result in results:
        metric = result["metrics"]
        for key in total:
            total[key] += metric["confusion"][key]
        for scenario, counts in metric["by_scenario"].items():
            target = by_scenario.setdefault(scenario, dict.fromkeys(total, 0))
            for key in target:
                target[key] += counts[key]
        for key in unauthorised:
            unauthorised[key] += metric["unauthorised_entry"][key]
        frames += result["performance"]["frames"]
        seconds += result["performance"]["processing_seconds"]
    return dict(confusion=total, tailgating=scores(total["tp"], total["fp"], total["fn"]),
                count_accuracy=total["exact_count"] / total["grants"] if total["grants"] else None,
                by_scenario=by_scenario, unauthorised_entry=dict(**unauthorised, **scores(**unauthorised)),
                processing_fps=frames/seconds if seconds else 0, frames=frames,
                notes="Counts are pooled across clips (not mean F1).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("clips"))
    parser.add_argument("--labels", type=Path, default=Path("clips"))
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument("--config", help="Explicit shared camera config for all selected clips")
    configs.add_argument("--config-dir", type=Path, help="Per-clip <stem>.yaml/.yml/.json configs (default: configs); no fallback")
    parser.add_argument("--out", type=Path, default=Path("results/baseline"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clips", nargs="+", help="Only these exact video filenames or stems")
    args = parser.parse_args()
    results, rows = [], ["| Clip | Tailgating TP/FP/FN/TN | Precision | Recall | F1 | Exact count | FPS |",
                     "|---|---|---:|---:|---:|---:|---:|"]
    videos = sorted(args.videos.glob("*.mp4"))
    if args.clips:
        selected = {Path(name).stem for name in args.clips}
        missing = selected - {p.stem for p in videos}
        if missing:
            parser.error(f"Clips not found: {sorted(missing)}")
        videos = [p for p in videos if p.stem in selected]
    if not videos:
        parser.error("No MP4 clips found")
    camera_configs = {}
    for video in videos:
        for suffix in ("_grants.csv", "_gt.csv"):
            if not (args.labels / (video.stem + suffix)).is_file():
                parser.error(f"Missing {video.stem + suffix}")
        try:
            camera_configs[video] = clip_config(video, args.config, args.config_dir)
        except ValueError as error:
            parser.error(str(error))
        if not camera_configs[video].is_file():
            parser.error(f"Missing config: {camera_configs[video]}")
    for video in videos:
        result = run(video, args.labels / (video.stem + "_grants.csv"), camera_configs[video],
                     args.out / video.stem, args.labels / (video.stem + "_gt.csv"), overwrite=args.overwrite)
        result["out"] = (args.out / video.stem).as_posix()
        results.append(result)
        m = result["metrics"]
        s = m["tailgating"]
        c = m["confusion"]
        rows.append(f"| {video.stem} | {c['tp']}/{c['fp']}/{c['fn']}/{c['tn']} | {fmt(s['precision'])} | {fmt(s['recall'])} | "
                    f"{fmt(s['f1'])} | {c['exact_count']}/{c['grants']} | {result['performance']['processing_fps']:.2f} |")
    summary = aggregate(results)
    summary["clips"] = results
    c, s = summary["confusion"], summary["tailgating"]
    rows.append(f"| **Total (pooled)** | {c['tp']}/{c['fp']}/{c['fn']}/{c['tn']} | {fmt(s['precision'])} | {fmt(s['recall'])} | "
                f"{fmt(s['f1'])} | {c['exact_count']}/{c['grants']} | {summary['processing_fps']:.2f} |")
    write_json(args.out / "summary.json", summary)
    (args.out / "metrics_table.md").write_text("# Results\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    confusion_rows = ["# Confusion by scenario", "", "Grant-level counts. Scenarios with no grants are listed as untested.",
                      "", "| Scenario | Grants | TP | FP | FN | TN | Exact count |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in range(1, 11):
        if scenario == 9:
            continue
        c = summary["by_scenario"].get(str(scenario))
        confusion_rows.append(f"| {scenario} | {c['grants']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {c['exact_count']} |"
                              if c else f"| {scenario} | 0 (untested) | — | — | — | — | — |")
    for label, c in summary["by_scenario"].items():
        if label not in {str(i) for i in range(1, 11)}:
            confusion_rows.append(f"| {label} | {c['grants']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {c['exact_count']} |")
    u = summary["unauthorised_entry"]
    confusion_rows.extend(["", f"Unauthorised entries (time matching, tolerance 1 s): TP={u['tp']}, FP={u['fp']}, FN={u['fn']}."])
    (args.out / "confusion_by_scenario.md").write_text("\n".join(confusion_rows) + "\n", encoding="utf-8")
    print(f"Summary: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
