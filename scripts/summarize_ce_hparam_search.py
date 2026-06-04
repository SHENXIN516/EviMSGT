import argparse
import csv
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional


METRIC_KEYS = ["acc", "auc", "mcc", "se", "sp", "ba"]
SPLIT_PREFIXES = {
    "val": "val",
    "test": "test",
    "independent": "ind",
    "independent_test": "ind_test",
}


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def config_sort_key(path: Path):
    match = re.search(r"lr([^_]+)_drop([^_]+)_wd(.+)", path.name)
    if not match:
        return (path.name,)

    def parse_tag(text: str) -> float:
        return float(text.replace("p", "."))

    return (parse_tag(match.group(1)), parse_tag(match.group(2)), parse_tag(match.group(3)))


def find_metrics_csv(results_dir: Path, task_id: int) -> Optional[Path]:
    candidates = [
        results_dir / ("metrics.csv" if task_id == 1 else f"metrics({task_id}).csv"),
        results_dir / f"metrics_{task_id:02d}.csv",
        results_dir / f"task_{task_id:02d}.csv",
        results_dir / f"task_{task_id}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_values(values: List[float]) -> str:
    if not values:
        return ""
    return f"{mean(values):.4f} ± {pstdev(values):.4f}"


def collect_values(path: Path, prefix: str) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    for row in read_rows(path):
        for key in METRIC_KEYS:
            value = parse_float(row.get(f"{prefix}_{key}"))
            if value is not None:
                values[key].append(value)
    return values


def write_rows(path: Path, rows: List[Dict[str, object]], header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def summarize_split(search_root: Path, output_dir: Path, tasks: List[int], split_name: str, prefix: str):
    configs = sorted([p for p in search_root.iterdir() if p.is_dir()], key=config_sort_key)
    header = ["config", "attribute", "n_seeds", *METRIC_KEYS]
    per_task_rows = []
    overall_rows = []

    for config_dir in configs:
        overall_values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
        for task_id in tasks:
            path = find_metrics_csv(config_dir, task_id)
            if path is None:
                print(f"Warning: missing {config_dir.name} task {task_id}")
                continue
            values = collect_values(path, prefix)
            n_seeds = max((len(v) for v in values.values()), default=0)
            if n_seeds == 0:
                continue
            row: Dict[str, object] = {
                "config": config_dir.name,
                "attribute": task_id,
                "n_seeds": n_seeds,
            }
            for key in METRIC_KEYS:
                row[key] = format_values(values[key])
                overall_values[key].extend(values[key])
            per_task_rows.append(row)

        overall_row: Dict[str, object] = {
            "config": config_dir.name,
            "attribute": "overall",
            "n_seeds": max((len(v) for v in overall_values.values()), default=0),
        }
        for key in METRIC_KEYS:
            overall_row[key] = format_values(overall_values[key])
        overall_rows.append(overall_row)

    split_dir = output_dir / f"{split_name}_metrics"
    write_rows(split_dir / "summary_by_task.csv", per_task_rows, header)
    write_rows(split_dir / "summary_overall.csv", overall_rows, header)


def main():
    parser = argparse.ArgumentParser(description="Summarize CE hyperparameter search results.")
    parser.add_argument("--search_root", default="results/ce_hparam_search")
    parser.add_argument("--output_dir", default="results/ce_hparam_search_summary")
    parser.add_argument("--tasks", default="6,9,12")
    args = parser.parse_args()

    search_root = Path(args.search_root)
    if not search_root.exists():
        raise FileNotFoundError(f"Missing search root: {search_root}")

    tasks = parse_int_list(args.tasks)
    output_dir = Path(args.output_dir)
    for split_name, prefix in SPLIT_PREFIXES.items():
        summarize_split(search_root, output_dir, tasks, split_name, prefix)


if __name__ == "__main__":
    main()
