import argparse
import csv
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


def summarize_split(
    config_name: str,
    config_dir: Path,
    output_dir: Path,
    tasks: List[int],
    split_name: str,
    prefix: str,
):
    header = ["config", "attribute", "n_seeds", *METRIC_KEYS]
    per_task_rows = []
    overall_values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    missing_tasks = []

    for task_id in tasks:
        path = find_metrics_csv(config_dir, task_id)
        if path is None:
            missing_tasks.append(task_id)
            continue

        values = collect_values(path, prefix)
        n_seeds = max((len(v) for v in values.values()), default=0)
        if n_seeds == 0:
            continue

        row: Dict[str, object] = {
            "config": config_name,
            "attribute": task_id,
            "n_seeds": n_seeds,
        }
        for key in METRIC_KEYS:
            row[key] = format_values(values[key])
            overall_values[key].extend(values[key])
        per_task_rows.append(row)

    overall_row: Dict[str, object] = {
        "config": config_name,
        "attribute": "overall",
        "n_seeds": max((len(v) for v in overall_values.values()), default=0),
    }
    for key in METRIC_KEYS:
        overall_row[key] = format_values(overall_values[key])

    split_dir = output_dir / f"{split_name}_metrics"
    write_rows(split_dir / "summary_by_task.csv", per_task_rows, header)
    write_rows(split_dir / "summary_overall.csv", [overall_row], header)

    if missing_tasks:
        print(f"Warning: missing {split_name} metrics for tasks: {','.join(map(str, missing_tasks))}")


def main():
    parser = argparse.ArgumentParser(description="Summarize fixed main-config EviMSGT results for tasks 1-19.")
    parser.add_argument("--config_name", default="lr3e-4_drop0p05_wd1e-5")
    parser.add_argument("--config_dir", default="results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5")
    parser.add_argument("--output_dir", default="results/main_config_1_19_summary")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Missing config results dir: {config_dir}")

    tasks = parse_int_list(args.tasks)
    output_dir = Path(args.output_dir)
    for split_name, prefix in SPLIT_PREFIXES.items():
        summarize_split(
            config_name=args.config_name,
            config_dir=config_dir,
            output_dir=output_dir,
            tasks=tasks,
            split_name=split_name,
            prefix=prefix,
        )


if __name__ == "__main__":
    main()

