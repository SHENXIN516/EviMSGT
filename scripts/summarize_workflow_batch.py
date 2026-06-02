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


def summarize_task(path: Path, task_id: int, prefix: str) -> Optional[Dict[str, object]]:
    rows = read_rows(path)
    values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}

    for row in rows:
        for key in METRIC_KEYS:
            value = parse_float(row.get(f"{prefix}_{key}"))
            if value is not None:
                values[key].append(value)

    n_seeds = max((len(v) for v in values.values()), default=0)
    if n_seeds == 0:
        return None

    out: Dict[str, object] = {
        "attribute": task_id,
        "n_seeds": n_seeds,
    }
    for key in METRIC_KEYS:
        metric_values = values[key]
        if metric_values:
            out[key] = f"{mean(metric_values):.4f} ± {pstdev(metric_values):.4f}"
        else:
            out[key] = ""
    return out


def write_summary(rows: List[Dict[str, object]], output_dir: Path, split_name: str):
    split_dir = output_dir / f"{split_name}_metrics"
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / "summary.csv"
    header = ["attribute", "n_seeds", *METRIC_KEYS]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["attribute"])))
    print(f"Saved {split_name} summary: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize EviMSGT workflow batch metrics.")
    parser.add_argument("--results_dir", default="results/workflow_batch")
    parser.add_argument("--output_dir", default="results/workflow_batch_summary")
    parser.add_argument("--task_min", type=int, default=1)
    parser.add_argument("--task_max", type=int, default=19)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    task_files = {
        task_id: find_metrics_csv(results_dir, task_id)
        for task_id in range(args.task_min, args.task_max + 1)
    }
    found = {task_id: path for task_id, path in task_files.items() if path is not None}
    if not found:
        raise FileNotFoundError(f"No metrics CSV files found in {results_dir}")

    missing = [task_id for task_id, path in task_files.items() if path is None]
    if missing:
        print(f"Warning: missing task metrics: {','.join(str(i) for i in missing)}")

    for split_name, prefix in SPLIT_PREFIXES.items():
        rows = []
        for task_id, path in found.items():
            row = summarize_task(path, task_id, prefix)
            if row is not None:
                rows.append(row)
        write_summary(rows, output_dir, split_name)


if __name__ == "__main__":
    main()
