import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fasta_workflow_batch import read_independent_manifest, read_manifest, resolve_manifest_paths
from scripts.train_5split_ensemble_grid import load_model_from_ckpt, predict_probs
from scripts.train_plat import BBBP_Dataset


METRIC_KEYS = ["acc", "auc", "mcc", "se", "sp", "ba"]


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def parse_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def align_dataset_rows(dataset: BBBP_Dataset, raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    aligned = []
    raw_pos = 0
    for smi in dataset.smiles:
        matched = {}
        while raw_pos < len(raw_rows):
            raw = raw_rows[raw_pos]
            raw_pos += 1
            candidate = raw.get("sequence", raw.get("smiles", ""))
            if str(candidate) == str(smi):
                matched = raw
                break
        aligned.append(matched)
    return aligned


def split_distribution_rows(dataset_root: Path, tasks: List[int], seeds: List[int]) -> List[Dict[str, object]]:
    out = []
    for task_id in tasks:
        split_manifest, independent_manifest = resolve_manifest_paths(dataset_root, task_id)
        for seed in seeds:
            rows = read_manifest(split_manifest, seed=seed, task=task_id)
            for split in ["train", "val", "test"]:
                part = [r for r in rows if str(r.get("split", "")).lower() == split]
                out.append(distribution_row(task_id, seed, split, part))
        independent_rows = read_independent_manifest(independent_manifest, task=task_id)
        out.append(distribution_row(task_id, "all", "independent", independent_rows))
    return out


def distribution_row(task_id: int, seed: object, split: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    labels = [int(float(r.get("label", 0))) for r in rows if str(r.get("label", "")) != ""]
    seq_lens = [len(str(r.get("sequence", "")).strip()) for r in rows]
    pos = int(sum(labels))
    neg = int(len(labels) - pos)
    return {
        "task": task_id,
        "seed": seed,
        "split": split,
        "n": len(rows),
        "pos": pos,
        "neg": neg,
        "pos_rate": f"{(pos / len(labels)):.4f}" if labels else "",
        "seq_len_mean": f"{mean(seq_lens):.4f}" if seq_lens else "",
        "seq_len_std": f"{pstdev(seq_lens):.4f}" if len(seq_lens) > 1 else "0.0000",
        "seq_len_min": min(seq_lens) if seq_lens else "",
        "seq_len_max": max(seq_lens) if seq_lens else "",
    }


def summarize_metrics(results_dir: Path, tasks: List[int]) -> List[Dict[str, object]]:
    out = []
    for task_id in tasks:
        path = find_metrics_csv(results_dir, task_id)
        if path is None:
            continue
        rows = read_csv(path)
        for prefix in ["val", "test", "ind"]:
            metric_values: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
            for row in rows:
                for key in METRIC_KEYS:
                    value = parse_float(row.get(f"{prefix}_{key}"))
                    if value is not None:
                        metric_values[key].append(value)
            out_row = {"task": task_id, "split": prefix, "n_seeds": max((len(v) for v in metric_values.values()), default=0)}
            for key in METRIC_KEYS:
                vals = metric_values[key]
                out_row[key] = f"{mean(vals):.4f} ± {pstdev(vals):.4f}" if vals else ""
            out.append(out_row)
    return out


def prediction_rows(csv_path: str, ckpt_path: str, mapping_mode: str, batch_size: int, device, split_filter: Optional[str] = None):
    dataset = BBBP_Dataset(csv_path, split_col="split" if split_filter else None, label_col="label", mapping_mode=mapping_mode)
    graphs = []
    meta = []
    raw_rows = read_csv(Path(csv_path))
    aligned_rows = align_dataset_rows(dataset, raw_rows)
    for i in range(len(dataset)):
        if split_filter and dataset.splits is not None and str(dataset.splits[i]).lower() != split_filter:
            continue
        graph = dataset.get(i)
        graphs.append(graph)
        raw = aligned_rows[i] if i < len(aligned_rows) else {}
        meta.append(raw)
    if not graphs:
        return []

    model, use_evidential = load_model_from_ckpt(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, y_true, _ = predict_probs(model, loader, device, use_evidential=use_evidential)
    threshold = float(ckpt.get("best_threshold", 0.5))
    out = []
    for raw, y, prob in zip(meta, y_true, probs):
        pred = int(float(prob) >= threshold)
        y_int = int(y)
        if pred == y_int:
            error_type = "correct"
        elif pred == 1:
            error_type = "fp"
        else:
            error_type = "fn"
        out.append(
            {
                "sample_id": raw.get("sample_id", raw.get("id", "")),
                "orig_id": raw.get("orig_id", ""),
                "split": raw.get("split", split_filter or ""),
                "label": y_int,
                "prob_pos": float(prob),
                "pred": pred,
                "error_type": error_type,
                "sequence_len": len(str(raw.get("sequence", "")).strip()),
                "sequence": raw.get("sequence", ""),
            }
        )
    return out


def export_errors(results_dir: Path, tasks: List[int], split: str, mapping_mode: str, batch_size: int, output_dir: Path, device):
    rows_out = []
    for task_id in tasks:
        metrics_path = find_metrics_csv(results_dir, task_id)
        if metrics_path is None:
            continue
        for metric_row in read_csv(metrics_path):
            seed = metric_row.get("seed", "")
            split_csv = metric_row.get("split_csv", "")
            ckpt_path = metric_row.get("ckpt_path", "")
            if not split_csv or not ckpt_path or not Path(split_csv).exists() or not Path(ckpt_path).exists():
                continue
            for row in prediction_rows(split_csv, ckpt_path, mapping_mode, batch_size, device, split_filter=split):
                row.update({"task": task_id, "seed": seed, "ckpt_path": ckpt_path})
                rows_out.append(row)

    write_csv(
        output_dir / f"errors_{split}.csv",
        rows_out,
        ["task", "seed", "sample_id", "orig_id", "split", "label", "prob_pos", "pred", "error_type", "sequence_len", "sequence", "ckpt_path"],
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze task performance, split distribution, and error samples.")
    parser.add_argument("--results_dir", default="results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5")
    parser.add_argument("--dataset_root", default="/home/shenxin/benchmark/dataset")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--seeds", default="10,20,30,40,50")
    parser.add_argument("--mapping_mode", default="helm_force")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--error_split", default="test", choices=["val", "test"])
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument("--output_dir", default="results/task_failure_analysis")
    args = parser.parse_args()

    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist_rows = split_distribution_rows(Path(args.dataset_root), tasks, seeds)
    write_csv(
        output_dir / "split_distribution.csv",
        dist_rows,
        ["task", "seed", "split", "n", "pos", "neg", "pos_rate", "seq_len_mean", "seq_len_std", "seq_len_min", "seq_len_max"],
    )
    metric_rows = summarize_metrics(Path(args.results_dir), tasks)
    write_csv(output_dir / "metric_summary.csv", metric_rows, ["task", "split", "n_seeds", *METRIC_KEYS])
    if not args.skip_errors:
        export_errors(Path(args.results_dir), tasks, args.error_split, args.mapping_mode, args.batch_size, output_dir, device)


if __name__ == "__main__":
    main()
