import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fasta_workflow_batch import read_independent_manifest, read_manifest, resolve_manifest_paths
from scripts.train_plat import BBBP_Dataset, mol_to_graph


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def base_output(row: Dict[str, object], task_id: int, seed: str, source: str) -> Dict[str, object]:
    sequence = str(row.get("sequence", "")).strip().upper()
    return {
        "task": task_id,
        "seed": seed,
        "source": source,
        "split": row.get("split", source),
        "sample_id": row.get("sample_id", row.get("id", "")),
        "orig_id": row.get("orig_id", ""),
        "label": row.get("label", ""),
        "sequence": sequence,
        "sequence_len": len(sequence),
        "status": "ok",
        "num_atoms": "",
        "residue_count": "",
        "residue_count_delta": "",
        "empty_residue_count": "",
        "atom_per_residue_min": "",
        "atom_per_residue_mean": "",
        "atom_per_residue_max": "",
        "atom_per_residue_std": "",
        "merged_or_missing": "",
    }


def summarize_graph(out: Dict[str, object], graph):
    if graph is None:
        out["status"] = "invalid_graph"
        return out

    atom_residue = graph.atom_residue_index.detach().cpu().numpy().astype(int)
    unique = np.unique(atom_residue)
    residue_count = int(unique.size)
    counts = np.bincount(atom_residue, minlength=max(int(atom_residue.max()) + 1, len(sequence))) if atom_residue.size else np.array([])
    empty_count = int((counts[: len(sequence)] == 0).sum()) if len(sequence) > 0 and counts.size > 0 else 0
    nonzero_counts = counts[counts > 0]

    out["num_atoms"] = int(graph.x.size(0))
    out["residue_count"] = residue_count
    out["residue_count_delta"] = int(residue_count - len(sequence))
    out["empty_residue_count"] = empty_count
    out["merged_or_missing"] = int(residue_count != len(sequence) or empty_count > 0)
    if nonzero_counts.size > 0:
        out["atom_per_residue_min"] = int(nonzero_counts.min())
        out["atom_per_residue_mean"] = float(nonzero_counts.mean())
        out["atom_per_residue_max"] = int(nonzero_counts.max())
        out["atom_per_residue_std"] = float(nonzero_counts.std())
    return out


def residue_audit_row(row: Dict[str, object], task_id: int, seed: str, source: str, mapping_mode: str):
    sequence = str(row.get("sequence", "")).strip().upper()
    out = base_output(row, task_id, seed, source)
    graph = mol_to_graph(
        sequence,
        sequence=sequence,
        helm=row.get("helm"),
        num_monomers=row.get("num_monomers"),
        mapping_mode=mapping_mode,
    )
    return summarize_graph(out, graph)


def cached_dataset_rows(csv_path: Path, task_id: int, seed: str, mapping_mode: str, cache_dir: str) -> List[Dict[str, object]]:
    dataset = BBBP_Dataset(
        str(csv_path),
        cache_dir=cache_dir,
        split_col="split",
        label_col="label",
        mapping_mode=mapping_mode,
    )
    raw_rows = read_csv(csv_path)
    aligned_rows = align_dataset_rows(dataset, raw_rows)
    rows = []
    for idx in range(len(dataset)):
        raw = aligned_rows[idx] if idx < len(aligned_rows) else {}
        if dataset.splits is not None and idx < len(dataset.splits):
            raw = dict(raw)
            raw["split"] = dataset.splits[idx]
        out = base_output(raw, task_id, seed, "split_csv_cache")
        rows.append(summarize_graph(out, dataset.get(idx)))
    return rows


def split_csv_by_seed(results_dir: Path, task_id: int) -> Dict[str, Path]:
    metrics_path = find_metrics_csv(results_dir, task_id)
    out = {}
    if metrics_path is None:
        return out
    for row in read_csv(metrics_path):
        seed = str(row.get("seed", "")).strip()
        split_csv = row.get("split_csv", "")
        if seed and split_csv and Path(split_csv).exists():
            out[seed] = Path(split_csv)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = {}
    for row in rows:
        key = (row["task"], row["source"], row["split"])
        groups.setdefault(key, []).append(row)

    out = []
    for (task, source, split), group in sorted(groups.items()):
        valid = [r for r in group if r["status"] == "ok"]
        seq_lens = [float(r["sequence_len"]) for r in valid]
        res_counts = [float(r["residue_count"]) for r in valid if r["residue_count"] != ""]
        atoms = [float(r["num_atoms"]) for r in valid if r["num_atoms"] != ""]
        merged = [int(r["merged_or_missing"]) for r in valid if r["merged_or_missing"] != ""]
        out.append(
            {
                "task": task,
                "source": source,
                "split": split,
                "n": len(group),
                "n_valid": len(valid),
                "invalid_graphs": len(group) - len(valid),
                "seq_len_mean": f"{mean(seq_lens):.4f}" if seq_lens else "",
                "seq_len_std": f"{pstdev(seq_lens):.4f}" if len(seq_lens) > 1 else "0.0000",
                "residue_count_mean": f"{mean(res_counts):.4f}" if res_counts else "",
                "atom_count_mean": f"{mean(atoms):.4f}" if atoms else "",
                "merged_or_missing_rate": f"{mean(merged):.4f}" if merged else "",
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Audit atom-to-residue mapping quality for benchmark manifests.")
    parser.add_argument("--dataset_root", default="/home/shenxin/benchmark/dataset")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--mapping_mode", default="helm_force")
    parser.add_argument("--pos_mode", default="zero")
    parser.add_argument("--results_dir", default="results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5")
    parser.add_argument("--cache_dir", default=str(ROOT / "dataset"))
    parser.add_argument("--no_cache_reuse", action="store_true")
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--output_dir", default="results/residue_mapping_audit")
    args = parser.parse_args()

    os.environ["EVIMSGT_POS_MODE"] = str(args.pos_mode)
    dataset_root = Path(args.dataset_root)
    tasks = parse_int_list(args.tasks)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    rows_out = []

    for task_id in tasks:
        print(f"[task {task_id}] residue mapping audit started", flush=True)
        split_manifest, independent_manifest = resolve_manifest_paths(dataset_root, task_id)
        cached_split_csvs = {} if args.no_cache_reuse else split_csv_by_seed(Path(args.results_dir), task_id)
        for seed in seeds:
            if seed in cached_split_csvs:
                print(f"[task {task_id} seed {seed}] loading cached dataset from {cached_split_csvs[seed]}", flush=True)
                rows_out.extend(cached_dataset_rows(cached_split_csvs[seed], task_id, seed, args.mapping_mode, args.cache_dir))
                continue

            manifest_rows = read_manifest(split_manifest, seed=int(seed), task=task_id)
            print(f"[task {task_id} seed {seed}] fallback manifest graph audit: {len(manifest_rows)} rows", flush=True)
            for i, row in enumerate(manifest_rows, 1):
                rows_out.append(residue_audit_row(row, task_id, seed, "split_manifest", args.mapping_mode))
                if args.progress_every > 0 and i % args.progress_every == 0:
                    print(f"[task {task_id} seed {seed}] processed {i}/{len(manifest_rows)} split rows", flush=True)

        independent_rows = read_independent_manifest(independent_manifest, task=task_id)
        print(f"[task {task_id}] independent fallback graph audit: {len(independent_rows)} rows", flush=True)
        for i, row in enumerate(independent_rows, 1):
            rows_out.append(residue_audit_row(row, task_id, "all", "independent", args.mapping_mode))
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[task {task_id}] processed {i}/{len(independent_rows)} independent rows", flush=True)
        print(f"[task {task_id}] residue mapping audit finished", flush=True)

    fieldnames = [
        "task",
        "seed",
        "source",
        "split",
        "sample_id",
        "orig_id",
        "label",
        "sequence",
        "sequence_len",
        "status",
        "num_atoms",
        "residue_count",
        "residue_count_delta",
        "empty_residue_count",
        "atom_per_residue_min",
        "atom_per_residue_mean",
        "atom_per_residue_max",
        "atom_per_residue_std",
        "merged_or_missing",
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "residue_mapping_audit_by_sample.csv", rows_out, fieldnames)
    write_csv(
        output_dir / "residue_mapping_audit_summary.csv",
        summarize(rows_out),
        [
            "task",
            "source",
            "split",
            "n",
            "n_valid",
            "invalid_graphs",
            "seq_len_mean",
            "seq_len_std",
            "residue_count_mean",
            "atom_count_mean",
            "merged_or_missing_rate",
        ],
    )


if __name__ == "__main__":
    main()
