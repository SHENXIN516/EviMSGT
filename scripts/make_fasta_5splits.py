import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def stratified_kfold_indices(labels, n_splits=5, seed=42):
    by_label = defaultdict(list)
    for i, y in enumerate(labels):
        by_label[int(y)].append(i)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for idxs in by_label.values():
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % n_splits].append(idx)
    return folds


def split_train_val(indices, val_ratio=0.1, seed=42):
    idxs = indices[:]
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_ratio)) if len(idxs) > 1 else 0
    val = set(idxs[:n_val])
    train = set(idxs[n_val:])
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Create split1..split5 columns for FASTA-derived CSV")
    parser.add_argument("--in_csv", default="/home/shenxin/EviMSGT/dataset/fasta_trainval.csv")
    parser.add_argument("--out_csv", default="/home/shenxin/EviMSGT/dataset/fasta_trainval_5splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "dataset"

    in_path = Path(args.in_csv)
    if not in_path.is_absolute():
        # Prefer provided relative path; fallback to dataset/ for convenience.
        candidate = (Path.cwd() / in_path).resolve()
        if candidate.exists():
            in_path = candidate
        else:
            in_path = (dataset_dir / in_path).resolve()

    out_path = Path(args.out_csv)
    if not out_path.is_absolute():
        out_path = (dataset_dir / out_path).resolve()

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    labels = [int(float(r["label"])) for r in rows]
    folds = stratified_kfold_indices(labels, n_splits=5, seed=args.seed)

    all_idx = set(range(len(rows)))
    for fold_id in range(5):
        test_idx = set(folds[fold_id])
        trainval_idx = sorted(list(all_idx - test_idx))
        train_idx, val_idx = split_train_val(trainval_idx, val_ratio=args.val_ratio, seed=args.seed + fold_id)

        col = f"split{fold_id + 1}"
        for i, r in enumerate(rows):
            if i in test_idx:
                r[col] = "test"
            elif i in val_idx:
                r[col] = "val"
            else:
                r[col] = "train"

    fieldnames = list(rows[0].keys())
    # Keep legacy split if present; add split1..split5 after it.
    for c in [f"split{i}" for i in range(1, 6)]:
        if c not in fieldnames:
            fieldnames.append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input: {in_path}")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
