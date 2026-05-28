import argparse
import csv
from pathlib import Path

from fasta_to_csv import assign_random_split, parse_fasta
from make_fasta_5splits import split_train_val, stratified_kfold_indices


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_5split_columns(rows, seed=42, val_ratio=0.1):
    labels = [int(float(r["label"])) for r in rows]
    folds = stratified_kfold_indices(labels, n_splits=5, seed=seed)
    all_idx = set(range(len(rows)))

    for fold_id in range(5):
        test_idx = set(folds[fold_id])
        trainval_idx = sorted(list(all_idx - test_idx))
        train_idx, val_idx = split_train_val(trainval_idx, val_ratio=val_ratio, seed=seed + fold_id)

        col = f"split{fold_id + 1}"
        for i, row in enumerate(rows):
            if i in test_idx:
                row[col] = "test"
            elif i in val_idx:
                row[col] = "val"
            else:
                row[col] = "train"


def summarize(rows):
    total = len(rows)
    label_counts = {0: 0, 1: 0}
    for r in rows:
        y = int(float(r["label"]))
        label_counts[y] = label_counts.get(y, 0) + 1

    print(f"Total rows: {total}")
    print(f"Label distribution: 0={label_counts.get(0, 0)}, 1={label_counts.get(1, 0)}")

    for i in range(1, 6):
        col = f"split{i}"
        counts = {"train": 0, "val": 0, "test": 0}
        for r in rows:
            s = r.get(col, "")
            if s in counts:
                counts[s] += 1
        print(
            f"{col}: train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Full FASTA workflow: convert -> 5-split -> drop legacy split -> summary"
    )
    parser.add_argument("--positive", required=True, help="Path to positive FASTA")
    parser.add_argument("--negative", required=True, help="Path to negative FASTA")
    parser.add_argument("--independent", required=True, help="Path to independent FASTA")
    parser.add_argument("--out_train", required=True, help="Output train CSV")
    parser.add_argument("--out_independent", required=True, help="Output independent CSV")
    parser.add_argument("--out_5split", required=True, help="Output 5-split train CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--keep_legacy_split", action="store_true")
    args = parser.parse_args()

    pos_rows = parse_fasta(Path(args.positive), default_label=1)
    neg_rows = parse_fasta(Path(args.negative), default_label=0)
    train_rows = pos_rows + neg_rows
    train_rows = assign_random_split(train_rows, seed=args.seed)

    independent_rows = parse_fasta(Path(args.independent), default_label=None)

    write_csv(
        Path(args.out_train),
        train_rows,
        ["id", "sequence", "label", "split"],
    )
    write_csv(
        Path(args.out_independent),
        independent_rows,
        ["id", "sequence", "label"],
    )

    rows_5split = [dict(r) for r in train_rows]
    add_5split_columns(rows_5split, seed=args.seed, val_ratio=args.val_ratio)

    if not args.keep_legacy_split:
        for r in rows_5split:
            r.pop("split", None)

    fields = ["id", "sequence", "label"] + [f"split{i}" for i in range(1, 6)]
    if args.keep_legacy_split:
        fields = ["id", "sequence", "label", "split"] + [f"split{i}" for i in range(1, 6)]

    write_csv(Path(args.out_5split), rows_5split, fields)

    print(f"Wrote train CSV: {args.out_train}")
    print(f"Wrote independent CSV: {args.out_independent}")
    print(f"Wrote 5-split CSV: {args.out_5split}")
    summarize(rows_5split)


if __name__ == "__main__":
    main()
