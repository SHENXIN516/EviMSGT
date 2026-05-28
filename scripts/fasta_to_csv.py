import argparse
import csv
import random
from pathlib import Path


AA_SET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def parse_header(header: str):
    header = header.strip()
    if not header.startswith(">"):
        raise ValueError(f"Invalid FASTA header: {header}")
    body = header[1:]
    if "|" in body:
        sample_id, label = body.rsplit("|", 1)
        return sample_id.strip(), int(label.strip())
    return body.strip(), None


def clean_sequence(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    if not seq:
        return seq
    invalid = sorted(set(ch for ch in seq if ch not in AA_SET))
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols found: {''.join(invalid)}")
    return seq


def parse_fasta(path: Path, default_label=None):
    rows = []
    current_id = None
    current_label = None
    seq_parts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    rows.append(
                        {
                            "id": current_id,
                            "sequence": clean_sequence("".join(seq_parts)),
                            "label": default_label if current_label is None else current_label,
                        }
                    )
                current_id, current_label = parse_header(line)
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_id is not None:
        rows.append(
            {
                "id": current_id,
                "sequence": clean_sequence("".join(seq_parts)),
                "label": default_label if current_label is None else current_label,
            }
        )

    return rows


def _assign_split_for_group(group_rows, rng, train_ratio=0.8, val_ratio=0.1):
    idxs = list(range(len(group_rows)))
    rng.shuffle(idxs)

    n = len(group_rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for rank, idx in enumerate(idxs):
        if rank < n_train:
            group_rows[idx]["split"] = "train"
        elif rank < n_train + n_val:
            group_rows[idx]["split"] = "val"
        else:
            group_rows[idx]["split"] = "test"


def assign_random_split(rows, seed=42, train_ratio=0.8, val_ratio=0.1):
    # Stratified split by label to keep class balance.
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        label = int(row["label"]) if row.get("label", None) is not None else -1
        buckets.setdefault(label, []).append(row)

    for group_rows in buckets.values():
        _assign_split_for_group(group_rows, rng, train_ratio=train_ratio, val_ratio=val_ratio)

    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert FASTA files to unified CSV for EviMSGT")
    parser.add_argument("--positive", default="/home/shenxin/EviMSGT/dataset/Positive.fasta")
    parser.add_argument("--negative", default="/home/shenxin/EviMSGT/dataset/Negative.fasta")
    parser.add_argument("--independent", default="/home/shenxin/EviMSGT/dataset/IndependentTest.fasta")
    parser.add_argument("--out_train", default="/home/shenxin/EviMSGT/dataset/fasta_trainval.csv")
    parser.add_argument("--out_independent", default="/home/shenxin/EviMSGT/dataset/fasta_independent_test.csv")
    parser.add_argument("--add_split", action="store_true", help="Add random train/val/test split to train CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    args = parser.parse_args()

    pos_rows = parse_fasta(Path(args.positive), default_label=1)
    neg_rows = parse_fasta(Path(args.negative), default_label=0)
    train_rows = pos_rows + neg_rows

    if args.add_split:
        train_rows = assign_random_split(train_rows, seed=args.seed)

    independent_rows = parse_fasta(Path(args.independent), default_label=None)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)

    train_fields = ["id", "sequence", "label"] + (["split"] if args.add_split else [])
    with open(args.out_train, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(args.out_independent, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "label"])
        writer.writeheader()
        writer.writerows(independent_rows)

    print(f"Wrote {len(train_rows)} rows to {args.out_train}")
    print(f"Wrote {len(independent_rows)} rows to {args.out_independent}")


if __name__ == "__main__":
    main()
