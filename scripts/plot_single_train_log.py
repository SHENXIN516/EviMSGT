import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = re.compile(
    r"\[(?P<split>[^\]]+)\]\s+epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+"
    r"train_loss=(?P<train_loss>[0-9]*\.?[0-9]+)\s+"
    r"val_acc=(?P<val_acc>[0-9]*\.?[0-9]+)\s+"
    r"val_auc=(?P<val_auc>[0-9]*\.?[0-9]+)"
)


def parse_log(log_path: str) -> pd.DataFrame:
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            rows.append(
                {
                    "split": m.group("split"),
                    "epoch": int(m.group("epoch")),
                    "total": int(m.group("total")),
                    "train_loss": float(m.group("train_loss")),
                    "val_acc": float(m.group("val_acc")),
                    "val_auc": float(m.group("val_auc")),
                }
            )
    if len(rows) == 0:
        raise RuntimeError("No epoch records parsed from log yet.")
    return pd.DataFrame(rows).sort_values("epoch")


def plot_metrics(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Combined plot
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(df["epoch"], df["train_loss"], color="#1f77b4", marker="o", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_acc"], color="#2ca02c", marker="s", label="val_acc")
    ax2.plot(df["epoch"], df["val_auc"], color="#d62728", marker="^", label="val_auc")
    ax2.set_ylabel("validation metrics", color="#333333")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Single-task Training Curves")
    plt.tight_layout()
    combined_path = os.path.join(out_dir, "split1_train_acc_auc_combined.png")
    plt.savefig(combined_path, dpi=180)
    plt.close(fig)

    # Separate metric plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(df["epoch"], df["val_acc"], marker="s", label="val_acc")
    plt.plot(df["epoch"], df["val_auc"], marker="^", label="val_auc")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Validation ACC/AUC")
    plt.legend()
    plt.tight_layout()
    metrics_path = os.path.join(out_dir, "split1_val_acc_auc.png")
    plt.savefig(metrics_path, dpi=180)
    plt.close()

    # CSV export
    csv_path = os.path.join(out_dir, "split1_metrics_from_log.csv")
    df.to_csv(csv_path, index=False)

    return combined_path, metrics_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Plot single-task training curves from log")
    parser.add_argument(
        "--log",
        default="/home/shenxin/EviMSGT/results/single_train/split1_formal.log",
        help="Path to training log",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/shenxin/EviMSGT/results/single_train/figures",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    df = parse_log(args.log)
    combined, metrics, csv_path = plot_metrics(df, args.out_dir)

    print(f"saved_combined={combined}")
    print(f"saved_metrics={metrics}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
