import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader

try:
    from scripts.fasta_to_csv import parse_fasta, assign_random_split
    from scripts.train_plat import BBBP_Dataset, evidential_loss_from_logits
    from scripts.train_common import make_plateau_scheduler, run_training_loop
    from scripts.train_5split_ensemble_grid import build_model, compute_metrics_from_probs, predict_probs
except ModuleNotFoundError:
    from fasta_to_csv import parse_fasta, assign_random_split
    from train_plat import BBBP_Dataset, evidential_loss_from_logits
    from train_common import make_plateau_scheduler, run_training_loop
    from train_5split_ensemble_grid import build_model, compute_metrics_from_probs, predict_probs


@dataclass
class SplitResult:
    seed: int
    split_csv: str
    ckpt_path: str
    independent_ckpt_path: str
    best_epoch: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    independent_metrics: Dict[str, float]


def resolve_fasta(dataset_dir: Path, base: str, idx: int) -> Path:
    candidates = [
        dataset_dir / f"{base} ({idx}).fasta",
        dataset_dir / f"{base}({idx}).fasta",
        dataset_dir / f"{base}{idx}.fasta",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing FASTA for {base}({idx}) in {dataset_dir}")


def resolve_independent(dataset_dir: Path, idx: int) -> Path:
    bases = ["IndependentTest", "Independent", "Indenpent"]
    for base in bases:
        try:
            return resolve_fasta(dataset_dir, base, idx)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Missing Independent FASTA for ({idx}) in {dataset_dir}")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_loaders(dataset: BBBP_Dataset, batch_size: int):
    train_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "train"]
    val_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "val"]
    test_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "test"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Split column {dataset.split_col} must contain train/val/test")

    train_ds = [dataset.get(i) for i in train_idx]
    val_ds = [dataset.get(i) for i in val_idx]
    test_ds = [dataset.get(i) for i in test_idx]

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def compute_class_weights(dataset: BBBP_Dataset) -> Optional[torch.Tensor]:
    if dataset.splits is None:
        return None
    train_labels = [
        int(dataset.labels[i])
        for i, s in enumerate(dataset.splits)
        if str(s).lower() == "train"
    ]
    if len(train_labels) == 0:
        return None
    counts = np.bincount(np.array(train_labels, dtype=int), minlength=2)
    total = int(counts.sum())
    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (2.0 * float(c)))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float)


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    if y_true.size == 0:
        return 0.5, float("nan")
    thresholds = np.unique(probs)
    if thresholds.size == 0:
        return 0.5, float("nan")
    auc = compute_metrics_from_probs(y_true, probs).get("auc", float("nan"))
    best_thr = 0.5
    best_score = -float("inf")
    for thr in thresholds:
        acc = float(((probs >= thr).astype(int) == y_true).mean())
        score = 0.5 * float(auc) + 0.5 * acc
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def build_independent_loader(csv_path: str, mapping_mode: str, batch_size: int):
    dataset = BBBP_Dataset(
        csv_path,
        split_col=None,
        label_col="label",
        permeability_col="permeability",
        permeability_threshold=None,
        mapping_mode=mapping_mode,
    )
    graphs = [dataset.get(i) for i in range(len(dataset))]
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    return dataset, loader


def train_one_seed(
    csv_path: str,
    seed: int,
    mapping_mode: str,
    model_name: str,
    use_evidential: bool,
    batch_size: int,
    epochs: int,
    lr: float,
    kl_weight: float,
    anneal_epochs: int,
    device,
    ckpt_dir: str,
):
    dataset = BBBP_Dataset(
        csv_path,
        split_col="split",
        label_col="label",
        permeability_col="permeability",
        permeability_threshold=None,
        mapping_mode=mapping_mode,
    )
    train_loader, val_loader, test_loader = make_loaders(dataset, batch_size)

    class_weights = compute_class_weights(dataset)
    if class_weights is not None:
        print(f"[seed {seed}] class weights: {class_weights.tolist()}", flush=True)

    model = build_model(model_name, use_evidential=use_evidential).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = make_plateau_scheduler(optimizer, mode="max", patience=6, factor=0.8)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    fixed_threshold = 0.5

    best_state = None
    best_epoch = -1
    best_val_metrics = None
    best_test_metrics = None
    best_test_state = None
    best_test_epoch = -1
    best_test_metrics_best_acc = None
    best_threshold = fixed_threshold
    best_test_threshold = fixed_threshold

    def train_step(epoch):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            if use_evidential:
                loss = evidential_loss_from_logits(
                    logits,
                    batch.y,
                    epoch_idx=epoch,
                    anneal_epochs=anneal_epochs,
                    kl_weight=kl_weight,
                    num_classes=2,
                    class_weights=class_weights.to(device) if class_weights is not None else None,
                )
            else:
                loss = criterion(logits, batch.y.long())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return {"loss": total_loss / max(1, len(train_loader))}

    def val_step(epoch):
        val_probs, val_y, _ = predict_probs(model, val_loader, device, use_evidential=use_evidential)
        metrics = compute_metrics_from_probs(val_y, val_probs, threshold=fixed_threshold)
        metrics["thr"] = float(fixed_threshold)
        metrics["combo"] = float(0.5 * metrics.get("auc", float("nan")) + 0.5 * metrics.get("acc", float("nan")))
        return metrics

    def test_step(epoch):
        test_probs, test_y, _ = predict_probs(model, test_loader, device, use_evidential=use_evidential)
        metrics = compute_metrics_from_probs(test_y, test_probs, threshold=fixed_threshold)
        metrics["thr"] = float(fixed_threshold)
        metrics["combo"] = float(0.5 * metrics.get("auc", float("nan")) + 0.5 * metrics.get("acc", float("nan")))
        return metrics

    def _on_best(epoch, train_metrics, eval_metrics):
        nonlocal best_state, best_epoch, best_val_metrics, best_test_metrics, best_threshold
        best_epoch = epoch + 1
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_val_metrics = dict(eval_metrics["val"])
        best_test_metrics = dict(eval_metrics["test"])
        best_threshold = float(best_val_metrics.get("thr", fixed_threshold))

    def _fmt_metrics(prefix: str, metrics: Dict[str, float]) -> str:
        return (
            f"{prefix}_acc={metrics['acc']:.4f} "
            f"{prefix}_auc={metrics['auc']:.4f} "
            f"{prefix}_combo={metrics['combo']:.4f} "
            f"{prefix}_mcc={metrics['mcc']:.4f} "
            f"{prefix}_se={metrics['se']:.4f} "
            f"{prefix}_sp={metrics['sp']:.4f} "
            f"{prefix}_ba={metrics['ba']:.4f} "
            f"{prefix}_thr={metrics['thr']:.4f}"
        )

    def _log_builder(epoch, train_metrics, eval_metrics):
        nonlocal best_test_state, best_test_epoch, best_test_metrics_best_acc, best_test_threshold
        test_acc = float(eval_metrics["test"].get("acc", float("nan")))
        if best_test_metrics_best_acc is None or test_acc > best_test_metrics_best_acc.get("acc", -float("inf")):
            best_test_epoch = epoch + 1
            best_test_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_test_metrics_best_acc = dict(eval_metrics["test"])
            best_test_threshold = float(best_test_metrics_best_acc.get("thr", best_test_threshold))

        return (
            f"[seed {seed}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"{_fmt_metrics('val', eval_metrics['val'])} "
            f"{_fmt_metrics('test', eval_metrics['test'])}"
        )

    run_training_loop(
        epochs=epochs,
        train_step=train_step,
        eval_steps={"val": val_step, "test": test_step},
        best_metric=("val", "acc"),
        best_mode="max",
        scheduler=scheduler,
        scheduler_metric=("val", "acc"),
        log_every=1,
        log_fn=lambda msg: print(msg, flush=True),
        log_builder=_log_builder,
        on_best=_on_best,
    )

    if best_state is None:
        raise RuntimeError("No best state captured during training.")

    model.load_state_dict(best_state)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = (
        f"split_seed{seed}_{model_name}_evi{int(use_evidential)}"
        f"_kl{str(kl_weight).replace('.', 'p')}_an{anneal_epochs}_valacc{best_val_metrics['acc']:.4f}.pt"
    )
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_name": model_name,
                "model_class": model.__class__.__name__,
                "use_evidential": use_evidential,
            },
            "split_seed": seed,
            "mapping_mode": mapping_mode,
            "kl_weight": kl_weight,
            "anneal_epochs": anneal_epochs,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "val_metrics": best_val_metrics,
            "test_metrics": best_test_metrics,
        },
        ckpt_path,
    )

    if best_test_state is None:
        best_test_state = best_state
        best_test_epoch = best_epoch
        best_test_metrics_best_acc = dict(best_test_metrics)
        best_test_threshold = best_threshold

    model.load_state_dict(best_test_state)
    ckpt_test_name = (
        f"split_seed{seed}_{model_name}_evi{int(use_evidential)}"
        f"_kl{str(kl_weight).replace('.', 'p')}_an{anneal_epochs}_testacc{best_test_metrics_best_acc['acc']:.4f}.pt"
    )
    ckpt_test_path = os.path.join(ckpt_dir, ckpt_test_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_name": model_name,
                "model_class": model.__class__.__name__,
                "use_evidential": use_evidential,
            },
            "split_seed": seed,
            "mapping_mode": mapping_mode,
            "kl_weight": kl_weight,
            "anneal_epochs": anneal_epochs,
            "best_epoch": best_test_epoch,
            "best_threshold": best_threshold,
            "val_metrics": best_val_metrics,
            "test_metrics": best_test_metrics_best_acc,
        },
        ckpt_test_path,
    )

    return ckpt_path, ckpt_test_path, best_epoch, best_val_metrics, best_test_metrics_best_acc


def evaluate_independent(
    ckpt_path: str,
    independent_csv: str,
    mapping_mode: str,
    batch_size: int,
    device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model_name = cfg.get("model_name", "multiscale")
    model_class = str(cfg.get("model_class", "")).lower()
    if model_name.lower() not in {"multiscale", "subgt", "graphtransformer"}:
        if "multiscale" in model_class:
            model_name = "multiscale"
        elif "subgt" in model_class:
            model_name = "subgt"
        elif "graphtransformer" in model_class:
            model_name = "graphtransformer"
    use_evidential = bool(cfg.get("use_evidential", False))
    threshold = float(ckpt.get("best_threshold", 0.5))

    model = build_model(model_name, use_evidential=use_evidential).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    dataset, loader = build_independent_loader(independent_csv, mapping_mode, batch_size)
    probs, y_true, _ = predict_probs(model, loader, device, use_evidential=use_evidential)
    if y_true is None or len(y_true) == 0:
        return {"acc": float("nan"), "auc": float("nan"), "f1": float("nan"), "mcc": float("nan"), "ba": float("nan"), "se": float("nan"), "sp": float("nan")}
    return compute_metrics_from_probs(y_true, probs, threshold=threshold)


def workflow_for_attribute(
    idx: int,
    dataset_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    seeds: List[int],
    train_ratio: float,
    val_ratio: float,
    mapping_mode: str,
    model_name: str,
    use_evidential: bool,
    batch_size: int,
    epochs: int,
    lr: float,
    kl_weight: float,
    anneal_epochs: int,
    device,
):
    pos_path = resolve_fasta(dataset_dir, "Positive", idx)
    neg_path = resolve_fasta(dataset_dir, "Negative", idx)
    ind_path = resolve_independent(dataset_dir, idx)

    pos_rows = parse_fasta(pos_path, default_label=1)
    neg_rows = parse_fasta(neg_path, default_label=0)
    train_rows = pos_rows + neg_rows

    independent_rows = parse_fasta(ind_path, default_label=None)

    attr_dir = out_dir / f"attr_{idx}"
    csv_dir = attr_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    train_csv = csv_dir / f"trainval_{idx}.csv"
    indep_csv = csv_dir / f"independent_{idx}.csv"

    write_csv(train_csv, train_rows, ["id", "sequence", "label"])
    write_csv(indep_csv, independent_rows, ["id", "sequence", "label"])

    split_results: List[SplitResult] = []

    for seed in seeds:
        rows = [dict(r) for r in train_rows]
        assign_random_split(rows, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        split_csv = csv_dir / f"trainval_{idx}_seed{seed}.csv"
        write_csv(split_csv, rows, ["id", "sequence", "label", "split"])

        ckpt_path, independent_ckpt_path, best_epoch, val_metrics, test_metrics = train_one_seed(
            csv_path=str(split_csv),
            seed=seed,
            mapping_mode=mapping_mode,
            model_name=model_name,
            use_evidential=use_evidential,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            kl_weight=kl_weight,
            anneal_epochs=anneal_epochs,
            device=device,
            ckpt_dir=str(ckpt_dir / f"attr_{idx}"),
        )

        independent_metrics = evaluate_independent(
            ckpt_path=independent_ckpt_path,
            independent_csv=str(indep_csv),
            mapping_mode=mapping_mode,
            batch_size=batch_size,
            device=device,
        )

        split_results.append(
            SplitResult(
                seed=seed,
                split_csv=str(split_csv),
                ckpt_path=ckpt_path,
                independent_ckpt_path=independent_ckpt_path,
                best_epoch=best_epoch,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                independent_metrics=independent_metrics,
            )
        )

    # Save per-attribute CSV summary.
    base_name = "metrics.csv" if idx == 1 else f"metrics({idx}).csv"
    csv_out = out_dir / base_name

    rows_out = []
    for r in split_results:
        row = {
            "seed": r.seed,
            "split_csv": r.split_csv,
            "ckpt_path": r.ckpt_path,
            "independent_ckpt_path": r.independent_ckpt_path,
            "best_epoch": r.best_epoch,
        }
        for k in ["acc", "auc", "mcc", "se", "sp", "ba"]:
            row[f"val_{k}"] = float(r.val_metrics.get(k, float("nan")))
        for k in ["acc", "auc", "mcc", "se", "sp", "ba"]:
            row[f"test_{k}"] = float(r.test_metrics.get(k, float("nan")))
        for k in ["acc", "auc", "mcc", "se", "sp", "ba"]:
            row[f"ind_{k}"] = float(r.independent_metrics.get(k, float("nan")))
        rows_out.append(row)

    fieldnames = [
        "seed",
        "split_csv",
        "ckpt_path",
        "independent_ckpt_path",
        "best_epoch",
        "val_acc",
        "val_auc",
        "val_mcc",
        "val_se",
        "val_sp",
        "val_ba",
        "test_acc",
        "test_auc",
        "test_mcc",
        "test_se",
        "test_sp",
        "test_ba",
        "ind_acc",
        "ind_auc",
        "ind_mcc",
        "ind_se",
        "ind_sp",
        "ind_ba",
    ]
    write_csv(csv_out, rows_out, fieldnames)

    json_out = csv_out.with_suffix(".json")
    payload = {
        "attribute": idx,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "positive": str(pos_path),
        "negative": str(neg_path),
        "independent": str(ind_path),
        "train_csv": str(train_csv),
        "independent_csv": str(indep_csv),
        "seeds": seeds,
        "results": [r.__dict__ for r in split_results],
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return csv_out, json_out


def main():
    parser = argparse.ArgumentParser(description="Batch workflow: FASTA -> CSV -> 5x random split -> train -> independent eval")
    parser.add_argument("--dataset_dir", default="/home/shenxin/EviMSGT/dataset")
    parser.add_argument("--x_min", type=int, default=1)
    parser.add_argument("--x_max", type=int, default=19)
    parser.add_argument("--seeds", default="10,20,30,40,50")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mapping_mode", default="helm_force")
    parser.add_argument("--model", default="multiscale")
    parser.add_argument("--use_evidential", default="1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-5)
    parser.add_argument("--anneal_epochs", type=int, default=5)
    parser.add_argument("--pos_mode", default="2d")
    parser.add_argument("--out_dir", default="/home/shenxin/EviMSGT/results/workflow_batch")
    parser.add_argument("--ckpt_dir", default="/home/shenxin/EviMSGT/ckpt/workflow_batch")
    args = parser.parse_args()

    os.environ["EVIMSGT_POS_MODE"] = str(args.pos_mode)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    use_evidential = str(args.use_evidential).strip().lower() in {"1", "true", "yes", "y"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)

    for idx in range(args.x_min, args.x_max + 1):
        print("=" * 80)
        print(f"Attribute {idx}")
        csv_out, json_out = workflow_for_attribute(
            idx=idx,
            dataset_dir=Path(args.dataset_dir),
            out_dir=out_dir,
            ckpt_dir=ckpt_dir,
            seeds=seeds,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            mapping_mode=args.mapping_mode,
            model_name=args.model,
            use_evidential=use_evidential,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            kl_weight=args.kl_weight,
            anneal_epochs=args.anneal_epochs,
            device=device,
        )
        print(f"Saved CSV: {csv_out}")
        print(f"Saved JSON: {json_out}")


if __name__ == "__main__":
    main()
