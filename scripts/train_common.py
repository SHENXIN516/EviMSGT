from typing import Callable, Dict, List, Optional, Tuple

import torch


def should_log_epoch(epoch_idx: int, total_epochs: int, log_every: int = 10) -> bool:
    epoch_num = epoch_idx + 1
    return epoch_num == 1 or epoch_num % log_every == 0 or epoch_num == total_epochs


def make_plateau_scheduler(
    optimizer,
    mode: str = "max",
    patience: int = 6,
    factor: float = 0.8,
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode,
        patience=patience,
        factor=factor,
        verbose=False,
    )


def _is_better(new_value: float, old_value: float, mode: str) -> bool:
    if old_value is None:
        return True
    if mode == "max":
        return new_value > old_value
    if mode == "min":
        return new_value < old_value
    raise ValueError(f"Unsupported mode: {mode}")


def run_training_loop(
    epochs: int,
    train_step: Callable[[int], Dict[str, float]],
    eval_steps: Dict[str, Callable[[int], Dict[str, float]]],
    best_metric: Tuple[str, str],
    best_mode: str,
    scheduler=None,
    scheduler_metric: Optional[Tuple[str, str]] = None,
    log_every: int = 10,
    log_fn: Callable[[str], None] = print,
    log_builder: Optional[Callable[[int, Dict[str, float], Dict[str, Dict[str, float]]], str]] = None,
    on_best: Optional[Callable[[int, Dict[str, float], Dict[str, Dict[str, float]]], None]] = None,
):
    history: List[Dict[str, float]] = []
    best_epoch = None
    best_value = None

    best_split, best_key = best_metric

    for epoch in range(epochs):
        train_metrics = train_step(epoch)
        eval_metrics: Dict[str, Dict[str, float]] = {}
        for split_name, step_fn in eval_steps.items():
            eval_metrics[split_name] = step_fn(epoch)

        if scheduler is not None and scheduler_metric is not None:
            sched_split, sched_key = scheduler_metric
            scheduler.step(eval_metrics[sched_split][sched_key])

        row = {"epoch": float(epoch + 1)}

        def _flatten(prefix: str, metrics: Dict[str, object]):
            for k, v in metrics.items():
                key = f"{prefix}_{k}"
                if isinstance(v, dict):
                    _flatten(key, v)
                else:
                    row[key] = float(v)

        _flatten("train", train_metrics)
        for split_name, split_metrics in eval_metrics.items():
            _flatten(split_name, split_metrics)
        history.append(row)

        cur_value = float(eval_metrics[best_split][best_key])
        if _is_better(cur_value, best_value, best_mode):
            best_value = cur_value
            best_epoch = epoch + 1
            if on_best is not None:
                on_best(epoch, train_metrics, eval_metrics)

        if should_log_epoch(epoch, epochs, log_every=log_every):
            if log_builder is not None:
                log_fn(log_builder(epoch, train_metrics, eval_metrics))

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_value": best_value,
    }
