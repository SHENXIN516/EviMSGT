import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.optim import Adam
from torch_geometric.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plat_model.model import GraphTransformer, MultiScaleGraphTransformer, SubGT

try:
    from scripts.train_plat import BBBP_Dataset, build_model_config, evidential_loss_from_logits
    from scripts.train_common import make_plateau_scheduler, run_training_loop
except ModuleNotFoundError:
    # Fallback when "scripts" is not recognized as a package.
    from train_plat import BBBP_Dataset, build_model_config, evidential_loss_from_logits
    from train_common import make_plateau_scheduler, run_training_loop


def parse_float_grid(env_value: str, default_values: List[float]) -> List[float]:
    if env_value is None:
        return default_values
    text = env_value.strip()
    if not text:
        return default_values
    out = []
    for token in text.split(','):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    return out if len(out) > 0 else default_values


def append_csv_row(csv_path: str, header: List[str], row: Dict[str, object]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        values = []
        for k in header:
            v = row.get(k, '')
            if isinstance(v, float):
                values.append(f"{v:.6f}")
            else:
                values.append(str(v))
        f.write(','.join(values) + '\n')


def build_model(
    model_name: str,
    use_evidential: bool,
    in_channels: int = 38,
    edge_features: int = 6,
    hidden: int = 256,
    ablation_mode: str = None,
    readout_mode: str = None,
    use_residue_position: bool = None,
    use_terminal_flags: bool = None,
    use_physchem_features: bool = None,
):
    model_name = model_name.lower()
    ablation_mode = ablation_mode or os.getenv('EVIMSGT_ABLATION_MODE', 'full')
    readout_mode = readout_mode or os.getenv('EVIMSGT_READOUT_MODE', 'mean_max')
    if use_residue_position is None:
        use_residue_position = os.getenv('EVIMSGT_USE_RESIDUE_POSITION', '0').lower() in {'1', 'true', 'yes', 'y'}
    if use_terminal_flags is None:
        use_terminal_flags = os.getenv('EVIMSGT_USE_TERMINAL_FLAGS', '0').lower() in {'1', 'true', 'yes', 'y'}
    if use_physchem_features is None:
        use_physchem_features = os.getenv('EVIMSGT_USE_PHYSCHEM_FEATURES', '0').lower() in {'1', 'true', 'yes', 'y'}
    if model_name == 'multiscale':
        return MultiScaleGraphTransformer(
            in_channels=in_channels,
            edge_features=edge_features,
            num_hidden_channels=hidden,
            num_layers=4,
            num_residue_layers=2,
            use_evidential=use_evidential,
            ablation_mode=ablation_mode,
            readout_mode=readout_mode,
            use_residue_position=use_residue_position,
            use_terminal_flags=use_terminal_flags,
            use_physchem_features=use_physchem_features,
        )
    if model_name == 'subgt':
        return SubGT(
            in_channels=in_channels,
            edge_features=edge_features,
            num_hidden_channels=hidden,
            num_layers=6,
        )
    return GraphTransformer(
        in_channels=in_channels,
        edge_features=edge_features,
        num_hidden_channels=hidden,
        use_evidential=use_evidential,
    )


def compute_metrics_from_probs(y_true: np.ndarray, probs_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (probs_pos >= threshold).astype(int)
    out = {}
    try:
        out['auc'] = float(roc_auc_score(y_true, probs_pos))
    except Exception:
        out['auc'] = float('nan')
    try:
        out['f1'] = float(f1_score(y_true, y_pred))
    except Exception:
        out['f1'] = float('nan')
    try:
        out['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out['mcc'] = float('nan')

    out['acc'] = float((y_pred == y_true).mean())
    try:
        out['ba'] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        out['ba'] = float('nan')

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    out['se'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    out['sp'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return out


def predict_probs(model, loader, device, use_evidential: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    y_all = []
    u_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_evidential:
                out = model(batch, return_evidential=True)
                probs = out['probs'][:, 1]
                u = out['uncertainty']
                u_all.append(u.detach().cpu().numpy())
            else:
                logits = model(batch)
                probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_all.append(probs.detach().cpu().numpy())
            y_all.append(batch.y.long().view(-1).detach().cpu().numpy())

    probs_np = np.concatenate(probs_all, axis=0)
    y_np = np.concatenate(y_all, axis=0)
    if len(u_all) > 0:
        u_np = np.concatenate(u_all, axis=0)
    else:
        u_np = np.zeros_like(probs_np)
    return probs_np, y_np, u_np


def make_loaders(dataset: BBBP_Dataset, batch_size: int):
    train_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == 'train']
    val_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == 'val']
    test_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == 'test']

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


def train_one_split(
    csv_path: str,
    split_col: str,
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
    log_every: int = 10,
):
    dataset = BBBP_Dataset(
        csv_path,
        split_col=split_col,
        label_col='label',
        permeability_col='permeability',
        permeability_threshold=None,
        mapping_mode=mapping_mode,
    )
    train_loader, val_loader, test_loader = make_loaders(dataset, batch_size)

    model = build_model(model_name, use_evidential=use_evidential).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = make_plateau_scheduler(optimizer, mode='max', patience=6, factor=0.8)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    best_info = None

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
                )
            else:
                loss = criterion(logits, batch.y.long())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        return {'loss': total_loss / max(1, len(train_loader))}

    def val_step(epoch):
        val_probs, val_y, _ = predict_probs(model, val_loader, device, use_evidential=use_evidential)
        return compute_metrics_from_probs(val_y, val_probs)

    def _on_best(epoch, train_metrics, eval_metrics):
        nonlocal best_val_acc, best_state, best_info
        best_val_acc = float(eval_metrics['val']['acc'])
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_info = {
            'epoch': epoch + 1,
            'train_loss': float(train_metrics['loss']),
            'val_metrics': eval_metrics['val'],
        }

    def _log_builder(epoch, train_metrics, eval_metrics):
        return (
            f"[{split_col}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_acc={eval_metrics['val']['acc']:.4f} "
            f"val_auc={eval_metrics['val']['auc']:.4f}"
        )

    run_training_loop(
        epochs=epochs,
        train_step=train_step,
        eval_steps={'val': val_step},
        best_metric=('val', 'acc'),
        best_mode='max',
        scheduler=scheduler,
        scheduler_metric=('val', 'acc'),
        log_every=log_every,
        log_fn=lambda msg: print(msg, flush=True),
        log_builder=_log_builder,
        on_best=_on_best,
    )

    model.load_state_dict(best_state)
    test_probs, test_y, test_u = predict_probs(model, test_loader, device, use_evidential=use_evidential)
    test_metrics = compute_metrics_from_probs(test_y, test_probs)
    test_metrics['u_mean'] = float(np.mean(test_u)) if use_evidential else float('nan')

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = (
        f"grid_{split_col}_{model_name}_evi{int(use_evidential)}"
        f"_kl{str(kl_weight).replace('.', 'p')}_an{anneal_epochs}_valacc{best_val_acc:.4f}.pt"
    )
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_config': build_model_config(model),
            'split_col': split_col,
            'use_evidential': use_evidential,
            'mapping_mode': mapping_mode,
            'kl_weight': kl_weight,
            'anneal_epochs': anneal_epochs,
            'best_info': best_info,
            'test_metrics': test_metrics,
            'test_probs': test_probs,
            'test_y': test_y,
        },
        ckpt_path,
    )

    return {
        'split': split_col,
        'ckpt_path': ckpt_path,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
    }


def load_model_from_ckpt(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('model_config', {})
    model_name = str(cfg.get('model_name', 'MultiScaleGraphTransformer'))
    model_class = str(cfg.get('model_class', model_name))
    use_evidential = bool(cfg.get('use_evidential', False))
    model_key = model_name.lower()
    class_key = model_class.lower()

    if model_key == 'multiscale' or 'multiscale' in class_key:
        model = MultiScaleGraphTransformer(
            in_channels=int(cfg.get('in_channels', 38)),
            edge_features=int(cfg.get('edge_features', 6)),
            num_hidden_channels=int(cfg.get('num_hidden_channels', 256)),
            num_layers=int(cfg.get('num_layers', 4)),
            num_residue_layers=int(cfg.get('num_residue_layers', 2)),
            num_attention_heads=int(cfg.get('num_attention_heads', 4)),
            dropout_rate=float(cfg.get('dropout_rate', 0.1)),
            norm_to_apply=str(cfg.get('norm_to_apply', 'batch')),
            use_evidential=use_evidential,
            ablation_mode=str(cfg.get('ablation_mode', 'full')),
            readout_mode=str(cfg.get('readout_mode', 'mean_max')),
            use_residue_position=bool(cfg.get('use_residue_position', False)),
            use_terminal_flags=bool(cfg.get('use_terminal_flags', False)),
            use_physchem_features=bool(cfg.get('use_physchem_features', False)),
        )
    elif model_key == 'graphtransformer' or 'graphtransformer' in class_key:
        model = GraphTransformer(
            in_channels=int(cfg.get('in_channels', 38)),
            edge_features=int(cfg.get('edge_features', 6)),
            num_hidden_channels=int(cfg.get('num_hidden_channels', 256)),
            num_layers=int(cfg.get('num_layers', 4)),
            num_attention_heads=int(cfg.get('num_attention_heads', 4)),
            dropout_rate=float(cfg.get('dropout_rate', 0.1)),
            norm_to_apply=str(cfg.get('norm_to_apply', 'batch')),
            use_evidential=use_evidential,
        )
    else:
        model = SubGT(
            in_channels=int(cfg.get('in_channels', 38)),
            edge_features=int(cfg.get('edge_features', 6)),
            num_hidden_channels=int(cfg.get('num_hidden_channels', 256)),
            num_layers=int(cfg.get('num_layers', 6)),
        )
        use_evidential = False

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    return model, use_evidential


def ensemble_eval_for_split(
    csv_path: str,
    split_col: str,
    mapping_mode: str,
    batch_size: int,
    model_paths: List[str],
    device,
):
    dataset = BBBP_Dataset(
        csv_path,
        split_col=split_col,
        label_col='label',
        permeability_col='permeability',
        permeability_threshold=None,
        mapping_mode=mapping_mode,
    )
    _, _, test_loader = make_loaders(dataset, batch_size)

    probs_list = []
    y_ref = None
    for p in model_paths:
        model, use_evidential = load_model_from_ckpt(p, device)
        probs, y_true, _ = predict_probs(model, test_loader, device, use_evidential=use_evidential)
        probs_list.append(probs)
        if y_ref is None:
            y_ref = y_true

    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    metrics = compute_metrics_from_probs(y_ref, avg_probs)
    return metrics


def main():
    csv_path = os.getenv('EVIMSGT_CSV', '/home/shenxin/EviMSGT/summary_cycpeptmpdb_5splits.csv')
    mapping_mode = os.getenv('EVIMSGT_MAPPING_MODE', 'hybrid').strip().lower()
    model_name = os.getenv('EVIMSGT_MODEL', 'multiscale').strip().lower()
    use_evidential = os.getenv('EVIMSGT_USE_EVIDENTIAL', '1').lower() in {'1', 'true', 'yes', 'y'}
    batch_size = int(os.getenv('EVIMSGT_BATCH_SIZE', '32'))
    epochs = int(os.getenv('EVIMSGT_EPOCHS', '200'))
    lr = float(os.getenv('EVIMSGT_LR', '5e-4'))

    run_limit = int(os.getenv('EVIMSGT_GRID_LIMIT', '0'))
    kl_grid = parse_float_grid(os.getenv('EVIMSGT_KL_GRID', ''), [1e-5, 3e-5, 1e-4, 3e-4])
    anneal_grid = [5, 10, 20, 40]
    splits = ['split1', 'split2', 'split3', 'split4', 'split5']

    out_dir = '/home/shenxin/EviMSGT/results/ensemble_grid'
    ckpt_dir = '/home/shenxin/EviMSGT/ckpt/ensemble_grid'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    split_live_csv = os.path.join(out_dir, 'split_metrics_live.csv')
    ensemble_live_csv = os.path.join(out_dir, 'ensemble_metrics_live.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grid_items = [(k, a) for k in kl_grid for a in anneal_grid]
    if run_limit > 0:
        grid_items = grid_items[:run_limit]

    all_results = []

    print(f"Grid size: {len(grid_items)}")
    print(f"Splits: {splits}")
    print(f"Model: {model_name}, evidential={use_evidential}, mapping_mode={mapping_mode}")

    for kl_weight, anneal_epochs in grid_items:
        print('=' * 100)
        print(f"Grid config -> kl_weight={kl_weight}, anneal_epochs={anneal_epochs}")

        split_train_results = []
        model_paths = []

        for split_col in splits:
            print('-' * 80)
            print(f"Training split: {split_col}")
            r = train_one_split(
                csv_path=csv_path,
                split_col=split_col,
                mapping_mode=mapping_mode,
                model_name=model_name,
                use_evidential=use_evidential,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                kl_weight=kl_weight,
                anneal_epochs=anneal_epochs,
                device=device,
                ckpt_dir=ckpt_dir,
            )
            split_train_results.append(r)
            model_paths.append(r['ckpt_path'])
            print(f"best_val_acc={r['best_val_acc']:.4f}, test_acc={r['test_metrics']['acc']:.4f}", flush=True)
            append_csv_row(
                split_live_csv,
                header=[
                    'kl_weight',
                    'anneal_epochs',
                    'split',
                    'best_val_acc',
                    'test_acc',
                    'test_auc',
                    'test_f1',
                    'test_mcc',
                    'test_ba',
                    'test_se',
                    'test_sp',
                    'test_u_mean',
                    'ckpt_path',
                ],
                row={
                    'kl_weight': kl_weight,
                    'anneal_epochs': anneal_epochs,
                    'split': split_col,
                    'best_val_acc': r['best_val_acc'],
                    'test_acc': r['test_metrics'].get('acc', float('nan')),
                    'test_auc': r['test_metrics'].get('auc', float('nan')),
                    'test_f1': r['test_metrics'].get('f1', float('nan')),
                    'test_mcc': r['test_metrics'].get('mcc', float('nan')),
                    'test_ba': r['test_metrics'].get('ba', float('nan')),
                    'test_se': r['test_metrics'].get('se', float('nan')),
                    'test_sp': r['test_metrics'].get('sp', float('nan')),
                    'test_u_mean': r['test_metrics'].get('u_mean', float('nan')),
                    'ckpt_path': r['ckpt_path'],
                },
            )

        # Ensemble across 5 trained models, evaluated on each official split test set.
        ensemble_by_split = {}
        for split_col in splits:
            m = ensemble_eval_for_split(
                csv_path=csv_path,
                split_col=split_col,
                mapping_mode=mapping_mode,
                batch_size=batch_size,
                model_paths=model_paths,
                device=device,
            )
            ensemble_by_split[split_col] = m
            print(
                f"ensemble_{split_col}: test_acc={m['acc']:.4f}, auc={m['auc']:.4f}, f1={m['f1']:.4f}, mcc={m['mcc']:.4f}",
                flush=True,
            )

        ensemble_acc_mean = float(np.mean([ensemble_by_split[s]['acc'] for s in splits]))
        ensemble_auc_mean = float(np.mean([ensemble_by_split[s]['auc'] for s in splits]))

        result = {
            'kl_weight': kl_weight,
            'anneal_epochs': anneal_epochs,
            'split_train_results': split_train_results,
            'ensemble_by_split': ensemble_by_split,
            'ensemble_acc_mean': ensemble_acc_mean,
            'ensemble_auc_mean': ensemble_auc_mean,
            'model_paths': model_paths,
        }
        all_results.append(result)

        append_csv_row(
            ensemble_live_csv,
            header=[
                'kl_weight',
                'anneal_epochs',
                'ensemble_acc_mean',
                'ensemble_auc_mean',
                'split1_acc',
                'split2_acc',
                'split3_acc',
                'split4_acc',
                'split5_acc',
            ],
            row={
                'kl_weight': kl_weight,
                'anneal_epochs': anneal_epochs,
                'ensemble_acc_mean': ensemble_acc_mean,
                'ensemble_auc_mean': ensemble_auc_mean,
                'split1_acc': ensemble_by_split['split1']['acc'],
                'split2_acc': ensemble_by_split['split2']['acc'],
                'split3_acc': ensemble_by_split['split3']['acc'],
                'split4_acc': ensemble_by_split['split4']['acc'],
                'split5_acc': ensemble_by_split['split5']['acc'],
            },
        )

        with open(os.path.join(out_dir, 'grid_results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Rank by ensemble mean ACC.
    ranked = sorted(all_results, key=lambda x: x['ensemble_acc_mean'], reverse=True)
    summary_lines = []
    summary_lines.append('kl_weight,anneal_epochs,ensemble_acc_mean,ensemble_auc_mean')
    for r in ranked:
        summary_lines.append(
            f"{r['kl_weight']},{r['anneal_epochs']},{r['ensemble_acc_mean']:.6f},{r['ensemble_auc_mean']:.6f}"
        )

    summary_path = os.path.join(out_dir, 'grid_summary.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print('=' * 100)
    print(f"Saved grid summary: {summary_path}")
    if ranked:
        best = ranked[0]
        print(
            f"Best config -> kl_weight={best['kl_weight']}, anneal_epochs={best['anneal_epochs']}, "
            f"ensemble_acc_mean={best['ensemble_acc_mean']:.4f}, ensemble_auc_mean={best['ensemble_auc_mean']:.4f}"
        )


if __name__ == '__main__':
    main()
