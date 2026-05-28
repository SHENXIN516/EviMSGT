import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data

sys.path.append('/home/shenxin/EviMSGT')
from plat_model.model import GraphTransformer


ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ATOM_HYBRID = ['sp', 'sp2', 'sp3', 'sp3d', 'sp3d2', 'other']
ATOM_HS = [0, 1, 2, 3, 4]


def build_atom_feature_names(explicit_h: bool = False) -> List[str]:
    names = []
    names += [f'atom_symbol_{s}' for s in ATOM_SYMBOLS]
    names += [f'atom_degree_{d}' for d in ATOM_DEGREES]
    names += ['formal_charge', 'num_radical_electrons']
    names += [f'hybridization_{h}' for h in ATOM_HYBRID]
    names += ['is_aromatic']
    if not explicit_h:
        names += [f'total_hs_{n}' for n in ATOM_HS]
    return names


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_h=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    )
    results += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    results += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    results += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ],
    )
    results += [atom.GetIsAromatic()]
    if not explicit_h:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results, dtype=np.float32)


def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
        )
    return np.array(bond_feats, dtype=np.float32)


def mol_to_graph(smiles, explicit_h=False, use_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.RemoveHs(mol)
    x = np.array([calc_atom_features(atom, explicit_h=explicit_h) for atom in mol.GetAtoms()], dtype=np.float32)

    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        feat = calc_bond_features(bond, use_chirality=use_chirality)

        row += [a, b]
        col += [b, a]
        edge_attr += [feat, feat]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data.pos = torch.zeros((data.x.size(0), 3), dtype=torch.float)
    return data


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('Checkpoint format not supported: missing model_state_dict')

    state_dict = ckpt['model_state_dict']
    model_config = ckpt.get('model_config', {})
    preprocessing = ckpt.get('preprocessing', {})

    in_channels = int(model_config.get('in_channels', state_dict['node_encoder.weight'].shape[1]))
    edge_features = int(model_config.get('edge_features', state_dict['edge_encoder.weight'].shape[1]))
    num_hidden_channels = int(model_config.get('num_hidden_channels', state_dict['node_encoder.weight'].shape[0]))

    gt_layers = set()
    for key in state_dict.keys():
        if key.startswith('gt_block.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                gt_layers.add(int(parts[1]))
    num_layers = max(gt_layers) + 1 if gt_layers else int(model_config.get('num_layers', 4))

    num_attention_heads = int(model_config.get('num_attention_heads', 4))
    dropout_rate = float(model_config.get('dropout_rate', 0.1))
    norm_to_apply = str(model_config.get('norm_to_apply', 'batch'))

    model = GraphTransformer(
        in_channels=in_channels,
        edge_features=edge_features,
        num_hidden_channels=num_hidden_channels,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate,
        norm_to_apply=norm_to_apply,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt, preprocessing


def export_attention(attentions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        edge_index = layer_attn['edge_index'].detach().cpu().numpy()
        edge_attention = layer_attn['edge_attention'].detach().cpu().numpy()
        edge_logits = layer_attn['edge_attention_logits'].detach().cpu().numpy()

        edge_mean = edge_attention.mean(axis=1)
        edge_max = edge_attention.max(axis=1)

        csv_path = os.path.join(out_dir, f'attention_layer_{layer_idx}.csv')
        num_heads = edge_attention.shape[1]
        header = ['layer', 'edge_id', 'src', 'dst', 'attn_mean', 'attn_max']
        header += [f'attn_head_{h}' for h in range(num_heads)]
        header += [f'logit_head_{h}' for h in range(num_heads)]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for edge_id in range(edge_attention.shape[0]):
                row = [
                    layer_idx,
                    edge_id,
                    int(edge_index[0, edge_id]),
                    int(edge_index[1, edge_id]),
                    float(edge_mean[edge_id]),
                    float(edge_max[edge_id]),
                ]
                row += [float(edge_attention[edge_id, h]) for h in range(num_heads)]
                row += [float(edge_logits[edge_id, h]) for h in range(num_heads)]
                writer.writerow(row)

        summary_rows.append(
            {
                'layer': layer_idx,
                'num_edges': int(edge_attention.shape[0]),
                'num_heads': int(num_heads),
                'attn_mean_overall': float(edge_attention.mean()),
                'attn_max_overall': float(edge_attention.max()),
            }
        )

    summary_json = os.path.join(out_dir, 'attention_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)


def aggregate_node_attention(attentions: List[Optional[Dict[str, torch.Tensor]]], num_nodes: int) -> np.ndarray:
    if not attentions:
        return np.zeros((num_nodes,), dtype=np.float32)

    layer_node_scores = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        edge_index = layer_attn['edge_index']
        edge_attention = layer_attn['edge_attention']
        edge_score = edge_attention.mean(dim=1)
        dst = edge_index[1]
        node_score = torch.zeros((num_nodes,), dtype=edge_score.dtype, device=edge_score.device)
        node_score.index_add_(0, dst, edge_score)
        layer_node_scores.append(node_score)

    if not layer_node_scores:
        return np.zeros((num_nodes,), dtype=np.float32)

    stacked = torch.stack(layer_node_scores, dim=0)
    return stacked.mean(dim=0).detach().cpu().numpy().astype(np.float32)


def export_integrated_gradients(
    model: nn.Module,
    data: Data,
    target_class: int,
    out_dir: str,
    ig_steps: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from captum.attr import IntegratedGradients
    except Exception:
        return None, 'captum not installed. Please install captum to use Integrated Gradients.'

    model.eval()
    x = data.x.detach().clone().requires_grad_(True)
    baseline = torch.zeros_like(x)

    edge_index = data.edge_index.detach().clone()
    edge_attr = data.edge_attr.detach().clone()
    batch = data.batch.detach().clone()
    pos = data.pos.detach().clone() if hasattr(data, 'pos') and data.pos is not None else torch.zeros((x.size(0), 3), device=x.device)

    def forward_func(node_x):
        local_data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr)
        local_data.batch = batch
        local_data.pos = pos
        logits = model(local_data)
        return logits[:, target_class]

    ig = IntegratedGradients(forward_func)
    attrs = ig.attribute(x, baselines=baseline, n_steps=ig_steps)
    node_scores = attrs.abs().sum(dim=1).detach().cpu().numpy().astype(np.float32)
    attrs_np = attrs.detach().cpu().numpy().astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'integrated_gradients_node.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['node_id', 'ig_abs_sum'] + [f'ig_feature_{i}' for i in range(attrs_np.shape[1])]
        writer.writerow(header)
        for i in range(attrs_np.shape[0]):
            writer.writerow([i, float(node_scores[i])] + [float(v) for v in attrs_np[i]])

    return node_scores, None


class GraphClassifierWrapper(nn.Module):
    """Wrap GraphTransformer to match explainer APIs that take tensor inputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None, pos=None):
        graph_data = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            graph_data.edge_attr = edge_attr
        if batch is None:
            graph_data.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            graph_data.batch = batch
        if pos is None:
            graph_data.pos = torch.zeros((x.size(0), 3), dtype=torch.float32, device=x.device)
        else:
            graph_data.pos = pos
        return self.model(graph_data)


def export_gnn_explainer(
    model: nn.Module,
    data: Data,
    target_class: int,
    out_dir: str,
    epochs: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    except Exception:
        return None, 'torch_geometric.explain is unavailable. Please install a compatible torch-geometric version.'

    wrapped_model = GraphClassifierWrapper(model)
    wrapped_model.eval()

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    try:
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            pos=data.pos,
            target=torch.tensor([target_class], device=data.x.device),
        )
    except Exception as e:
        return None, f'GNNExplainer failed: {e}'

    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    if node_mask is None:
        return None, 'GNNExplainer returned empty node mask.'

    if node_mask.dim() == 2:
        node_scores = node_mask.abs().sum(dim=1)
    else:
        node_scores = node_mask.abs()

    node_scores_np = node_scores.detach().cpu().numpy().astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    node_csv = os.path.join(out_dir, 'gnnexplainer_node.csv')
    with open(node_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'gnnexplainer_score'])
        for i, score in enumerate(node_scores_np.tolist()):
            writer.writerow([i, float(score)])

    if edge_mask is not None:
        edge_mask_np = edge_mask.detach().cpu().numpy().astype(np.float32)
        edge_index_np = data.edge_index.detach().cpu().numpy()
        edge_csv = os.path.join(out_dir, 'gnnexplainer_edge.csv')
        with open(edge_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['edge_id', 'src', 'dst', 'gnnexplainer_edge_score'])
            for edge_id, score in enumerate(edge_mask_np.tolist()):
                writer.writerow([edge_id, int(edge_index_np[0, edge_id]), int(edge_index_np[1, edge_id]), float(score)])

    return node_scores_np, None


def export_shap_ready_csv(
    data: Data,
    out_dir: str,
    feature_names: List[str],
    pred_json: Dict,
    attention_node_scores: Optional[np.ndarray] = None,
    ig_node_scores: Optional[np.ndarray] = None,
    gnn_node_scores: Optional[np.ndarray] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    x_np = data.x.detach().cpu().numpy()

    # Backward compatibility for checkpoints with a different atom feature dimensionality.
    if len(feature_names) != x_np.shape[1]:
        feature_names = [f'feature_{i}' for i in range(x_np.shape[1])]

    csv_path = os.path.join(out_dir, 'shap_ready_features.csv')
    header = ['node_id'] + feature_names + [
        'attention_score',
        'ig_score',
        'gnnexplainer_score',
        'pred_label',
        'prob_class0',
        'prob_class1',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(x_np.shape[0]):
            writer.writerow(
                [i]
                + [float(v) for v in x_np[i].tolist()]
                + [
                    float(attention_node_scores[i]) if attention_node_scores is not None else np.nan,
                    float(ig_node_scores[i]) if ig_node_scores is not None else np.nan,
                    float(gnn_node_scores[i]) if gnn_node_scores is not None else np.nan,
                    int(pred_json['pred_label']),
                    float(pred_json['prob_class0']),
                    float(pred_json['prob_class1']),
                ]
            )


def parse_methods(method_arg: str) -> List[str]:
    raw = [m.strip().lower() for m in method_arg.split(',') if m.strip()]
    if not raw:
        return ['attention']
    if 'all' in raw:
        return ['attention', 'ig', 'gnnexplainer']
    valid = {'attention', 'ig', 'gnnexplainer'}
    methods = [m for m in raw if m in valid]
    return methods if methods else ['attention']


def main():
    parser = argparse.ArgumentParser(description='Single-sample inference and explanation export for LiBP checkpoint')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--smiles', required=True, help='Single SMILES string')
    parser.add_argument('--out_dir', default='/home/shenxin/EviMSGT/eval_out/explain_single', help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda')
    parser.add_argument(
        '--method',
        default='attention',
        help='Explanation method(s): attention, ig, gnnexplainer, all, or comma-separated combo',
    )
    parser.add_argument('--target_class', type=int, default=-1, help='Target class index for IG/GNNExplainer; -1 uses predicted class')
    parser.add_argument('--ig_steps', type=int, default=64, help='Integrated Gradients steps')
    parser.add_argument('--gnnexplainer_epochs', type=int, default=200, help='GNNExplainer optimization epochs')
    args = parser.parse_args()

    device = torch.device(args.device)
    methods = parse_methods(args.method)

    model, ckpt, preprocessing = load_checkpoint(args.ckpt, device)

    explicit_h = bool(preprocessing.get('explicit_H', False))
    use_chirality = bool(preprocessing.get('use_chirality', False))
    atom_feature_names = build_atom_feature_names(explicit_h=explicit_h)

    data = mol_to_graph(args.smiles, explicit_h=explicit_h, use_chirality=use_chirality).to(device)

    with torch.no_grad():
        logits, attentions = model(data, return_attention=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred = int(np.argmax(probs))

    target_class = pred if args.target_class < 0 else int(args.target_class)

    os.makedirs(args.out_dir, exist_ok=True)

    pred_json = {
        'smiles': args.smiles,
        'pred_label': pred,
        'target_class': target_class,
        'prob_class0': float(probs[0]),
        'prob_class1': float(probs[1]),
        'checkpoint': args.ckpt,
        'epoch': int(ckpt.get('epoch', -1)) if isinstance(ckpt.get('epoch', -1), (int, np.integer)) else ckpt.get('epoch', -1),
        'metrics': ckpt.get('metrics', {}),
    }

    with open(os.path.join(args.out_dir, 'prediction.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=2)

    method_status = {}
    attention_node_scores = None
    ig_node_scores = None
    gnn_node_scores = None

    if 'attention' in methods:
        export_attention(attentions, args.out_dir)
        attention_node_scores = aggregate_node_attention(attentions, num_nodes=data.x.size(0))
        method_status['attention'] = 'ok'

    if 'ig' in methods:
        ig_node_scores, ig_error = export_integrated_gradients(
            model,
            data,
            target_class=target_class,
            out_dir=args.out_dir,
            ig_steps=args.ig_steps,
        )
        method_status['ig'] = 'ok' if ig_error is None else f'failed: {ig_error}'

    if 'gnnexplainer' in methods:
        gnn_node_scores, gnn_error = export_gnn_explainer(
            model,
            data,
            target_class=target_class,
            out_dir=args.out_dir,
            epochs=args.gnnexplainer_epochs,
        )
        method_status['gnnexplainer'] = 'ok' if gnn_error is None else f'failed: {gnn_error}'

    export_shap_ready_csv(
        data=data,
        out_dir=args.out_dir,
        feature_names=atom_feature_names,
        pred_json=pred_json,
        attention_node_scores=attention_node_scores,
        ig_node_scores=ig_node_scores,
        gnn_node_scores=gnn_node_scores,
    )

    with open(os.path.join(args.out_dir, 'explain_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'methods': methods, 'status': method_status}, f, ensure_ascii=False, indent=2)

    print('Done.')
    print(f"Prediction saved to: {os.path.join(args.out_dir, 'prediction.json')}")
    if 'attention' in methods:
        print(f"Attention summary saved to: {os.path.join(args.out_dir, 'attention_summary.json')}")
    if 'ig' in methods:
        print(f"Integrated Gradients saved to: {os.path.join(args.out_dir, 'integrated_gradients_node.csv')}")
    if 'gnnexplainer' in methods:
        print(f"GNNExplainer node scores saved to: {os.path.join(args.out_dir, 'gnnexplainer_node.csv')}")
    print(f"SHAP-ready features saved to: {os.path.join(args.out_dir, 'shap_ready_features.csv')}")
    print(f"Method status saved to: {os.path.join(args.out_dir, 'explain_manifest.json')}")


if __name__ == '__main__':
    main()
