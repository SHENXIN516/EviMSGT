from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from sklearn.ensemble import RandomForestRegressor

import torch

ROOT = "/home/shenxin/EviMSGT"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from export_attention_single import aggregate_node_attention, load_checkpoint, mol_to_graph


CANDIDATE_SMILES_COLUMNS = ["SMILES", "smiles", "sequence", "mol"]

ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]
MOTIF_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "carboxyl": "C(=O)[OX2H1,OX1-]",
    "aromatic_ring": "a1aaaaa1",
    "halogen": "[F,Cl,Br,I]",
    "sulfonamide": "S(=O)(=O)N",
    "hetero5_ring": "[r5;!#6]",
    "hetero6_ring": "[r6;!#6]",
}


@dataclass
class MolRow:
    idx: int
    smiles: str
    label: int
    mol: Chem.Mol


def detect_smiles_col(df: pd.DataFrame) -> str:
    for c in CANDIDATE_SMILES_COLUMNS:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find SMILES column. Available columns: {list(df.columns)}")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_dataset(csv_path: str, smiles_col: Optional[str], label_col: str) -> List[MolRow]:
    df = pd.read_csv(csv_path)
    s_col = smiles_col if smiles_col and smiles_col in df.columns else detect_smiles_col(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    rows: List[MolRow] = []
    for i, r in df.iterrows():
        can = canonicalize_smiles(r[s_col])
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        rows.append(MolRow(idx=int(i), smiles=can, label=int(r[label_col]), mol=mol))
    return rows


def motif_feature_names() -> List[str]:
    names = [f"atom_count_{a}" for a in ATOM_TYPES]
    names += [f"motif_count_{k}" for k in MOTIF_SMARTS.keys()]
    names += ["num_atoms", "num_bonds", "tpsa", "logp", "hbd", "hba", "rot_bonds", "ring_count"]
    return names


def featurize_molecule(mol: Chem.Mol) -> np.ndarray:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    feats: List[float] = []

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in ATOM_TYPES:
        feats.append(float(symbols.count(a)))

    for smarts in MOTIF_SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        cnt = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
        feats.append(float(cnt))

    feats += [
        float(mol.GetNumAtoms()),
        float(mol.GetNumBonds()),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Crippen.MolLogP(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
    ]
    return np.asarray(feats, dtype=np.float32)


def model_predict_probabilities(
    rows: Sequence[MolRow],
    ckpt_path: str,
    device: str,
) -> Dict[str, float]:
    torch_device = torch.device(device)
    model, _, preprocessing = load_checkpoint(ckpt_path, torch_device)

    explicit_h = bool(preprocessing.get("explicit_H", False))
    use_chirality = bool(preprocessing.get("use_chirality", False))

    out: Dict[str, float] = {}
    with torch.no_grad():
        for r in rows:
            data = mol_to_graph(r.smiles, explicit_h=explicit_h, use_chirality=use_chirality).to(torch_device)
            logits = model(data)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            out[r.smiles] = float(probs[1])
    return out


def run_global_shap(
    rows: Sequence[MolRow],
    probs: Dict[str, float],
    out_dir: str,
    top_k: int,
) -> str:
    import shap

    X = np.stack([featurize_molecule(r.mol) for r in rows], axis=0)
    y = np.asarray([probs[r.smiles] for r in rows], dtype=np.float32)
    f_names = motif_feature_names()

    surrogate = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    surrogate.fit(X, y)

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    global_imp = np.mean(np.abs(shap_values), axis=0)

    imp_df = pd.DataFrame(
        {
            "feature": f_names,
            "mean_abs_shap": global_imp,
            "surrogate_feature_importance": surrogate.feature_importances_,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "global_shap_feature_ranking.csv")
    imp_df.to_csv(csv_path, index=False)

    top_df = imp_df.head(top_k)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1])
    plt.xlabel("mean(|SHAP|)")
    plt.title("Global Feature Contribution to BBB+ (Surrogate SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_shap_top_features.png"), dpi=220)
    plt.close()

    return csv_path


def collect_attention_for_smiles(
    smiles: str,
    ckpt_path: str,
    device: str,
) -> Tuple[np.ndarray, float, int]:
    torch_device = torch.device(device)
    model, _, preprocessing = load_checkpoint(ckpt_path, torch_device)

    explicit_h = bool(preprocessing.get("explicit_H", False))
    use_chirality = bool(preprocessing.get("use_chirality", False))

    data = mol_to_graph(smiles, explicit_h=explicit_h, use_chirality=use_chirality).to(torch_device)
    with torch.no_grad():
        logits, attentions = model(data, return_attention=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    node_scores = aggregate_node_attention(attentions, num_nodes=data.x.size(0))
    pred = int(np.argmax(probs))
    return node_scores, float(probs[1]), pred


def _score_to_color(v: float):
    import matplotlib.cm as cm

    cmap = cm.get_cmap("YlOrRd")
    r, g, b, _ = cmap(float(v))
    return (float(r), float(g), float(b))


def render_attention_mol(
    smiles: str,
    atom_scores: np.ndarray,
    title: str,
    atom_emphasis: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scores = np.asarray(atom_scores, dtype=np.float32)
    if scores.size != mol.GetNumAtoms():
        scores = np.resize(scores, (mol.GetNumAtoms(),))

    if float(scores.max()) - float(scores.min()) < 1e-9:
        norm = np.zeros_like(scores)
    else:
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    highlight_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {i: _score_to_color(float(norm[i])) for i in highlight_atoms}
    atom_radii = {i: 0.26 + 0.24 * float(norm[i]) for i in highlight_atoms}

    if atom_emphasis:
        for i, tag in atom_emphasis.items():
            if i not in atom_colors:
                continue
            if tag == "backbone":
                atom_radii[i] = max(atom_radii[i], 0.48)
            elif tag == "sidechain":
                atom_radii[i] = max(atom_radii[i], 0.35)

    img = Draw.MolToImage(
        mol,
        size=(640, 500),
        legend=title,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    return np.asarray(img)


def choose_similar_pairs(rows: Sequence[MolRow], n_pairs: int = 3) -> List[Tuple[MolRow, MolRow, float]]:
    bbb_pos = [r for r in rows if r.label == 1]
    fps: List[Tuple[MolRow, DataStructs.cDataStructs.ExplicitBitVect]] = []
    for r in bbb_pos:
        fp = AllChem.GetMorganFingerprintAsBitVect(r.mol, 2, nBits=2048)
        fps.append((r, fp))

    candidates: List[Tuple[float, int, int]] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = float(DataStructs.TanimotoSimilarity(fps[i][1], fps[j][1]))
            if sim >= 0.45:
                candidates.append((sim, i, j))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used = set()
    out: List[Tuple[MolRow, MolRow, float]] = []
    for sim, i, j in candidates:
        a, b = fps[i][0], fps[j][0]
        if a.idx in used or b.idx in used:
            continue
        out.append((a, b, sim))
        used.add(a.idx)
        used.add(b.idx)
        if len(out) >= n_pairs:
            break
    return out


def run_local_case_studies(
    rows: Sequence[MolRow],
    ckpt_path: str,
    device: str,
    out_dir: str,
    n_pairs: int,
):
    os.makedirs(out_dir, exist_ok=True)
    pairs = choose_similar_pairs(rows, n_pairs=n_pairs)

    records = []
    for k, (a, b, sim) in enumerate(pairs, start=1):
        a_scores, a_p1, a_pred = collect_attention_for_smiles(a.smiles, ckpt_path, device)
        b_scores, b_p1, b_pred = collect_attention_for_smiles(b.smiles, ckpt_path, device)

        a_img = render_attention_mol(a.smiles, a_scores, f"Mol A | p(BBB+)={a_p1:.3f} | pred={a_pred}")
        b_img = render_attention_mol(b.smiles, b_scores, f"Mol B | p(BBB+)={b_p1:.3f} | pred={b_pred}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].imshow(a_img)
        axes[0].axis("off")
        axes[1].imshow(b_img)
        axes[1].axis("off")
        fig.suptitle(f"Case Pair {k} | Tanimoto={sim:.3f}", fontsize=13)
        fig.tight_layout()

        out_png = os.path.join(out_dir, f"pair_{k}_attention_heatmap.png")
        fig.savefig(out_png, dpi=220)
        plt.close(fig)

        records.append(
            {
                "pair_id": k,
                "tanimoto": sim,
                "mol_a_smiles": a.smiles,
                "mol_b_smiles": b.smiles,
                "mol_a_prob_bbb_plus": a_p1,
                "mol_b_prob_bbb_plus": b_p1,
                "mol_a_pred": a_pred,
                "mol_b_pred": b_pred,
                "image": out_png,
            }
        )

    pd.DataFrame(records).to_csv(os.path.join(out_dir, "local_case_pairs.csv"), index=False)


def peptide_backbone_atoms(mol: Chem.Mol) -> set:
    backbone = set()
    patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    if patt is None:
        return backbone

    for match in mol.GetSubstructMatches(patt):
        # match order: N, carbonyl C, oxygen
        if len(match) == 3:
            n_id, c_id, o_id = match
            backbone.update([n_id, c_id, o_id])
            c_atom = mol.GetAtomWithIdx(c_id)
            for nb in c_atom.GetNeighbors():
                sym = nb.GetSymbol()
                if nb.GetIdx() == n_id or sym in {"O", "N"}:
                    continue
                backbone.add(nb.GetIdx())
    return backbone


def choose_cross_modality_pair(
    rows: Sequence[MolRow],
    probs: Dict[str, float],
) -> Tuple[MolRow, MolRow]:
    peptide_candidates: List[Tuple[float, MolRow]] = []
    small_candidates: List[Tuple[float, MolRow]] = []

    amide_patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")

    for r in rows:
        if r.label != 1:
            continue
        amide_count = len(r.mol.GetSubstructMatches(amide_patt)) if amide_patt is not None else 0
        heavy = r.mol.GetNumHeavyAtoms()
        prob = probs.get(r.smiles, 0.0)

        if amide_count >= 2 and heavy >= 20:
            peptide_candidates.append((prob + 0.05 * amide_count, r))
        if amide_count <= 1 and heavy <= 35:
            small_candidates.append((prob, r))

    if not peptide_candidates or not small_candidates:
        raise RuntimeError("Cannot find suitable BBB+ peptide/small-molecule candidates.")

    peptide_candidates.sort(reverse=True, key=lambda x: x[0])
    small_candidates.sort(reverse=True, key=lambda x: x[0])
    return small_candidates[0][1], peptide_candidates[0][1]


def run_cross_modality(
    rows: Sequence[MolRow],
    probs: Dict[str, float],
    ckpt_path: str,
    device: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    small, peptide = choose_cross_modality_pair(rows, probs)

    s_scores, s_p1, s_pred = collect_attention_for_smiles(small.smiles, ckpt_path, device)
    p_scores, p_p1, p_pred = collect_attention_for_smiles(peptide.smiles, ckpt_path, device)

    p_backbone = peptide_backbone_atoms(peptide.mol)
    atom_tags = {i: ("backbone" if i in p_backbone else "sidechain") for i in range(peptide.mol.GetNumAtoms())}

    s_img = render_attention_mol(small.smiles, s_scores, f"Small Molecule | p(BBB+)={s_p1:.3f} | pred={s_pred}")
    p_img = render_attention_mol(peptide.smiles, p_scores, f"Peptide | p(BBB+)={p_p1:.3f} | pred={p_pred}", atom_emphasis=atom_tags)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].imshow(s_img)
    axes[0].axis("off")
    axes[1].imshow(p_img)
    axes[1].axis("off")
    fig.suptitle("Cross-modality: BBB+ small molecule vs BBB+ peptide", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_modality_attention.png"), dpi=220)
    plt.close(fig)

    peptide_scores = np.asarray(p_scores, dtype=np.float32)
    if peptide_scores.max() - peptide_scores.min() < 1e-9:
        p_norm = np.zeros_like(peptide_scores)
    else:
        p_norm = (peptide_scores - peptide_scores.min()) / (peptide_scores.max() - peptide_scores.min() + 1e-12)

    atom_rows = []
    for atom in peptide.mol.GetAtoms():
        i = atom.GetIdx()
        atom_rows.append(
            {
                "atom_id": i,
                "symbol": atom.GetSymbol(),
                "attention_score": float(peptide_scores[i]),
                "attention_norm": float(p_norm[i]),
                "group": atom_tags[i],
            }
        )
    pd.DataFrame(atom_rows).to_csv(os.path.join(out_dir, "peptide_backbone_sidechain_attention.csv"), index=False)

    with open(os.path.join(out_dir, "cross_modality_selection.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "small_molecule_smiles": small.smiles,
                "small_molecule_prob_bbb_plus": s_p1,
                "peptide_smiles": peptide.smiles,
                "peptide_prob_bbb_plus": p_p1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="LiBP A/B/C interpretability suite")
    parser.add_argument("--input_csv", default="/home/shenxin/EviMSGT/dataset/benchmark_0.csv")
    parser.add_argument("--smiles_col", default="", help="Optional smiles column name")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--ckpt", default="/home/shenxin/EviMSGT/ckpt/02best_model_epoch154_acc0.8693.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="/home/shenxin/EviMSGT/results/interpretability_suite")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_pairs", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    rows = parse_dataset(args.input_csv, args.smiles_col or None, args.label_col)
    if len(rows) < 20:
        raise RuntimeError(f"Too few valid molecules in dataset: {len(rows)}")

    os.makedirs(args.out_dir, exist_ok=True)

    probs = model_predict_probabilities(rows, args.ckpt, args.device)

    a_dir = os.path.join(args.out_dir, "A_global_shap")
    b_dir = os.path.join(args.out_dir, "B_local_case_studies")
    c_dir = os.path.join(args.out_dir, "C_cross_modality")

    ranking_csv = run_global_shap(rows, probs, a_dir, top_k=args.top_k)
    run_local_case_studies(rows, args.ckpt, args.device, b_dir, n_pairs=args.num_pairs)
    run_cross_modality(rows, probs, args.ckpt, args.device, c_dir)

    manifest = {
        "input_csv": args.input_csv,
        "checkpoint": args.ckpt,
        "num_valid_molecules": len(rows),
        "outputs": {
            "A_global_shap_ranking_csv": ranking_csv,
            "B_local_case_dir": b_dir,
            "C_cross_modality_dir": c_dir,
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done interpretability suite.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
