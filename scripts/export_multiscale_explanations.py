import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_5split_ensemble_grid import load_model_from_ckpt
from scripts.train_plat import BBBP_Dataset


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def load_model(ckpt_path: str, device):
    model, _ = load_model_from_ckpt(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return model, ckpt


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


def atom_scores_from_attention(atom_attentions, num_atoms: int) -> np.ndarray:
    if not atom_attentions:
        return np.zeros(num_atoms, dtype=np.float32)
    scores = np.zeros(num_atoms, dtype=np.float32)
    used = 0
    for attn in atom_attentions:
        if attn is None:
            continue
        edge_index = attn["edge_index"].detach().cpu().numpy()
        edge_attention = attn["edge_attention"].detach().cpu().numpy()
        edge_score = edge_attention.mean(axis=1)
        for dst, score in zip(edge_index[1], edge_score):
            if 0 <= int(dst) < num_atoms:
                scores[int(dst)] += float(score)
        used += 1
    if used > 0 and scores.max() > 0:
        scores = scores / scores.max()
    return scores


def residue_scores_from_attention(residue_attentions, num_residues: int) -> np.ndarray:
    if not residue_attentions:
        return np.zeros(num_residues, dtype=np.float32)
    scores = np.zeros(num_residues, dtype=np.float32)
    used = 0
    for attn in residue_attentions:
        arr = attn.detach().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            residue_score = arr.mean(axis=0)
        elif arr.ndim == 3:
            residue_score = arr.mean(axis=(0, 1))
        else:
            continue
        n = min(num_residues, residue_score.shape[-1])
        scores[:n] += residue_score[:n]
        used += 1
    if used > 0 and scores.max() > 0:
        scores = scores / scores.max()
    return scores


def residue_edge_scores(residue_attentions, top_k: int) -> List[Dict[str, object]]:
    if not residue_attentions:
        return []
    arr = residue_attentions[-1].detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        mat = arr.mean(axis=0)
    elif arr.ndim == 2:
        mat = arr
    else:
        return []
    rows = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                continue
            rows.append({"residue_i": i, "residue_j": j, "score": float(mat[i, j])})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


def safe_mol_from_sequence(sequence: str):
    seq = "".join(str(sequence).split()).upper()
    try:
        return Chem.MolFromFASTA(seq)
    except Exception:
        return None


def draw_atom_highlight(sequence: str, atom_scores: np.ndarray, out_path: Path, top_k: int):
    mol = safe_mol_from_sequence(sequence)
    if mol is None:
        return
    top_atoms = np.argsort(-atom_scores)[: min(top_k, len(atom_scores), mol.GetNumAtoms())]
    top_atoms = [int(i) for i in top_atoms if float(atom_scores[i]) > 0]
    if not top_atoms:
        return
    drawer = rdMolDraw2D.MolDraw2DSVG(900, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=top_atoms)
    drawer.FinishDrawing()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def draw_residue_edge_svg(num_residues: int, residue_scores: np.ndarray, edges: List[Dict[str, object]], out_path: Path):
    if num_residues <= 0:
        return
    width = max(640, min(1600, 80 + num_residues * 34))
    height = 260
    margin = 40
    y = 185
    denom = max(1, num_residues - 1)
    xs = [margin + (width - 2 * margin) * i / denom for i in range(num_residues)]
    top_residues = set(int(i) for i in np.argsort(-residue_scores)[: min(10, len(residue_scores))] if residue_scores[i] > 0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="Arial" font-size="16" fill="#1f2937">Top residue connections</text>',
    ]
    for edge in edges:
        i = int(edge["residue_i"])
        j = int(edge["residue_j"])
        if i < 0 or j < 0 or i >= num_residues or j >= num_residues:
            continue
        score = max(0.0, min(1.0, float(edge["score"])))
        x1, x2 = xs[i], xs[j]
        mid = (x1 + x2) / 2
        arc_height = 30 + min(95, abs(j - i) * 8)
        stroke_width = 1.0 + 4.0 * score
        lines.append(
            f'<path d="M{x1:.1f},{y:.1f} Q{mid:.1f},{y - arc_height:.1f} {x2:.1f},{y:.1f}" '
            f'fill="none" stroke="#2563eb" stroke-opacity="{0.25 + 0.55 * score:.3f}" stroke-width="{stroke_width:.2f}"/>'
        )
    for i, x in enumerate(xs):
        score = float(residue_scores[i]) if i < len(residue_scores) else 0.0
        radius = 6 + 8 * max(0.0, min(1.0, score))
        fill = "#dc2626" if i in top_residues else "#9ca3af"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity="0.85"/>')
        if num_residues <= 60 or i in top_residues:
            lines.append(f'<text x="{x:.1f}" y="{y + 28:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">{i}</text>')
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def safe_filename(text: object) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))
    return out[:160] if out else "sample"


def explain_samples(
    ckpt_path: str,
    csv_path: str,
    out_dir: Path,
    mapping_mode: str,
    split: Optional[str],
    max_samples: int,
    top_k: int,
    device,
):
    model, ckpt = load_model(ckpt_path, device)
    threshold = float(ckpt.get("best_threshold", 0.5))
    dataset = BBBP_Dataset(csv_path, split_col="split" if split else None, label_col="label", mapping_mode=mapping_mode)
    raw_rows = read_csv(Path(csv_path))
    aligned_rows = align_dataset_rows(dataset, raw_rows)

    sample_rows = []
    atom_rows = []
    residue_rows = []
    residue_edge_rows = []
    exported = 0
    for i in range(len(dataset)):
        if split and dataset.splits is not None and str(dataset.splits[i]).lower() != split:
            continue
        graph = dataset.get(i)
        raw = aligned_rows[i] if i < len(aligned_rows) else {}
        data = graph.to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(data, return_attention=True)
        logits, attn = output
        if isinstance(logits, dict):
            prob_pos = float(logits["probs"][:, 1].detach().cpu().item())
        else:
            prob_pos = float(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().item())
        pred = int(prob_pos >= threshold)
        label = int(graph.y.item())
        atom_to_residue = attn["atom_to_residue"].detach().cpu().numpy().astype(int)
        num_atoms = int(graph.x.size(0))
        num_residues = int(atom_to_residue.max()) + 1 if atom_to_residue.size else 0
        atom_scores = atom_scores_from_attention(attn.get("atom_attentions"), num_atoms)
        residue_scores = residue_scores_from_attention(attn.get("residue_attentions"), num_residues)

        sample_id = raw.get("sample_id", raw.get("id", str(i)))
        sequence = raw.get("sequence", dataset.smiles[i] if i < len(dataset.smiles) else "")
        sample_rows.append(
            {
                "sample_index": i,
                "sample_id": sample_id,
                "orig_id": raw.get("orig_id", ""),
                "split": raw.get("split", split or ""),
                "label": label,
                "prob_pos": prob_pos,
                "pred": pred,
                "error_type": "correct" if pred == label else ("fp" if pred == 1 else "fn"),
                "sequence_len": len(str(sequence).strip()),
                "sequence": sequence,
            }
        )
        for atom_idx in np.argsort(-atom_scores)[:top_k]:
            atom_rows.append(
                {
                    "sample_id": sample_id,
                    "atom_idx": int(atom_idx),
                    "residue_idx": int(atom_to_residue[atom_idx]) if atom_idx < len(atom_to_residue) else "",
                    "score": float(atom_scores[atom_idx]),
                }
            )
        for residue_idx in np.argsort(-residue_scores)[:top_k]:
            residue_rows.append(
                {
                    "sample_id": sample_id,
                    "residue_idx": int(residue_idx),
                    "score": float(residue_scores[residue_idx]),
                    "atom_count": int((atom_to_residue == residue_idx).sum()) if atom_to_residue.size else 0,
                }
            )
        sample_edges = residue_edge_scores(attn.get("residue_attentions"), top_k)
        for edge in sample_edges:
            edge["sample_id"] = sample_id
            residue_edge_rows.append(edge)
        draw_atom_highlight(str(sequence), atom_scores, out_dir / "svg" / f"{safe_filename(sample_id)}.svg", top_k=top_k)
        draw_residue_edge_svg(
            num_residues,
            residue_scores,
            sample_edges,
            out_dir / "residue_edges" / f"{safe_filename(sample_id)}.svg",
        )

        exported += 1
        if exported >= max_samples:
            break

    write_csv(out_dir / "sample_predictions.csv", sample_rows, ["sample_index", "sample_id", "orig_id", "split", "label", "prob_pos", "pred", "error_type", "sequence_len", "sequence"])
    write_csv(out_dir / "top_atoms.csv", atom_rows, ["sample_id", "atom_idx", "residue_idx", "score"])
    write_csv(out_dir / "top_residues.csv", residue_rows, ["sample_id", "residue_idx", "score", "atom_count"])
    write_csv(out_dir / "top_residue_edges.csv", residue_edge_rows, ["sample_id", "residue_i", "residue_j", "score"])
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"ckpt_path": ckpt_path, "csv_path": csv_path, "split": split, "threshold": threshold}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Export minimal multiscale explanation reports for selected samples.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--mapping_mode", default="helm_force")
    parser.add_argument("--split", default="test", help="Filter split if csv has split column; use empty string for no filter.")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_dir", default="results/explanations")
    args = parser.parse_args()

    split = str(args.split).strip().lower() or None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explain_samples(
        ckpt_path=args.ckpt_path,
        csv_path=args.csv_path,
        out_dir=Path(args.out_dir),
        mapping_mode=args.mapping_mode,
        split=split,
        max_samples=args.max_samples,
        top_k=args.top_k,
        device=device,
    )


if __name__ == "__main__":
    main()
