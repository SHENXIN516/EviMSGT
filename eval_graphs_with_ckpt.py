#!/usr/bin/env python3
# generate_graphs_pt.py
"""
生成 graphs_from_smiles.pt（以及一个 clean 版本）。
用法示例：
python generate_graphs_pt.py --csv /home/shenxin/EviMSGT/dataset/SMILES.csv \
    --out /home/shenxin/EviMSGT/dataset/graphs_from_smiles.pt \
    --out-clean /home/shenxin/EviMSGT/dataset/graphs_from_smiles_clean.pt
"""

import os, argparse, time
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
           one_of_k_encoding_unk(atom.GetHybridization(), [
               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)

def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = np.array([calc_atom_features(a) for a in mol.GetAtoms()])

    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond)
        row += [a, b]
        col += [b, a]
        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    if len(row) == 0:
        # no bonds (single-atom mol?) - create empty tensors with correct shape
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = None
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    return data

# ---------- main: 读取 CSV -> 生成 graphs -> 保存两个版本 ----------

def build_graphs_from_csv(csv_path, type_col="type", seq_col="sequence", label_col="label",
                          require_type_smiles=True, filter_invalid=True):
    df = pd.read_csv(csv_path)
    if require_type_smiles and (type_col in df.columns):
        try:
            df_filtered = df[df[type_col] == "SMILES"].reset_index(drop=True)
            if len(df_filtered) == 0:
                df_filtered = df.reset_index(drop=True)
        except Exception:
            df_filtered = df.reset_index(drop=True)
    else:
        df_filtered = df.reset_index(drop=True)

    smiles_list = df_filtered[seq_col].astype(str).tolist() if seq_col in df_filtered.columns else []
    labels_list = df_filtered[label_col].tolist() if label_col in df_filtered.columns else [-1]*len(smiles_list)
    orig_idx = df_filtered.index.to_list()

    graphs = []
    valid_smiles = []
    valid_labels = []
    valid_orig_idx = []

    start = time.time()
    for i, (smi, lab, idx) in enumerate(zip(smiles_list, labels_list, orig_idx)):
        g = mol_to_graph(smi)
        if g is None:
            if filter_invalid:
                # skip invalid SMILES
                continue
            else:
                # create empty placeholder
                continue
        graphs.append(g)
        valid_smiles.append(smi)
        valid_labels.append(int(lab) if not pd.isna(lab) else -1)
        valid_orig_idx.append(int(idx))
        if (i+1) % 200 == 0:
            print(f"[{i+1}] processed, valid graphs so far: {len(graphs)}")
    elapsed = time.time() - start
    print(f"Done. Parsed {len(graphs)} graphs in {elapsed:.1f}s (from {len(smiles_list)} SMILES).")
    return valid_smiles, valid_labels, graphs, valid_orig_idx

def save_full_pt(out_path, smiles, labels, graphs, orig_idx):
    # 保存包含 torch_geometric.Data 对象（与训练时相同）
    obj = {"smiles": smiles, "labels": labels, "graphs": graphs, "orig_idx": orig_idx}
    torch.save(obj, out_path)
    print("Saved full pt to", out_path)

def save_clean_pt(out_path, smiles, labels, graphs, orig_idx):
    # 把每个 Data 解包成纯 tensors / lists，避免 torch.load 的 safe globals 问题
    clean_graphs = []
    for g in graphs:
        gdict = {
            "x": g.x.detach().cpu(), 
            "edge_index": g.edge_index.detach().cpu() if hasattr(g, "edge_index") else None,
            "edge_attr": g.edge_attr.detach().cpu() if hasattr(g, "edge_attr") and g.edge_attr is not None else None,
            "y": getattr(g, "y", None)
        }
        # make sure y is tensor or None
        if gdict["y"] is not None and torch.is_tensor(gdict["y"]):
            gdict["y"] = gdict["y"].detach().cpu()
        clean_graphs.append(gdict)
    out = {"smiles": smiles, "labels": labels, "graphs": clean_graphs, "orig_idx": orig_idx}
    torch.save(out, out_path)
    print("Saved clean pt to", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True, help="output full pt (contains Data objects)")
    p.add_argument("--out-clean", required=False, help="output clean pt (pure tensors/lists)")
    p.add_argument("--type-col", default="type")
    p.add_argument("--seq-col", default="sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--no-filter-invalid", action="store_true", help="do not drop invalid SMILES")
    args = p.parse_args()

    smiles, labels, graphs, orig_idx = build_graphs_from_csv(
        args.csv, type_col=args.type_col, seq_col=args.seq_col, label_col=args.label_col,
        require_type_smiles=True, filter_invalid=(not args.no_filter_invalid)
    )

    save_full_pt(args.out, smiles, labels, graphs, orig_idx)
    if args.out_clean:
        save_clean_pt(args.out_clean, smiles, labels, graphs, orig_idx)

if __name__ == "__main__":
    import argparse
    main()
