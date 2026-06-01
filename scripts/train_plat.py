import os
import sys
import re
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from sklearn.model_selection import train_test_split  # 导入划分训练集和测试集的工具
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('/home/shenxin/EviMSGT')
from plat_model.model import SubGT, GraphTransformer, MultiScaleGraphTransformer  # 你自己的 SubGT 文件路径


NODE_FEATURE_NAMES = [
    "atom_type_onehot",
    "degree_onehot",
    "formal_charge",
    "num_radical_electrons",
    "hybridization_onehot",
    "aromatic",
    "num_hydrogens_onehot",
]

EDGE_FEATURE_NAMES = [
    "bond_type_single",
    "bond_type_double",
    "bond_type_triple",
    "bond_type_aromatic",
    "conjugation",
    "ring",
]

AA_TO_INDEX = {
    aa: i
    for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")
}


def build_model_config(model):
    model_class = model.__class__.__name__
    model_name_map = {
        "MultiScaleGraphTransformer": "multiscale",
        "GraphTransformer": "graphtransformer",
        "SubGT": "subgt",
    }
    hidden_dim = int(model.node_encoder.out_features)
    num_heads = int(getattr(model, "num_attention_heads", 1))
    config = {
        "model_name": model_name_map.get(model_class, model_class),
        "model_class": model_class,
        "num_layers": int(getattr(model, "num_layers", len(getattr(model, "gt_block", [])))),
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": hidden_dim // num_heads if num_heads > 0 else hidden_dim,
        "dropout": float(getattr(model, "dropout_rate", 0.0)),
        "edge_dim": int(model.edge_encoder.in_features),
        "node_dim": int(model.node_encoder.in_features),
        "activation": model.activ_fn.__class__.__name__ if hasattr(model, "activ_fn") else "Unknown",
        "readout": "global_mean_pool + MLP",
        # Keep old keys for backward compatibility with existing eval scripts.
        "in_channels": int(model.node_encoder.in_features),
        "edge_features": int(model.edge_encoder.in_features),
        "num_hidden_channels": hidden_dim,
        "num_attention_heads": num_heads,
        "dropout_rate": float(getattr(model, "dropout_rate", 0.0)),
        "norm_to_apply": str(getattr(model, "norm_to_apply", "batch")),
    }
    if hasattr(model, "num_residue_layers"):
        config["num_residue_layers"] = int(getattr(model, "num_residue_layers"))
    config["use_evidential"] = bool(getattr(model, "use_evidential", False))
    config["ablation_mode"] = str(getattr(model, "ablation_mode", "full"))
    config["readout_mode"] = str(getattr(model, "readout_mode", "mean_max"))
    config["use_residue_position"] = bool(getattr(model, "use_residue_position", False))
    config["use_terminal_flags"] = bool(getattr(model, "use_terminal_flags", False))
    config["use_physchem_features"] = bool(getattr(model, "use_physchem_features", False))
    return config


def build_feature_config(use_chirality=False, explicit_h=False):
    return {
        "node_features": NODE_FEATURE_NAMES,
        "edge_features": EDGE_FEATURE_NAMES,
        "featurizer": "RDKit",
        "smiles_processing": "canonical (RDKit default) + stereochemistry preserved",
        "use_chirality": bool(use_chirality),
        "explicit_H": bool(explicit_h),
    }


def build_training_config(optimizer, scheduler, batch_size, epochs, criterion):
    return {
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "loss_function": criterion.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "scheduler_mode": getattr(scheduler, "mode", None),
        "scheduler_factor": float(getattr(scheduler, "factor", 1.0)),
        "scheduler_patience": int(getattr(scheduler, "patience", 0)),
        "early_stopping": False,
    }


def _kl_dirichlet_to_uniform(alpha):
    num_classes = alpha.size(1)
    beta = torch.ones((1, num_classes), device=alpha.device, dtype=alpha.dtype)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b_alpha = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(s_alpha)
    kl = torch.sum((alpha - beta) * digamma_diff, dim=1, keepdim=True) + ln_b_alpha + ln_b_beta
    return kl.squeeze(1)


def evidential_loss_from_logits(
    logits,
    target,
    epoch_idx,
    anneal_epochs=10,
    kl_weight=1e-3,
    num_classes=2,
    class_weights=None,
):
    target = target.long().view(-1)
    one_hot = F.one_hot(target, num_classes=num_classes).float()

    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    s = torch.sum(alpha, dim=1, keepdim=True)

    # Expected cross-entropy under Dirichlet predictive distribution.
    nll = torch.sum(one_hot * (torch.digamma(s) - torch.digamma(alpha)), dim=1)

    # Do not regularize evidence for true class too aggressively.
    alpha_tilde = (alpha - 1.0) * (1.0 - one_hot) + 1.0
    kl = _kl_dirichlet_to_uniform(alpha_tilde)

    anneal = min(1.0, float(epoch_idx + 1) / float(max(1, anneal_epochs)))
    loss = nll + anneal * kl_weight * kl
    if class_weights is not None:
        weights = class_weights[target].view(-1).to(loss.dtype)
        denom = torch.clamp(weights.sum(), min=1.0)
        return (loss * weights).sum() / denom
    return loss.mean()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
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


def parse_residue_count(sequence=None, num_monomers=None):
    if num_monomers is not None and not pd.isna(num_monomers):
        try:
            n = int(num_monomers)
            if n > 1:
                return n
        except Exception:
            pass

    if sequence is None or pd.isna(sequence):
        sequence_count = None
    else:
        seq = str(sequence).strip()
        # cycpep sequence is tokenized as "A.B.[dP].X". For plain SMILES we skip residue parsing.
        if "." not in seq:
            # FASTA-like peptide sequence: use residue length as residue count.
            fasta_seq = re.sub(r"\s+", "", seq).upper()
            if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", fasta_seq) and len(fasta_seq) > 1:
                sequence_count = len(fasta_seq)
            else:
                sequence_count = None
        else:
            tokens = [t for t in seq.split(".") if t]
            sequence_count = len(tokens) if len(tokens) > 1 else None

    return sequence_count


def parse_residue_count_from_helm(helm):
    if helm is None or pd.isna(helm):
        return None
    text = str(helm)
    # Typical HELM fragment: PEPTIDE1{[A].[B].C}...
    m = re.search(r"\{([^}]*)\}", text)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return None
    tokens = [t.strip() for t in body.split(".") if t.strip()]
    return len(tokens) if len(tokens) > 1 else None


def helm_has_cycle(helm):
    if helm is None or pd.isna(helm):
        return False
    return re.search(r"\d+:R\d+-\d+:R\d+", str(helm)) is not None


def build_atom_residue_index_from_peptide(mol, residue_count, cyclic, mapping_mode="helm_force"):
    num_atoms = mol.GetNumAtoms()
    mode = str(mapping_mode).lower().strip()
    if mode not in {"strict", "hybrid", "helm_force"}:
        mode = "helm_force"

    if residue_count is None or residue_count <= 1:
        return torch.arange(num_atoms, dtype=torch.long)

    # Candidate inter-residue cuts are peptide-like linkages around carbonyl carbon.
    # We intentionally include amide/ester/thioester to better cover non-canonical residues.
    linkage_patterns = [
        Chem.MolFromSmarts("[CX3](=[OX1])[NX2,NX3,NX4+]"),
        Chem.MolFromSmarts("[CX3](=[OX1])[OX2;!H0]"),
        Chem.MolFromSmarts("[CX3](=[OX1])[SX2]"),
    ]

    candidate_bonds = []
    for patt in linkage_patterns:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for m in matches:
            if len(m) < 3:
                continue
            c_idx = int(m[0])
            x_idx = int(m[2])
            bond = mol.GetBondBetweenAtoms(c_idx, x_idx)
            if bond is None:
                continue

            c_atom = mol.GetAtomWithIdx(c_idx)
            x_atom = mol.GetAtomWithIdx(x_idx)

            # Heuristic filters:
            # 1) linkage atom should not be terminal H-bearing oxygen (free acid/alcohol)
            # 2) carbonyl carbon should connect to a carbon skeleton (avoid tiny side fragments)
            if x_atom.GetAtomicNum() == 8 and x_atom.GetDegree() < 2:
                continue

            carbon_neighbors = [n for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 6 and n.GetIdx() != x_idx]
            if len(carbon_neighbors) == 0:
                continue

            a, b = sorted((c_idx, x_idx))
            if (a, b) not in candidate_bonds:
                candidate_bonds.append((a, b))

    if not candidate_bonds:
        return torch.arange(num_atoms, dtype=torch.long)

    target_cuts = residue_count if cyclic else (residue_count - 1)
    target_cuts = max(1, min(target_cuts, len(candidate_bonds)))

    base_adj = [[] for _ in range(num_atoms)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        base_adj[i].append(j)
        base_adj[j].append(i)

    def components_after_cut(cut_set):
        visited = [False] * num_atoms
        comps = []
        for start in range(num_atoms):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in base_adj[u]:
                    e = (u, v) if u < v else (v, u)
                    if e in cut_set:
                        continue
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(sorted(comp))
        return comps

    def is_carbonyl_carbon(atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 6:
            return False
        for b in atom.GetBonds():
            other = b.GetOtherAtom(atom)
            if b.GetBondType() == Chem.rdchem.BondType.DOUBLE and other.GetAtomicNum() == 8:
                return True
        return False

    carbonyl_flags = [is_carbonyl_carbon(i) for i in range(num_atoms)]

    def bond_priority(i, j):
        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)
        zi, zj = ai.GetAtomicNum(), aj.GetAtomicNum()
        c_i, c_j = carbonyl_flags[i], carbonyl_flags[j]
        if (c_i and zj in (7, 8, 16)) or (c_j and zi in (7, 8, 16)):
            return 5
        if c_i or c_j:
            return 4
        if (zi in (7, 8, 16) and zj == 6) or (zj in (7, 8, 16) and zi == 6):
            return 3
        if zi == 6 and zj == 6:
            return 2
        return 1

    def fallback_cut_to_target(initial_comps):
        if residue_count is None:
            return initial_comps
        current_cut = set()
        current_comps = initial_comps
        current_n = len(current_comps)
        if current_n >= residue_count:
            return current_comps

        # Use single edges as soft cut candidates in fallback.
        # Ring edges are allowed but down-weighted.
        fallback_edges = []
        for b in mol.GetBonds():
            if b.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            a, c = sorted((i, j))
            fallback_edges.append((a, c, bool(b.IsInRing())))

        safety_iters = len(fallback_edges)
        for _ in range(safety_iters):
            if current_n >= residue_count:
                break
            best = None
            best_key = None
            prep = None
            prep_key = None
            for e0, e1, is_ring in fallback_edges:
                e = (e0, e1)
                if e in current_cut:
                    continue
                trial_cut = current_cut | {e}
                trial_comps = components_after_cut(trial_cut)
                trial_n = len(trial_comps)
                gain = trial_n - current_n
                if gain <= 0:
                    continue

                min_size = min(len(c) for c in trial_comps) if trial_comps else 0
                pri = bond_priority(e[0], e[1])
                ring_bonus = 0 if is_ring else 1
                # Prefer increasing component count, then backbone-like edges, then avoid tiny fragments.
                key = (gain, pri, ring_bonus, min_size)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (e, trial_comps, trial_n)

                # Store best preparatory cut for cyclic cases where one cut may not disconnect yet.
                prep_candidate_key = (pri, ring_bonus)
                if prep_key is None or prep_candidate_key > prep_key:
                    prep_key = prep_candidate_key
                    prep = (e, trial_comps, trial_n)

            if best is not None:
                e, current_comps, current_n = best
                current_cut.add(e)
                continue

            # No immediate gain: still cut the most likely backbone edge to break cycles progressively.
            if prep is None:
                break
            e, current_comps, current_n = prep
            current_cut.add(e)

        return current_comps

    def spanning_tree_force_partition(target_parts):
        if target_parts is None or target_parts <= 1:
            return [list(range(num_atoms))]

        # Build a DFS spanning tree.
        tree_adj = [[] for _ in range(num_atoms)]
        tree_edges = []
        seen = [False] * num_atoms
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            atom_u = mol.GetAtomWithIdx(u)
            for b in atom_u.GetBonds():
                v = b.GetOtherAtomIdx(u)
                if seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
                tree_adj[u].append(v)
                tree_adj[v].append(u)
                tree_edges.append((u, v))

        if len(tree_edges) == 0:
            return [list(range(num_atoms))]

        # Rank tree edges by likely inter-residue nature.
        scored = []
        for u, v in tree_edges:
            b = mol.GetBondBetweenAtoms(u, v)
            is_ring = bool(b.IsInRing()) if b is not None else False
            is_single = bool(b.GetBondType() == Chem.rdchem.BondType.SINGLE) if b is not None else False
            pri = bond_priority(u, v)
            key = (pri, 1 if is_single else 0, 0 if is_ring else 1)
            scored.append((key, (u, v)))
        scored.sort(reverse=True)

        num_cuts = min(target_parts - 1, len(scored))
        cut_set = set()
        for _, e in scored[:num_cuts]:
            a, b = sorted(e)
            cut_set.add((a, b))

        # Components on the spanning tree after cuts.
        visited = [False] * num_atoms
        comps = []
        for s in range(num_atoms):
            if visited[s]:
                continue
            q = [s]
            visited[s] = True
            comp = []
            while q:
                x = q.pop()
                comp.append(x)
                for y in tree_adj[x]:
                    e = (x, y) if x < y else (y, x)
                    if e in cut_set:
                        continue
                    if not visited[y]:
                        visited[y] = True
                        q.append(y)
            comps.append(sorted(comp))
        return comps

    best_comps = None
    best_score = None

    # Exact combinational search for typical peptide size; fallback to greedy when large.
    max_enumeration = 20000
    n = len(candidate_bonds)
    total_comb = 1
    for i in range(target_cuts):
        total_comb = total_comb * (n - i) // (i + 1)

    if total_comb <= max_enumeration:
        iterator = itertools.combinations(candidate_bonds, target_cuts)
    else:
        # Greedy fallback: iteratively choose bonds that increase component count most.
        selected = []
        remaining = list(candidate_bonds)
        for _ in range(target_cuts):
            current_cut = set(selected)
            current_comps = components_after_cut(current_cut)
            current_n = len(current_comps)
            best_local = None
            best_local_gain = -1
            for e in remaining:
                new_comps = components_after_cut(current_cut | {e})
                gain = len(new_comps) - current_n
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_local = e
            if best_local is None:
                break
            selected.append(best_local)
            remaining.remove(best_local)
        iterator = [tuple(selected)]

    for cut_tuple in iterator:
        cut_set = set(cut_tuple)
        comps = components_after_cut(cut_set)
        comp_sizes = [len(c) for c in comps]
        # Score priority: component count close to residue_count, then avoid tiny fragments.
        score = (
            abs(len(comps) - residue_count),
            -min(comp_sizes) if comp_sizes else 0,
            np.std(comp_sizes) if comp_sizes else 0.0,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_comps = comps

    if not best_comps:
        return torch.arange(num_atoms, dtype=torch.long)

    if residue_count is not None and len(best_comps) != residue_count:
        if mode in {"hybrid", "helm_force"}:
            best_comps = fallback_cut_to_target(best_comps)
        if mode == "helm_force" and len(best_comps) != residue_count:
            best_comps = spanning_tree_force_partition(residue_count)

    # Assign residue ids by sorted component order (stable and deterministic).
    best_comps = sorted(best_comps, key=lambda c: c[0])
    atom_residue = torch.zeros(num_atoms, dtype=torch.long)
    for rid, comp in enumerate(best_comps):
        atom_residue[torch.tensor(comp, dtype=torch.long)] = rid
    return atom_residue


def build_atom_residue_index_from_atom_metadata(mol):
    """Fast path for FASTA-derived molecules using RDKit atom residue metadata.

    Returns:
        torch.LongTensor[num_atoms] if metadata exists and is valid, else None.
    """
    residue_ids = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None:
            residue_ids.append(int(info.GetResidueNumber()))
            continue

        monomer = atom.GetMonomerInfo()
        if monomer is None:
            return None
        try:
            residue_ids.append(int(monomer.GetResidueNumber()))
        except Exception:
            return None

    if len(residue_ids) != mol.GetNumAtoms() or len(set(residue_ids)) <= 1:
        return None

    # Re-index residue ids to contiguous 0..R-1 per molecule.
    unique_sorted = sorted(set(residue_ids))
    rid_map = {rid: i for i, rid in enumerate(unique_sorted)}
    return torch.tensor([rid_map[rid] for rid in residue_ids], dtype=torch.long)


def build_atom_positions(mol):
    """Generate per-atom coordinates with robust fallbacks.

    Priority:
    1) RDKit 3D embedding (ETKDG + optional UFF optimization)
    2) RDKit 2D coordinates
    3) all-zero coordinates
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return torch.zeros((0, 3), dtype=torch.float)

    # Control coordinate generation mode via env for stability/speed.
    # auto (default): try 3D for smaller molecules, otherwise 2D.
    # 3d: always try 3D then fallback.
    # 2d: skip 3D and use 2D coords.
    # zero: always return zeros.
    pos_mode = os.getenv("EVIMSGT_POS_MODE", "auto").strip().lower()
    if pos_mode == "zero":
        return torch.zeros((num_atoms, 3), dtype=torch.float)

    # 3D attempt on a temporary H-added molecule, then remove Hs back.
    # Macrocycles/large peptides can make 3D embedding very slow, so auto mode
    # uses 3D only under a conservative atom-count threshold.
    use_3d = pos_mode == "3d" or (pos_mode == "auto" and num_atoms <= 96)
    try:
        if use_3d:
            mol_h = Chem.AddHs(Chem.Mol(mol))
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            params.maxAttempts = 5
            status = AllChem.EmbedMolecule(mol_h, params)
            if status == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
                except Exception:
                    pass
                mol_3d = Chem.RemoveHs(mol_h)
                if mol_3d.GetNumAtoms() == num_atoms and mol_3d.GetNumConformers() > 0:
                    conf = mol_3d.GetConformer()
                    coords = np.array([
                        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in range(num_atoms)
                    ], dtype=np.float32)
                    return torch.tensor(coords, dtype=torch.float)
    except Exception:
        pass

    # 2D fallback.
    try:
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        if mol_2d.GetNumConformers() > 0:
            conf = mol_2d.GetConformer()
            coords = np.array([
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, 0.0]
                for i in range(num_atoms)
            ], dtype=np.float32)
            return torch.tensor(coords, dtype=torch.float)
    except Exception:
        pass

    return torch.zeros((num_atoms, 3), dtype=torch.float)


def mol_to_graph(smiles, sequence=None, helm=None, num_monomers=None, mapping_mode="helm_force"):
    seq_like = re.sub(r"\s+", "", str(smiles).strip()).upper()
    sequence_fasta = None
    if sequence is not None and not pd.isna(sequence):
        candidate = re.sub(r"\s+", "", str(sequence).strip()).upper()
        if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", candidate):
            sequence_fasta = candidate

    smiles_fasta = seq_like if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", seq_like) else None

    # Prefer FASTA parsing first when sequence-like input is detected, to avoid RDKit SMILES parse spam.
    if sequence_fasta is not None:
        try:
            mol = Chem.MolFromFASTA(sequence_fasta)
        except Exception:
            mol = None
    elif smiles_fasta is not None:
        try:
            mol = Chem.MolFromFASTA(smiles_fasta)
        except Exception:
            mol = None
    else:
        mol = Chem.MolFromSmiles(smiles)

    if mol is None and sequence_fasta is not None:
        try:
            mol = Chem.MolFromFASTA(sequence_fasta)
        except Exception:
            mol = None

    if mol is None and smiles_fasta is not None:
        try:
            mol = Chem.MolFromFASTA(smiles_fasta)
        except Exception:
            mol = None

    if mol is None:
        return None

    # -------------------------
    # 节点特征
    # -------------------------
    mol = Chem.RemoveHs(mol)
    x = np.array([calc_atom_features(a) for a in mol.GetAtoms()])

    # -------------------------
    # 边特征
    # -------------------------
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        bond_feats = calc_bond_features(bond)

        row += [a, b]
        col += [b, a]

        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    # Converting list-of-numpy-arrays directly is slow; stack first.
    edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)

    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    data.atom_z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    residue_count = parse_residue_count(sequence=sequence, num_monomers=num_monomers)
    if residue_count is None:
        residue_count = parse_residue_count_from_helm(helm)
    cyclic = helm_has_cycle(helm)

    # FASTA-derived molecules expose residue ids per atom via RDKit metadata.
    # Using it avoids expensive combinational bond-cut search.
    atom_residue_fast = None
    if sequence_fasta is not None or smiles_fasta is not None:
        atom_residue_fast = build_atom_residue_index_from_atom_metadata(mol)

    if atom_residue_fast is not None:
        data.atom_residue_index = atom_residue_fast
    else:
        data.atom_residue_index = build_atom_residue_index_from_peptide(
            mol,
            residue_count=residue_count,
            cyclic=cyclic,
            mapping_mode=mapping_mode,
        )
    fasta_for_residue_features = sequence_fasta if sequence_fasta is not None else smiles_fasta
    if fasta_for_residue_features is not None:
        aa_ids = [AA_TO_INDEX.get(aa, 20) for aa in fasta_for_residue_features]
        data.residue_aa_index = torch.tensor(aa_ids, dtype=torch.long)
    data.pos = build_atom_positions(mol)
    return data


class BBBP_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        cache_dir="/home/shenxin/EviMSGT/dataset",
        split_col=None,
        label_col="label",
        permeability_col="permeability",
        permeability_threshold=None,
        mapping_mode="helm_force",
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.csv_path = csv_path
        self.split_col = split_col
        self.label_col = label_col
        self.permeability_col = permeability_col
        self.permeability_threshold = permeability_threshold
        self.mapping_mode = mapping_mode
        self.label_source = None

        base = os.path.splitext(os.path.basename(csv_path))[0]
        split_tag = split_col if split_col else "random"
        thr_tag = "none" if permeability_threshold is None else str(permeability_threshold).replace(".", "p")
        pos_tag = os.getenv("EVIMSGT_POS_MODE", "auto").strip().lower()
        cache_name = f"graphs_{base}_{split_tag}_{label_col}_{thr_tag}_{self.mapping_mode}_{pos_tag}_geomv2.pt"
        cache_file = os.path.join(self.cache_dir, cache_name)

        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            data = torch.load(cache_file,weights_only=False)
            self.smiles = data["smiles"]
            self.labels = data["labels"]
            self.graphs = data["graphs"]
            self.splits = data.get("splits", None)
            self.label_source = data.get("label_source", None)
            self.permeability_threshold = data.get("permeability_threshold", self.permeability_threshold)
        else:
            df = pd.read_csv(self.csv_path)
            if "type" in df.columns:
                df = df[df["type"] == "SMILES"]

            smiles_col = "sequence" if "type" in df.columns and "sequence" in df.columns else (
                "smiles" if "smiles" in df.columns else "sequence"
            )
            sequence_col = "sequence" if "sequence" in df.columns else None
            helm_col = "helm" if "helm" in df.columns else None
            num_monomers_col = "num_monomers" if "num_monomers" in df.columns else None
            split_values = df[split_col].astype(str).tolist() if split_col and split_col in df.columns else None

            smiles = df[smiles_col].astype(str).tolist()
            if label_col in df.columns:
                labels = df[label_col].astype(int).tolist()
                self.label_source = label_col
            elif permeability_col in df.columns:
                perm = pd.to_numeric(df[permeability_col], errors="coerce")
                thr = permeability_threshold
                if thr is None:
                    thr = float(perm.median())
                labels = (perm >= thr).astype(int).tolist()
                self.label_source = f"{permeability_col}>= {thr:.4f}"
                self.permeability_threshold = thr
            else:
                raise ValueError(f"No label column '{label_col}' or permeability column '{permeability_col}' found.")

            self.graphs = []
            self.labels = []
            self.smiles = []
            self.splits = [] if split_values is not None else None
            total_rows = len(smiles)
            for row_idx, (smi, label) in enumerate(zip(smiles, labels)):
                seq = df.iloc[row_idx][sequence_col] if sequence_col is not None else None
                helm = df.iloc[row_idx][helm_col] if helm_col is not None else None
                num_monomers = df.iloc[row_idx][num_monomers_col] if num_monomers_col is not None else None
                g = mol_to_graph(
                    smi,
                    sequence=seq,
                    helm=helm,
                    num_monomers=num_monomers,
                    mapping_mode=self.mapping_mode,
                )
                if g is not None:
                    self.graphs.append(g)
                    self.labels.append(label)
                    self.smiles.append(smi)
                    if self.splits is not None:
                        self.splits.append(str(split_values[row_idx]))

                if (row_idx + 1) % 200 == 0 or (row_idx + 1) == total_rows:
                    print(
                        f"Preprocessing graphs: {row_idx + 1}/{total_rows} "
                        f"(valid={len(self.graphs)})",
                        flush=True,
                    )

            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(
                {
                    "smiles": self.smiles,
                    "labels": self.labels,
                    "graphs": self.graphs,
                    "splits": self.splits,
                    "label_source": self.label_source,
                    "permeability_threshold": self.permeability_threshold,
                },
                cache_file,
            )

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return graph


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch_idx,
    use_evidential=False,
    evi_anneal_epochs=10,
    evi_kl_weight=1e-3,
):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)  # SubGT forward

        if use_evidential:
            loss = evidential_loss_from_logits(
                out,
                batch.y,
                epoch_idx=epoch_idx,
                anneal_epochs=evi_anneal_epochs,
                kl_weight=evi_kl_weight,
                num_classes=2,
            )
        else:
            # 交叉熵损失
            loss = criterion(out, batch.y.long())  # 对于交叉熵，标签需要是整数型 (long)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, use_evidential=False):
    model.eval()
    preds, trues, probs = [], [], []
    uncertainties = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_evidential:
                out_dict = model(batch, return_evidential=True)
                out = out_dict["probs"]
                u = out_dict["uncertainty"].detach().cpu().numpy()
                uncertainties.extend(u.tolist())
            else:
                out = model(batch)
                # 使用 softmax 转换为概率
                out = torch.softmax(out, dim=-1)

            # 预测类别：选择最大概率对应的类别
            pred = torch.argmax(out, dim=-1). cpu().numpy()  # 预测类别
            probs_batch = out[:, 1]. cpu().numpy()  # 取得正类别的概率（适用于二分类）

            preds.extend(pred)
            trues.extend(batch.y.cpu().numpy())
            probs.extend(probs_batch)  # 保存概率（用于 AUC 计算）

    # 转换为 PyTorch tensor
    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    # 计算 AUC (假设是二分类，probs 是正类的概率)
    try:
        auc = roc_auc_score(trues.numpy(), probs)
    except Exception:
        auc = float("nan")

    # 计算 F1-Score (二分类)
    try:
        f1 = f1_score(trues. numpy(), preds.numpy())
    except Exception:
        f1 = float("nan")

    # 计算 MCC (Matthews Correlation Coefficient)
    try:
        mcc = matthews_corrcoef(trues.numpy(), preds.numpy())
    except Exception:
        mcc = float("nan")

    # 计算准确率
    acc = (preds == trues). float().mean().item()

    # ========== 新增：计算 BA, SE, SP ==========
    # 计算混淆矩阵
    try:
        tn, fp, fn, tp = confusion_matrix(trues.numpy(), preds.numpy()).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # BA - Balanced Accuracy (平衡准确率)
    ba = balanced_accuracy_score(trues. numpy(), preds.numpy())
    
    # SE - Sensitivity (灵敏度/召回率/真正例率)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # SP - Specificity (特异度/真负例率)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # ==========================================

    u_mean = float(np.mean(uncertainties)) if len(uncertainties) > 0 else float("nan")
    return auc, f1, mcc, acc, ba, se, sp, u_mean


def main():
    csv_path = os.getenv("EVIMSGT_CSV", "/home/shenxin/EviMSGT/summary_cycpeptmpdb_5splits.csv")
    split_col = os.getenv("EVIMSGT_SPLIT_COL", "split1")
    label_col = os.getenv("EVIMSGT_LABEL_COL", "label")
    permeability_col = os.getenv("EVIMSGT_PERM_COL", "permeability")
    threshold_env = os.getenv("EVIMSGT_PERM_THRESHOLD", "")
    perm_threshold = float(threshold_env) if threshold_env.strip() else None
    mapping_mode = os.getenv("EVIMSGT_MAPPING_MODE", "helm_force").strip().lower()
    dataset = BBBP_Dataset(
        csv_path,
        split_col=split_col,
        label_col=label_col,
        permeability_col=permeability_col,
        permeability_threshold=perm_threshold,
        mapping_mode=mapping_mode,
    )
    split_random_state = 42
    split_strategy = "fixed" if dataset.splits is not None else "random"
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    batch_size = int(os.getenv("EVIMSGT_BATCH_SIZE", "32"))

    if dataset.splits is not None:
        train_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "train"]
        val_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "val"]
        test_idx = [i for i, s in enumerate(dataset.splits) if s.lower() == "test"]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Split column {split_col} must contain train/val/test.")
        train_dataset = [dataset.get(i) for i in train_idx]
        val_dataset = [dataset.get(i) for i in val_idx]
        test_dataset = [dataset.get(i) for i in test_idx]
    else:
        # 划分数据集，80% 训练集，10% 验证集，10% 测试集
        train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
            dataset.smiles, dataset.labels, test_size=0.2, random_state=split_random_state)

        # 从临时集进一步划分，50% 用于验证集，50% 用于测试集（相当于原数据集的10%）
        val_smiles, test_smiles, val_labels, test_labels = train_test_split(
            temp_smiles, temp_labels, test_size=0.5, random_state=split_random_state)

        # 创建训练集、验证集和测试集
        train_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in train_smiles]
        val_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in val_smiles]
        test_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in test_smiles]

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = os.getenv("EVIMSGT_MODEL", "multiscale").lower()
    use_evidential = os.getenv("EVIMSGT_USE_EVIDENTIAL", "1").lower() in {"1", "true", "yes", "y"}
    if model_name == "multiscale":
        model = MultiScaleGraphTransformer(
            in_channels=38,
            edge_features=6,
            num_hidden_channels=256,
            num_layers=4,
            num_residue_layers=2,
            use_evidential=use_evidential,
        ).to(device)
    elif model_name == "subgt":
        model = SubGT(
            in_channels=38,
            edge_features=6,
            num_hidden_channels=256,
            num_layers=6,
        ).to(device)
        use_evidential = False
    else:
        model = GraphTransformer(
            in_channels=38,
            edge_features=6,
            num_hidden_channels=256,
            use_evidential=use_evidential,
        ).to(device)

    # model = SubGT(
    #     in_channels=38,
    #     edge_features=6,
    #     num_hidden_channels=256,
    #     num_layers=6
    # ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 使用 ReduceLROnPlateau 调度器，监控 val_loss 来调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.9, verbose=True)

    epochs=int(os.getenv("EVIMSGT_EPOCHS", "200"))  # 设定训练轮数
    evi_kl_weight = float(os.getenv("EVIMSGT_EVI_KL_WEIGHT", "1e-3"))
    evi_anneal_epochs = int(os.getenv("EVIMSGT_EVI_ANNEAL_EPOCHS", "10"))
    best_acc = 0.0

    print(f"Dataset: {csv_path}")
    print(f"Label source: {dataset.label_source}")
    print(f"Residue mapping mode: {mapping_mode}")
    print(f"Split strategy: {split_strategy} ({split_col if split_strategy == 'fixed' else 'random'})")
    print(f"Samples -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    print(f"Use evidential head: {use_evidential}")
    if use_evidential:
        print(f"Evidential settings -> kl_weight: {evi_kl_weight}, anneal_epochs: {evi_anneal_epochs}")

    feature_config = build_feature_config(use_chirality=False, explicit_h=False)
    data_config = {
        "dataset": "BBBP",
        "train_csv": csv_path,
        "cache_file": "dynamic cache (see dataset cache_dir)",
        "split_strategy": split_strategy,
        "split_random_state": split_random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_total": len(dataset),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "external_test_set": None,
        "label_definition": "1: high permeability / 0: low permeability",
        "label_source": dataset.label_source,
        "use_evidential": use_evidential,
        "residue_mapping_mode": mapping_mode,
    }

    for epoch in range(epochs):
        # 训练过程
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            use_evidential=use_evidential,
            evi_anneal_epochs=evi_anneal_epochs,
            evi_kl_weight=evi_kl_weight,
        )

        # 在测试集上评估
        auc_test, f1_test, mcc_test, acc_test, ba_test, se_test, sp_test, u_mean = evaluate(
            model,
            test_loader,
            device,
            use_evidential=use_evidential,
        )

        # 输出训练过程中的各项指标
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {loss:.4f}, Test AUC: {auc_test:.4f}, Test F1: {f1_test:.4f}, Test MCC: {mcc_test:.4f}, "
            f"Test ACC: {acc_test:.4f}, Test BA: {ba_test:.4f}, Test SE: {se_test:.4f}, Test SP: {sp_test:.4f}")
        if use_evidential:
            print(f"Test Uncertainty Mean: {u_mean:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 80)

        # 在每个epoch结束后调用scheduler.step(val_loss)来调整学习率
        scheduler.step(acc_test)
        
        if acc_test > best_acc:
            best_acc = acc_test
            model_config = build_model_config(model)
            training_config = build_training_config(optimizer, scheduler, batch_size, epochs, criterion)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'model_config': model_config,
                'feature_config': feature_config,
                'training_config': training_config,
                'data_config': data_config,
                'preprocessing': {
                    'atom_feature_dim': 38,
                    'bond_feature_dim': 6,
                    'use_chirality': False,
                    'explicit_H': False,
                    'multiscale_atom_residue_index': True,
                    'use_evidential': use_evidential,
                    'residue_mapping_mode': mapping_mode,
                },
                'metrics': {
                    'train_loss': loss,
                    'test_acc': acc_test,
                    'test_auc': auc_test,
                    'test_f1': f1_test,
                    'test_mcc': mcc_test,
                    'test_ba': ba_test,
                    'test_se': se_test,
                    'test_sp': sp_test,
                    'test_uncertainty_mean': u_mean,
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            
            os.makedirs('/home/shenxin/EviMSGT/ckpt', exist_ok=True)
            save_path = f'/home/shenxin/EviMSGT/ckpt/09best_model_epoch{epoch+1}_acc{acc_test:.4f}.pt'
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    main()
