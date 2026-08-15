#!/usr/bin/env python
"""Check TF->TF vs TF->non-TF edge counts in BEELINE evaluation."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import result_dir

# Reuse functions from beeline_cross_specialist
from beeline_cross_specialist import (
    BL, BEELINE_DIR, THRESHOLD, G,
    encoders, edge_heads, dir_enc_nt, dir_head_nt, dir_enc_tt, dir_head_tt,
    compute_P_D, predict_edge, predict_dir_both, build_tf_windows
)
import torch
from sklearn.covariance import LedoitWolf
import gc
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Checking TF->TF edge counts in BEELINE multi-window evaluation...")
print(f"Threshold: edge_prob > {THRESHOLD}")
print()

results = []

for ds_name, (net_rel, tf_rel, expr_rel) in BL.items():
    expr_path = BEELINE_DIR / expr_rel
    if not expr_path.exists():
        continue

    print(f"{ds_name}:")

    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes = list(expr_df.columns)
    X = expr_df.values.astype(np.float32)
    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = list(tf_df[0])
    tf_set = set(t.upper() for t in tf_names)

    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    gt = np.zeros((len(genes), len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in gene_to_idx and t in gene_to_idx:
            gt[gene_to_idx[s], gene_to_idx[t]] = 1.0

    X_std = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    # Build TF index set
    tf_idx_set = set()
    for i, g in enumerate(genes):
        if g.upper() in tf_set:
            tf_idx_set.add(i)

    # Count GT edges by type
    gt_tftf = 0
    gt_nontf = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] > 0:
                if i in tf_idx_set and j in tf_idx_set:
                    gt_tftf += 1
                else:
                    gt_nontf += 1

    print(f"  GT edges: TF->TF={gt_tftf}, TF->non-TF={gt_nontf}, ratio={gt_tftf/max(gt_tftf+gt_nontf,1)*100:.2f}%")

    # Multi-window evaluation
    windows, tf_in = build_tf_windows(genes, tf_names, X_std)
    if not windows:
        continue

    ep_full = np.zeros((len(genes), len(genes)), dtype=np.float32)
    ds_cnt = np.zeros((len(genes), len(genes)), dtype=np.float32)

    for wi, wg in enumerate(windows):
        widx = [gene_to_idx[g] for g in wg if g in gene_to_idx]
        nw = len(widx)
        if nw < 10:
            continue
        Xw = X_std[:, widx]
        Xp = np.zeros((Xw.shape[0], G), dtype=np.float32)
        Xp[:, :nw] = Xw
        P, D = compute_P_D(Xp)
        ep_w = predict_edge(P, D)[:nw, :nw]
        for i, gi in enumerate(widx):
            for j, gj in enumerate(widx):
                ep_full[gi, gj] = max(ep_full[gi, gj], ep_w[i, j])
                ds_cnt[gi, gj] += 1

    # Count evaluated edges by type (edge_prob > threshold & gt > 0)
    pred_mask = ep_full > THRESHOLD
    gt_mask = gt > 0
    both = pred_mask & gt_mask

    eval_tftf = 0
    eval_nontf = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if both[i, j]:
                if i in tf_idx_set and j in tf_idx_set:
                    eval_tftf += 1
                else:
                    eval_nontf += 1

    print(f"  Eval edges (prob>{THRESHOLD} & GT>0): TF->TF={eval_tftf}, TF->non-TF={eval_nontf}")
    print(f"  TF->TF eval rate: {eval_tftf/max(gt_tftf,1)*100:.1f}% of GT TF->TF")
    print()

    results.append({
        'dataset': ds_name,
        'gt_tftf': gt_tftf,
        'gt_nontf': gt_nontf,
        'gt_tftf_pct': gt_tftf / max(gt_tftf + gt_nontf, 1) * 100,
        'eval_tftf': eval_tftf,
        'eval_nontf': eval_nontf,
        'eval_tftf_pct': eval_tftf / max(eval_tftf + eval_nontf, 1) * 100,
        'n_windows': len(windows),
    })

    del X, X_std, ep_full, ds_cnt, gt
    gc.collect()

df = pd.DataFrame(results)
print("=" * 80)
print("SUMMARY: TF->TF Edge Counts")
print("=" * 80)
print(df.to_string(index=False))

out_path = result_dir("3_full_gene") / "beeline_edge_type_counts.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
