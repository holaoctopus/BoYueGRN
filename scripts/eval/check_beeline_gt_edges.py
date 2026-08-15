#!/usr/bin/env python
"""Quick check: TF->TF vs TF->non-TF edge counts in BEELINE ground truth."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import DATA_ROOT, result_dir

BEELINE_DIR = DATA_ROOT / "BEELINE"

BL = {
    'mDC': ("mouse/mDC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mDC/ExpressionData.csv"),
    'mHSC-E': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv"),
    'mHSC-GM': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv"),
    'mHSC-L': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv"),
    'hESC': ("human/hESC-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv"),
    'hHep': ("human/HepG2-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hHep/ExpressionData.csv"),
}

results = []

for ds_name, (net_rel, tf_rel, expr_rel) in BL.items():
    expr_path = BEELINE_DIR / expr_rel
    if not expr_path.exists():
        continue

    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes = list(expr_df.columns)
    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = list(tf_df[0])
    tf_set = set(t.upper() for t in tf_names)

    # Build TF index set
    tf_idx_set = set()
    for i, g in enumerate(genes):
        if g.upper() in tf_set:
            tf_idx_set.add(i)

    # Load GT network
    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    gt = np.zeros((len(genes), len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in gene_to_idx and t in gene_to_idx:
            gt[gene_to_idx[s], gene_to_idx[t]] = 1.0

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

    total = gt_tftf + gt_nontf
    pct = gt_tftf / total * 100 if total > 0 else 0

    print(f"{ds_name}: TF->TF={gt_tftf}, TF->non-TF={gt_nontf}, TF->TF%={pct:.2f}%")

    results.append({
        'dataset': ds_name,
        'n_genes': len(genes),
        'n_tfs': len(tf_idx_set),
        'gt_tftf': gt_tftf,
        'gt_nontf': gt_nontf,
        'gt_total': total,
        'gt_tftf_pct': pct,
    })

df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("BEELINE GT Edge Type Distribution")
print("=" * 70)
print(df.to_string(index=False))

out_path = result_dir("3_full_gene") / "beeline_gt_edge_type_counts.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
