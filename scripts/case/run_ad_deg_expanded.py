#!/usr/bin/env python
"""AD (GSE157827) DEG-Expanded + TT Specialist.

Data: GSE157827_raw.h5ad (29,736 full genes) + celltype annotation
      (barcode-aligned, merged from GSE157827_celltype.h5ad)
Conditions: Control (baseline) vs AD
Cell types: Microglia (AD neuroinflammation core), Astrocyte

Output: results/4_case_studies/ad/deg_expanded/{cell_type}/{stage}.pkl
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import torch
import time
import gc
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deg_expanded_core import (load_known_tfs, load_models, compute_degs,
                                expanded_deg_grn)
from scipy import sparse as sp_sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, RESULT_ROOT
DATA_RAW = DATA_ROOT / "GSE157827" / "processed" / "GSE157827_raw.h5ad"
DATA_CT = DATA_ROOT / "GSE157827" / "processed" / "GSE157827_celltype.h5ad"
OUT_DIR = RESULT_ROOT / "4_case_studies" / "ad" / "deg_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['Control', 'AD']
BASELINE = 'Control'
DEG_COMPARISONS = [('AD', 'Control')]
CELL_TYPES = ['Microglia', 'Astrocyte']
EDGE_THRESHOLD = 0.2
MAX_CELLS = 1000

device = torch.device('cuda')
print(f"device={device}")
t_start = time.time()

# -- Load models --
print("Loading models (TT specialist)...")
known_tfs = load_known_tfs('human')
enc, edge_head, dir_enc, dir_head, G_val = load_models(device)
print(f"  G={G_val}, models ready")

# -- Load data: raw full genes + celltype annotation (barcode-aligned) --
print("\nLoading raw h5ad (full genes) + celltype annotation...")
adata = sc.read_h5ad(DATA_RAW)
adata_ct = sc.read_h5ad(DATA_CT, backed='r')
adata.obs['cell_type'] = adata_ct.obs['cell_type'].reindex(adata.obs_names).values
adata.obs['condition'] = adata_ct.obs['condition'].reindex(adata.obs_names).values
del adata_ct; gc.collect()

adata = adata[adata.obs['condition'].isin(CONDITIONS)].copy()
adata = adata[adata.obs['cell_type'].isin(CELL_TYPES)].copy()
sc.pp.filter_genes(adata, min_cells=3)
print(f"  Filtered: {adata.shape}, cell_types: {adata.obs['cell_type'].unique().tolist()}")
print(f"  Condition distribution: {adata.obs['condition'].value_counts().to_dict()}")

# -- Per cell-type DEG-Expanded inference --
for ct in CELL_TYPES:
    ct_mask = adata.obs['cell_type'] == ct
    ct_data = adata[ct_mask].copy()
    if ct_data.n_obs < 100:
        print(f"\n[{ct}] only {ct_data.n_obs} cells, skip")
        continue
    out_ct = OUT_DIR / ct.lower()
    out_ct.mkdir(parents=True, exist_ok=True)
    # Skip if all conditions already have pkls (resume support)
    done = all((out_ct / f"{c.lower()}.pkl").exists() for c in CONDITIONS)
    if done and (out_ct / "metadata.pkl").exists():
        print(f"\n[{ct}] all conditions already done, skip")
        continue
    print(f"\n{'='*60}\n[{ct}] {ct_data.n_obs} cells")
    all_genes = list(ct_data.var_names)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    obs_cond = ct_data.obs['condition']

    print("  Computing DEGs (AD vs Control, no cap)...")
    t_deg = time.time()
    X = ct_data.X.toarray() if sparse.issparse(ct_data.X) else np.asarray(ct_data.X)
    deg_df = compute_degs(X, all_genes, obs_cond, DEG_COMPARISONS, BASELINE,
                          known_tfs=known_tfs)
    n_deg = len(deg_df)
    n_tf = int(deg_df['is_tf'].sum()) if n_deg > 0 else 0
    print(f"  DEGs: {n_deg} ({n_tf} TFs, {n_deg - n_tf} targets) [{time.time()-t_deg:.1f}s]")
    if n_deg < 50:
        print(f"  [SKIP] too few DEGs ({n_deg})")
        continue

    deg_sorted = deg_df.sort_values(['max_abs_log2fc', 'n_stages_sig'],
                                    ascending=[False, False])
    all_deg_genes = deg_sorted['gene'].tolist()
    tf_in_data = [g for g in all_deg_genes if g in known_tfs]
    print(f"  Selected: {len(all_deg_genes)} genes ({len(tf_in_data)} TFs)")
    print(f"  Coverage: {100*len(all_deg_genes)/len(all_genes):.1f}% of dataset")

    meta = {'tf_genes': tf_in_data, 'all_deg_genes': all_deg_genes,
            'n_total_genes': len(all_genes), 'n_deg': n_deg, 'n_tf': n_tf,
            'method': 'deg_expanded_tt', 'case': 'AD', 'cell_type': ct,
            'conditions': CONDITIONS, 'baseline': BASELINE}
    with open(out_ct / "metadata.pkl", 'wb') as f:
        pickle.dump(meta, f)

    del X; gc.collect()

    for cond in CONDITIONS:
        cond_mask = ct_data.obs['condition'] == cond
        X_ct = ct_data[cond_mask].X
        if sparse.issparse(X_ct):
            X_ct = X_ct.toarray().astype(np.float32)
        else:
            X_ct = np.asarray(X_ct, dtype=np.float32)
        n_cells = X_ct.shape[0]
        print(f"\n  {cond}: {n_cells} cells", end='')
        if n_cells > MAX_CELLS:
            rng = np.random.RandomState(42)
            X_ct = X_ct[rng.choice(n_cells, MAX_CELLS, replace=False)]
            print(f" (->{MAX_CELLS})", end='')

        t0 = time.time()
        ep, ds, n_wins = expanded_deg_grn(
            X_ct, all_deg_genes, tf_in_data, gene_to_idx,
            enc, edge_head, dir_enc, dir_head, device, G_val,
            top_k=50, tf_per_win=5, max_cells=500, max_tfs=150, verbose=True)
        n_edges = int((ep > EDGE_THRESHOLD).sum())
        print(f"    -> {n_wins} windows, {n_edges:,} edges ({time.time()-t0:.1f}s)")

        ep_sparse = sp_sparse.csr_matrix(ep)
        ds_sparse = sp_sparse.csr_matrix(ds)
        with open(out_ct / f"{cond.lower()}.pkl", 'wb') as f:
            pickle.dump({'edge_prob_sparse': ep_sparse, 'dir_score_sparse': ds_sparse,
                         'n_cells': n_cells, 'n_windows': n_wins,
                         'n_edges_above_thresh': n_edges}, f)
        del ep, ds, ep_sparse, ds_sparse, X_ct; gc.collect()

    print(f"\n[{ct}] done in {(time.time()-t_start)/60:.1f} min")

print(f"\n{'='*60}\nAD DEG-Expanded complete. Total: {(time.time()-t_start)/60:.1f} min")
print("DONE")
