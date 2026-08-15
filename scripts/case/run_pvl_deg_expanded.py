#!/usr/bin/env python
"""PVL (GSE196296 oral mucosa) DEG-Expanded + TT Specialist.

PVL is an oral mucosal disease (NOT brain). It has 3 lesion samples (A=early,
B=mid, C=late) but NO healthy control. We use Periodontitis BM (healthy oral
gingival tissue) as a cross-dataset control, with explicit labeling.

Data (from prep_pvl_data.py):
  pvl_fullgene.h5ad      : PVL full-gene (A/B/C), log-normalized
  bm_healthy_cells.h5ad  : Periodontitis BM healthy oral cells, log-normalized

Strategy: intersect genes -> concatenate -> for each shared cell type, DEG-Expanded
  with Healthy (BM) as baseline, A/B/C as disease progression stages.

Output: results/4_case_studies/pvl/deg_expanded/{cell_type}/{stage}.pkl
"""
import sys, time, gc, pickle, torch
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deg_expanded_core import (load_known_tfs, load_models, compute_degs,
                                expanded_deg_grn)
from scipy import sparse as sp_sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, RESULT_ROOT
OUT_DIR = RESULT_ROOT / "4_case_studies" / "pvl" / "deg_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Conditions: Healthy (Periodontitis BM, cross-dataset control) -> A -> B -> C
CONDITIONS = ['Healthy', 'A', 'B', 'C']
BASELINE = 'Healthy'
DEG_COMPARISONS = [('A', 'Healthy'), ('B', 'Healthy'), ('C', 'Healthy')]
# Cell types: oral mucosa core (Epithelial, Fibroblast); others auto-selected
CELL_TYPES_FALLBACK = ['Epithelial', 'Fibroblast']
EDGE_THRESHOLD = 0.2
MAX_CELLS = 1000
MIN_CELLS_PER_COND = 50  # lower threshold since BM may have fewer cells per type

device = torch.device('cuda')
print(f"device={device}")
t_start = time.time()

# -- Load models --
print("Loading models (TT specialist)...")
known_tfs = load_known_tfs('human')
enc, edge_head, dir_enc, dir_head, G_val = load_models(device)
print(f"  G={G_val}, models ready")

# ===========================================================
# 1. Load PVL full-gene + BM healthy, intersect genes, concatenate
# ===========================================================
print("\n[1/4] Loading PVL + BM healthy data...")
adata_pvl = sc.read_h5ad(OUT_DIR / "pvl_fullgene.h5ad")
print(f"  PVL: {adata_pvl.shape}, samples: {adata_pvl.obs['sample'].value_counts().to_dict()}")

adata_bm = sc.read_h5ad(OUT_DIR / "bm_healthy_cells.h5ad")
print(f"  BM:  {adata_bm.shape}")

# Intersect genes
shared_genes = sorted(set(adata_pvl.var_names) & set(adata_bm.var_names))
print(f"  Shared genes: {len(shared_genes):,}")
adata_pvl = adata_pvl[:, shared_genes].copy()
adata_bm = adata_bm[:, shared_genes].copy()

# Label conditions
# PVL: sample A/B/C -> condition A/B/C
adata_pvl.obs['condition'] = adata_pvl.obs['sample'].astype(str)
# BM: condition -> 'Healthy' (cross-dataset control from Periodontitis)
adata_bm.obs['condition'] = 'Healthy'
adata_bm.obs['sample'] = 'Perio_BM'
adata_bm.obs['control_source'] = 'Periodontitis_BM'

# Ensure cell_type column exists on BM
if 'cell_type' not in adata_bm.obs.columns:
    adata_bm.obs['cell_type'] = 'Unknown'

# Concatenate
import anndata
adata = anndata.concat([adata_pvl, adata_bm], join='inner', merge='same',
                        index_unique='-')
adata.obs['condition'] = adata.obs['condition'].astype(str)
adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)
print(f"  Combined: {adata.shape}")
print(f"  Condition: {adata.obs['condition'].value_counts().to_dict()}")
del adata_pvl, adata_bm; gc.collect()

# ===========================================================
# 2. Auto-select viable cell types (shared between PVL and BM)
# ===========================================================
print("\n[2/4] Cell type viability (Healthy + A/B/C):")
ct_counts = adata.obs['cell_type'].value_counts()
viable_types = []
for ct in ct_counts.index:
    if ct == 'Unknown':
        continue
    counts = {c: int(((adata.obs['cell_type'] == ct) & (adata.obs['condition'] == c)).sum())
              for c in CONDITIONS}
    viable = all(v >= MIN_CELLS_PER_COND for v in counts.values())
    cs = "  ".join(f"{c}={counts[c]:>5,d}" for c in CONDITIONS)
    print(f"  {ct:<14s}: {cs}  {'OK' if viable else 'skip'}")
    if viable:
        viable_types.append(ct)

if not viable_types:
    print("  [WARN] no fully-viable types; using fallback (Epithelial/Fibroblast)")
    viable_types = [ct for ct in CELL_TYPES_FALLBACK if ct in ct_counts.index]
print(f"  Viable: {viable_types}")

# ===========================================================
# 3. Per cell-type DEG-Expanded inference
# ===========================================================
for ct in viable_types:
    ct_data = adata[adata.obs['cell_type'] == ct].copy()
    if ct_data.n_obs < 100:
        print(f"\n[{ct}] only {ct_data.n_obs} cells, skip"); continue
    print(f"\n{'='*60}\n[{ct}] {ct_data.n_obs} cells")
    all_genes = list(ct_data.var_names)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    obs_cond = ct_data.obs['condition']

    print(f"  Computing DEGs (A/B/C vs Healthy=Periodontitis_BM, no cap)...")
    t_deg = time.time()
    X = ct_data.X.toarray() if sparse.issparse(ct_data.X) else np.asarray(ct_data.X)
    deg_df = compute_degs(X, all_genes, obs_cond, DEG_COMPARISONS, BASELINE,
                          known_tfs=known_tfs)
    n_deg = len(deg_df)
    n_tf = int(deg_df['is_tf'].sum()) if n_deg > 0 else 0
    print(f"  DEGs: {n_deg} ({n_tf} TFs) [{time.time()-t_deg:.1f}s]")
    if n_deg < 50:
        print(f"  [SKIP] too few DEGs ({n_deg})"); continue

    deg_sorted = deg_df.sort_values(['max_abs_log2fc', 'n_stages_sig'],
                                    ascending=[False, False])
    all_deg_genes = deg_sorted['gene'].tolist()
    tf_in_data = [g for g in all_deg_genes if g in known_tfs]
    print(f"  Selected: {len(all_deg_genes)} genes ({len(tf_in_data)} TFs), "
          f"coverage {100*len(all_deg_genes)/len(all_genes):.1f}%")

    out_ct = OUT_DIR / ct.lower()
    out_ct.mkdir(parents=True, exist_ok=True)
    meta_save = {'tf_genes': tf_in_data, 'all_deg_genes': all_deg_genes,
                 'n_total_genes': len(all_genes), 'n_deg': n_deg, 'n_tf': n_tf,
                 'method': 'deg_expanded_tt', 'case': 'PVL', 'cell_type': ct,
                 'conditions': CONDITIONS, 'baseline': BASELINE,
                 'control_source': 'Periodontitis_BM (cross-dataset healthy oral)'}
    with open(out_ct / "metadata.pkl", 'wb') as f:
        pickle.dump(meta_save, f)
    del X; gc.collect()

    for cond in CONDITIONS:
        cond_mask = ct_data.obs['condition'] == cond
        X_ct = ct_data[cond_mask].X
        if sparse.issparse(X_ct):
            X_ct = X_ct.toarray().astype(np.float32)
        else:
            X_ct = np.asarray(X_ct, dtype=np.float32)
        n_cells = X_ct.shape[0]
        if n_cells < 30:
            print(f"  {cond}: only {n_cells} cells, skip"); continue
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
        ep_sparse = sp_sparse.csr_matrix(ep); ds_sparse = sp_sparse.csr_matrix(ds)
        with open(out_ct / f"{cond.lower()}.pkl", 'wb') as f:
            pickle.dump({'edge_prob_sparse': ep_sparse, 'dir_score_sparse': ds_sparse,
                         'n_cells': n_cells, 'n_windows': n_wins,
                         'n_edges_above_thresh': n_edges}, f)
        del ep, ds, ep_sparse, ds_sparse, X_ct; gc.collect()

print(f"\n{'='*60}\nPVL DEG-Expanded complete: {(time.time()-t_start)/60:.1f} min")
print("DONE")
