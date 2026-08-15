#!/usr/bin/env python
"""Periodontitis (GSE164241pero) DEG-Expanded + TT Specialist.

Data: GSE164241pero/mtx/count/{features,barcodes,matrix}.mtx.gz
Conditions: BM=Healthy (baseline) -> GM=Gingivitis -> PD=Periodontitis
Cell types: Epithelial, Fibroblast (oral tissue core)

Output: results/4_case_studies/periodontitis/deg_expanded/{cell_type}/{stage}.pkl
        results/4_case_studies/periodontitis/deg_expanded/bm_healthy_cells.h5ad  (for PVL cross-dataset control)
"""
import sys, gzip, time, gc, pickle, torch
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
DATA_DIR = DATA_ROOT / "GSE164241pero" / "mtx" / "count"
OUT_DIR = RESULT_ROOT / "4_case_studies" / "periodontitis" / "deg_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['BM', 'GM', 'PD']           # BM=Healthy, GM=Gingivitis, PD=Periodontitis
BASELINE = 'BM'
DEG_COMPARISONS = [('GM', 'BM'), ('PD', 'BM')]
CELL_TYPES = ['Epithelial', 'Fibroblast']
EDGE_THRESHOLD = 0.2
MAX_CELLS = 1000

# Oral/gingival marker genes (from eval_periodontitis)
MARKERS = {
    'Epithelial':  ['KRT5','KRT14','KRT13','KRT4','TP63','CDH1','EPCAM','KRT15','KRT19'],
    'Fibroblast':  ['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA','FAP','COL6A1','THY1','SPARC'],
    'T_cell':      ['CD3D','CD3E','CD3G','CD4','CD8A','TRAC','CD2','CCL5','NKG7'],
    'B_Plasma':    ['CD19','MS4A1','CD79A','JCHAIN','MZB1','SDC1','XBP1','IGHG1'],
    'Myeloid':     ['CD14','CD68','CD163','CSF1R','LYZ','CST3','AIF1','ITGAX'],
    'Endothelial': ['PECAM1','VWF','CDH5','ENG','CLDN5','FLT1'],
    'Neutrophil':  ['ELANE','MPO','CXCL8','CSF3R','FCGR3B','MMP8','S100A8','S100A9'],
    'Mast':        ['KIT','TPSAB1','CPA3','HDC','MS4A2','GATA2'],
}

device = torch.device('cuda')
print(f"device={device}")
t_start = time.time()

# -- Load models --
print("Loading models (TT specialist)...")
known_tfs = load_known_tfs('human')
enc, edge_head, dir_enc, dir_head, G_val = load_models(device)
print(f"  G={G_val}, models ready")

# ===========================================================
# 1. Load mtx data (features have 2 cols, barcodes encode condition)
# ===========================================================
print("\n[1/4] Loading mtx data...")
genes = []
with gzip.open(DATA_DIR / "features.tsv.gz", 'rt') as f:
    for line in f:
        cols = line.strip().split('\t')
        genes.append(cols[0])
print(f"  {len(genes):,} genes")

barcodes = []
with gzip.open(DATA_DIR / "barcodes.tsv.gz", 'rt') as f:
    for line in f:
        barcodes.append(line.strip())

# Parse condition from barcode prefix (BM/GM/PD)
cond_map = {}
for bc in barcodes:
    parts = bc.split('_')
    donor = parts[1] if len(parts) > 1 else parts[0]
    if donor.startswith('BM'):
        cond = 'BM'
    elif donor.startswith('GM'):
        cond = 'GM'
    elif donor.startswith('PD'):
        cond = 'PD'
    else:
        cond = 'PD'
    cond_map[bc] = cond
print(f"  Condition: {pd.Series(list(cond_map.values())).value_counts().to_dict()}")

print("  Loading sparse matrix...")
t0 = time.time()
rows, cols, vals = [], [], []
with gzip.open(DATA_DIR / "matrix.mtx.gz", 'rt') as f:
    header = f.readline()
    while header.startswith('%'):
        header = f.readline()
    n_genes, n_cells, n_entries = map(int, header.strip().split())
    print(f"  Matrix: {n_genes} x {n_cells}, {n_entries:,} entries")
    for line in f:
        g, c, v = line.strip().split()
        rows.append(int(g) - 1)
        cols.append(int(c) - 1)
        vals.append(float(v))
data_sparse = sparse.coo_matrix((vals, (rows, cols)), shape=(n_genes, n_cells)).tocsc().T.tocsr()
del rows, cols, vals; gc.collect()
print(f"  Loaded in {time.time()-t0:.0f}s, shape: {data_sparse.shape}")

obs = pd.DataFrame({'barcode': barcodes,
                    'condition': [cond_map[bc] for bc in barcodes]}, index=barcodes)
var = pd.DataFrame({'gene': genes}, index=genes)
adata = sc.AnnData(X=data_sparse, obs=obs, var=var)
adata = adata[adata.obs['condition'].isin(CONDITIONS)].copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"  After QC+norm: {adata.shape}")

# ===========================================================
# 2. Cell type annotation (marker-based, same as eval script)
# ===========================================================
print("\n[2/4] Cell type annotation...")
expr = adata.X
scores = {}
var_list = list(adata.var_names)
for ct, mgs in MARKERS.items():
    present = [g for g in mgs if g in adata.var_names]
    if len(present) < 2:
        scores[ct] = np.zeros(adata.shape[0]); continue
    gene_idx = [var_list.index(g) for g in present]
    scores[ct] = np.array(expr[:, gene_idx].mean(axis=1)).ravel()
score_matrix = np.column_stack([scores[ct] for ct in MARKERS.keys()])
best_idx = np.argmax(score_matrix, axis=1)
ct_names = list(MARKERS.keys())
adata.obs['cell_type'] = [ct_names[i] for i in best_idx]
max_scores = np.max(score_matrix, axis=1)
adata.obs.loc[max_scores < 0.1, 'cell_type'] = 'Unknown'
print("  Cell type: " + str(adata.obs['cell_type'].value_counts().to_dict()))

# Save BM healthy cells (for PVL cross-dataset control)
bm_adata = adata[adata.obs['condition'] == 'BM'].copy()
bm_adata.write_h5ad(OUT_DIR / "bm_healthy_cells.h5ad")
print(f"  Saved BM healthy cells: {bm_adata.shape} -> bm_healthy_cells.h5ad")

# ===========================================================
# 3. Per cell-type DEG-Expanded inference
# ===========================================================
for ct in CELL_TYPES:
    ct_data = adata[adata.obs['cell_type'] == ct].copy()
    if ct_data.n_obs < 100:
        print(f"\n[{ct}] only {ct_data.n_obs} cells, skip"); continue
    out_ct = OUT_DIR / ct.lower()
    out_ct.mkdir(parents=True, exist_ok=True)
    # Skip if all conditions already have pkls (resume support)
    done = all((out_ct / f"{c.lower()}.pkl").exists() for c in CONDITIONS)
    if done and (out_ct / "metadata.pkl").exists():
        print(f"\n[{ct}] all conditions already done, skip"); continue
    print(f"\n{'='*60}\n[{ct}] {ct_data.n_obs} cells")
    all_genes = list(ct_data.var_names)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    obs_cond = ct_data.obs['condition']

    print("  Computing DEGs (GM/PD vs BM, no cap)...")
    t_deg = time.time()
    X = ct_data.X.toarray() if sparse.issparse(ct_data.X) else np.asarray(ct_data.X)
    deg_df = compute_degs(X, all_genes, obs_cond, DEG_COMPARISONS, BASELINE,
                          known_tfs=known_tfs)
    n_deg = len(deg_df)
    n_tf = int(deg_df['is_tf'].sum()) if n_deg > 0 else 0
    print(f"  DEGs: {n_deg} ({n_tf} TFs) [{time.time()-t_deg:.1f}s]")
    if n_deg < 50:
        print(f"  [SKIP] too few DEGs"); continue

    deg_sorted = deg_df.sort_values(['max_abs_log2fc', 'n_stages_sig'],
                                    ascending=[False, False])
    all_deg_genes = deg_sorted['gene'].tolist()
    tf_in_data = [g for g in all_deg_genes if g in known_tfs]
    print(f"  Selected: {len(all_deg_genes)} genes ({len(tf_in_data)} TFs), "
          f"coverage {100*len(all_deg_genes)/len(all_genes):.1f}%")

    meta = {'tf_genes': tf_in_data, 'all_deg_genes': all_deg_genes,
            'n_total_genes': len(all_genes), 'n_deg': n_deg, 'n_tf': n_tf,
            'method': 'deg_expanded_tt', 'case': 'Periodontitis', 'cell_type': ct,
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
        if n_cells < 50:
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

print(f"\n{'='*60}\nPeriodontitis DEG-Expanded complete: {(time.time()-t_start)/60:.1f} min")
print("DONE")
