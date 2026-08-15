#!/usr/bin/env python
"""HCC (GSE149614) DEG-Expanded + TT Specialist.

Data: HCC_GSE149614/count.txt.gz (cells x genes, raw counts) + metadata.txt.gz
  - metadata `Cell`     : cell barcode (matches count.txt.gz header)
  - metadata `celltype` : cell type annotation
  - metadata `site`     : condition (Normal / Tumor) -> Normal=baseline
Conditions: Normal (baseline) vs Tumor
Cell types: top 2-3 by abundance among Normal+Tumor (e.g. Hepatocyte, Myeloid, T/NK)

Output: results/4_case_studies/hcc/deg_expanded/{cell_type}/{stage}.pkl
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
DATA_DIR = DATA_ROOT / "HCC_GSE149614"
OUT_DIR = RESULT_ROOT / "4_case_studies" / "hcc" / "deg_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['Normal', 'Tumor']
BASELINE = 'Normal'
DEG_COMPARISONS = [('Tumor', 'Normal')]
# Will be auto-selected below; fallback list ordered by expected abundance
CELL_TYPES_FALLBACK = ['Hepatocyte', 'Myeloid', 'T/NK']
EDGE_THRESHOLD = 0.2
MAX_CELLS = 1000
MIN_CELLS_PER_COND = 100

device = torch.device('cuda')
print(f"device={device}")
t_start = time.time()

# -- Load models --
print("Loading models (TT specialist)...")
known_tfs = load_known_tfs('human')
enc, edge_head, dir_enc, dir_head, G_val = load_models(device)
print(f"  G={G_val}, models ready")

# ===========================================================
# 1. Load count.txt.gz (cells x genes, raw counts) + metadata
# ===========================================================
print("\n[1/4] Loading count.txt.gz + metadata...")
# Chunked loading: read genes×cells in batches -> build sparse incrementally (low memory)
t0 = time.time()
header_cells = None
gene_names = []
sparse_blocks = []  # list of (genes×cells) CSR blocks
chunk_iter = pd.read_csv(DATA_DIR / "count.txt.gz", sep='\t', compression='gzip',
                         index_col=0, chunksize=2000)
for chunk in chunk_iter:
    if header_cells is None:
        header_cells = list(chunk.columns)
        cell_to_idx = {cell: i for i, cell in enumerate(header_cells)}
    gene_names.extend(list(chunk.index))
    # Build sparse (genes×cells) from chunk, keep as CSR
    block = sparse.csr_matrix(chunk.values.astype(np.float32))
    sparse_blocks.append(block)
    print(f"  Read {len(gene_names):,} genes, block {block.shape}...", end='\r')
# vstack all blocks -> (genes×cells), then transpose -> (cells×genes)
data_sparse = sparse.vstack(sparse_blocks).T.tocsr()
del sparse_blocks; gc.collect()
print(f"\n  {len(header_cells):,} cells x {len(gene_names):,} genes "
      f"({time.time()-t0:.0f}s), nnz={data_sparse.nnz:,}")

meta = pd.read_csv(DATA_DIR / "metadata.txt.gz", sep='\t', compression='gzip')
print(f"  Metadata: {meta.shape[0]:,} rows, cols: {meta.columns.tolist()}")
meta = meta[meta['Cell'].isin(header_cells)].copy()
meta['_idx'] = meta['Cell'].map(cell_to_idx)
meta = meta.sort_values('_idx').reset_index(drop=True)
print(f"  Matched cells: {meta.shape[0]:,}")
print(f"  site: {sorted(meta['site'].unique().tolist())}")
print(f"  celltype: {sorted(meta['celltype'].unique().tolist())}")

obs = pd.DataFrame({
    'barcode': meta['Cell'].values,
    'cell_type': meta['celltype'].astype(str).values,
    'condition': meta['site'].astype(str).values,
}, index=meta['Cell'].values)
var = pd.DataFrame({'gene': gene_names}, index=gene_names)
adata = sc.AnnData(X=data_sparse, obs=obs, var=var)
adata = adata[adata.obs['condition'].isin(CONDITIONS)].copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"  After QC+norm: {adata.shape}")
print(f"  Condition: {adata.obs['condition'].value_counts().to_dict()}")

# ===========================================================
# 2. Auto-select viable cell types (>=MIN_CELLS_PER_COND in both Normal+Tumor)
# ===========================================================
print("\n[2/4] Cell type viability:")
ct_counts = adata.obs['cell_type'].value_counts()
viable_types = []
for ct in ct_counts.index:
    if ct == 'Unknown' or ct == 'nan':
        continue
    counts = {c: int(((adata.obs['cell_type'] == ct) & (adata.obs['condition'] == c)).sum())
              for c in CONDITIONS}
    viable = all(v >= MIN_CELLS_PER_COND for v in counts.values())
    print(f"  {ct:<14s}: Normal={counts['Normal']:>5,d}  Tumor={counts['Tumor']:>5,d}  {'OK' if viable else 'skip'}")
    if viable:
        viable_types.append(ct)

if not viable_types:
    print("  [WARN] no viable types from auto; using fallback list")
    viable_types = [ct for ct in CELL_TYPES_FALLBACK if ct in ct_counts.index]
print(f"  Viable: {viable_types}")

# ===========================================================
# 3. Per cell-type DEG-Expanded inference
# ===========================================================
for ct in viable_types:
    ct_data = adata[adata.obs['cell_type'] == ct].copy()
    if ct_data.n_obs < 100:
        print(f"\n[{ct}] only {ct_data.n_obs} cells, skip"); continue
    out_ct = OUT_DIR / ct.lower().replace('/', '_')
    out_ct.mkdir(parents=True, exist_ok=True)
    done = all((out_ct / f"{c.lower()}.pkl").exists() for c in CONDITIONS)
    if done and (out_ct / "metadata.pkl").exists():
        print(f"\n[{ct}] all conditions already done, skip"); continue
    print(f"\n{'='*60}\n[{ct}] {ct_data.n_obs} cells")
    all_genes = list(ct_data.var_names)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    obs_cond = ct_data.obs['condition']

    print("  Computing DEGs (Tumor vs Normal, no cap)...")
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

    meta_save = {'tf_genes': tf_in_data, 'all_deg_genes': all_deg_genes,
                 'n_total_genes': len(all_genes), 'n_deg': n_deg, 'n_tf': n_tf,
                 'method': 'deg_expanded_tt', 'case': 'HCC', 'cell_type': ct,
                 'conditions': CONDITIONS, 'baseline': BASELINE}
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

print(f"\n{'='*60}\nHCC DEG-Expanded complete: {(time.time()-t_start)/60:.1f} min")
print("DONE")
