#!/usr/bin/env python
"""NAFLD Cell-type-specific GRN Analysis using BoYue (Shared Gene Set).

Key design: Uses the SAME 200 genes as the tissue-level analysis across
Hepatocyte and LSEC, enabling direct comparison of GRN edge counts.

For each cell type (Hepatocyte, LSEC) × condition (5 NAFLD stages):
  1. Subset cells, subsample to max_cells
  2. Extract expression for the shared 200 genes
  3. Compute P (Ledoit-Wolf) + D (distance correlation)
  4. BoYue inference: edge_v3 + dir_specialist_tf_non_tf (4-seed ensemble)

Usage:
  python scripts/eval_nafld_celltype.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.covariance import LedoitWolf
import pickle, time, sys, gc, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3
from train_gt_g200_dir_v4 import AsymmetricDirHead

# -- Config ----------------------------------------------
G = 200; d_model = 512; n_heads = 8; n_layers = 8
# Data dir: set BOYUE_DATA env var to point to downloaded datasets
import os
_data_root = Path(os.environ.get("BOYUE_DATA", PROJECT_ROOT / "data_external"))
DATA_DIR = _data_root / "GSE202379"
TISSUE_RESULT = DATA_DIR / "results" / "nafld_detailed.pkl"
EDGE_CKPT = PROJECT_ROOT / "checkpoints" / "main"
DIR_CKPT  = PROJECT_ROOT / "checkpoints" / "main"
OUTPUT = DATA_DIR / "results" / "nafld_celltype_shared.pkl"
MAX_CELLS = 1500
CONDITIONS = ['Healthy', 'NAFLD', 'NASH', 'Cirrhosis', 'EndStage']
CELL_TYPES = ['Hepatocyte', 'LSEC']
device = torch.device('cuda')
rng = np.random.RandomState(42)

# -- Helpers ---------------------------------------------

def fast_dcor(X):
    """Distance correlation matrix (G×G)."""
    C, Gv = X.shape
    if C > 200:
        X = X[rng.choice(C, 200, replace=False)]; C = 200
    X_c = X - X.mean(axis=0, keepdims=True)
    A = np.zeros((Gv, C * C), dtype=np.float64)
    for i in range(Gv):
        d = np.abs(X_c[:, i:i+1] - X_c[:, i:i+1].T)
        A[i] = (d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()).ravel()
    dcov2 = A @ A.T / (C * C)
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    dcor = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(dcor, 1); dcor = np.clip(dcor, 0, 1)
    del A; gc.collect()
    return dcor.astype(np.float32)

# -- Main ------------------------------------------------
print("=" * 80)
print("NAFLD Cell-type-specific GRN (Shared Gene Set)")
print("=" * 80)

# 1. Load shared gene set from tissue-level analysis
print("\n[1] Loading shared gene set from tissue-level analysis...")
with open(TISSUE_RESULT, 'rb') as f:
    tissue = pickle.load(f)
shared_genes = tissue['selected_genes'][:G]  # 200 genes
shared_tfs = tissue['top_tfs'][:50]
shared_tf_set = set(shared_tfs)
shared_gene_set = set(shared_genes)
print(f"  {len(shared_genes)} genes ({len(shared_tfs)} TFs + {G-len(shared_tfs)} targets)")
print(f"  Top TFs: {shared_tfs[:10]}")

# 2. Load cell-type annotated data
print("\n[2] Loading cell-type annotated data...")
import anndata as ad
adata = ad.read_h5ad(DATA_DIR / "processed" / "GSE202379_celltype.h5ad")
print(f"  Shape: {adata.shape}")

# Check gene availability
missing = shared_gene_set - set(adata.var_names)
if missing:
    print(f"  WARNING: {len(missing)} genes not in expression matrix: {list(missing)[:10]}")
    valid_genes = [g for g in shared_genes if g in adata.var_names]
    print(f"  Using {len(valid_genes)}/{len(shared_genes)} genes")
else:
    valid_genes = shared_genes
    print(f"  All {len(shared_genes)} genes found")

# 3. Load BoYue models (4-seed ensemble)
print("\n[3] Loading BoYue models...")
encoders, edge_heads, dir_heads = [], [], []
for s in range(4):
    eckpt = torch.load(EDGE_CKPT / f"edge_v3_seed{s}.pt", map_location=device, weights_only=True)
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    enc.load_state_dict(eckpt['encoder']); enc.eval()
    ehead = EdgeHeadV3(d_model=d_model, d_k=128).to(device)
    ehead.load_state_dict(eckpt['edge_head']); ehead.eval()
    dckpt = torch.load(DIR_CKPT / f"seed_{s}" / "best.pt", map_location=device, weights_only=True)
    dhead = AsymmetricDirHead(d_model=d_model).to(device)
    dhead.load_state_dict(dckpt['dir_head']); dhead.eval()
    encoders.append(enc); edge_heads.append(ehead); dir_heads.append(dhead)
print(f"  Loaded 4-seed ensemble")

def run_bo(P_mat, D_mat):
    """4-seed ensemble inference: edge_prob, dir_score."""
    Pt = torch.from_numpy(P_mat).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D_mat).unsqueeze(0).to(device)
    eprobs, dir_s = None, None
    for enc, eh, dh in zip(encoders, edge_heads, dir_heads):
        with torch.no_grad():
            h = enc(Pt, Dt)
            el = eh(h, Pt, Dt); dl = dh(h, Pt, Dt)
        ep = torch.sigmoid(el.float()).cpu().squeeze(0)
        dp = torch.sigmoid(dl.float()).cpu().squeeze(0).squeeze(-1)
        eprobs = ep if eprobs is None else eprobs + ep
        dir_s = dp if dir_s is None else dir_s + dp
    return (eprobs / len(encoders)).numpy(), (dir_s / len(encoders)).numpy()

# 4. Per cell-type × condition inference
print("\n[4] Cell-type-specific GRN inference...")
all_results = {}

for ct in CELL_TYPES:
    print(f"\n{'-' * 60}")
    print(f"  Cell type: {ct}")
    ct_data = adata[adata.obs['cell_type'] == ct]
    print(f"    Total cells: {ct_data.shape[0]}")

    ct_results = {'results': {}, 'tf_genes': shared_tfs}

    for cond in CONDITIONS:
        t0 = time.time()
        sub = ct_data[ct_data.obs['condition'] == cond]
        n_cells = sub.shape[0]
        if n_cells < 50:
            print(f"    {cond}: {n_cells} cells (SKIP, < 50)")
            ct_results['results'][cond] = None
            continue

        # Subsample cells
        if n_cells > MAX_CELLS:
            idx = rng.choice(n_cells, MAX_CELLS, replace=False)
            X = sub.X[idx].toarray() if hasattr(sub.X, 'toarray') else sub.X[idx]
            actual_cells = MAX_CELLS
        else:
            X = sub.X.toarray() if hasattr(sub.X, 'toarray') else sub.X
            actual_cells = n_cells

        # Extract gene expression (cells × genes -> cells × 200)
        gene_to_idx = {g: i for i, g in enumerate(sub.var_names)}
        gene_idx = np.array([gene_to_idx[g] for g in valid_genes])
        X_sub = X[:, gene_idx].astype(np.float64)

        # Standardize
        X_sub = (X_sub - X_sub.mean(0, keepdims=True)) / (X_sub.std(0, keepdims=True) + 1e-8)

        # Compute P + D
        lw = LedoitWolf()
        lw.fit(X_sub)
        P_mat = lw.precision_.astype(np.float32)
        D_mat = fast_dcor(X_sub)

        # BoYue inference
        edge_prob, dir_score = run_bo(P_mat, D_mat)

        # Compute direction (0/1/-1): 1 = A->B, -1 = B->A, 0 = uncertain
        direction = np.zeros((G, G), dtype=np.int8)
        direction[edge_prob > 0.5] = 1  # default direction
        # For edge pairs where reverse direction is more likely
        reverse = (dir_score < 0.5) & (edge_prob > 0.5)
        direction[reverse] = -1

        ct_results['results'][cond] = {
            'genes': valid_genes,
            'edge_prob': edge_prob,
            'dir_score': dir_score,
            'direction': direction,
            'n_cells': actual_cells
        }

        n_edges = (edge_prob > 0.2).sum()
        print(f"    {cond}: {actual_cells} cells, {n_edges} edges (p>0.2) [{time.time()-t0:.0f}s]")
        gc.collect()

    all_results[ct] = ct_results

# 5. Comparison table
print(f"\n{'=' * 80}")
print("COMPARISON: Tissue-level vs Hepatocyte vs LSEC")
print(f"{'=' * 80}")
print(f"{'Stage':<12} {'Tissue':>8} {'Hepatocyte':>10} {'LSEC':>8}")
print("-" * 42)

for cond in CONDITIONS:
    # Tissue-level edges
    tr = tissue['results'].get(cond)
    tissue_e = (tr['edge_prob'] > 0.2).sum() if tr is not None else '-'

    # Hepatocyte edges
    hep_r = all_results['Hepatocyte']['results'].get(cond)
    hep_e = (hep_r['edge_prob'] > 0.2).sum() if hep_r is not None else '-'

    # LSEC edges
    lsec_r = all_results['LSEC']['results'].get(cond)
    lsec_e = (lsec_r['edge_prob'] > 0.2).sum() if lsec_r is not None else '-'

    print(f"  {cond:<10s}  {str(tissue_e):>8}  {str(hep_e):>10}  {str(lsec_e):>8}")

# 6. Save
print(f"\n[5] Saving to {OUTPUT}...")
with open(OUTPUT, 'wb') as f:
    pickle.dump({'tissue_genes': shared_genes, 'celltype_results': all_results}, f)
print("  Done!")

elapsed = time.time()
print(f"\nTotal time: {elapsed/60:.1f} min")
