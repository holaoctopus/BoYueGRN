#!/usr/bin/env python
"""HCT116 G=200 single-window cross-specialist evaluation (memory-efficient).

Based on eval_perturb_seq_direction.py but with full cross-specialist output.

Usage:
  python scripts/eval/hct116_cross_specialist_g200.py
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as tdist
import torch
from pathlib import Path
import sys, time, gc, json
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
from train_gt_g200_edge_v3 import (
    GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers,
    dropout, sd_prob, device
)
from train_gt_g200_dir_specialist import (
    GraphTransformerEncoderV3 as DirEncoder,
    AsymmetricDirHead
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import DATA_ROOT, CKPT_ROOT, RESULT_ROOT
TF_FILE = DATA_ROOT / "BEELINE" / "human-tfs.csv"

MIN_CELLS_PER_GENE = 50
P_THRESHOLD = 0.01
LOG2FC_THRESHOLD = 0.25

print(f"G={G}, device={device}")
t0_total = time.time()
rng = np.random.RandomState(42)

# ===========================================================
# Step 1: Open anndata backed + cross-reference
# ===========================================================
print("\n[1/4] Loading anndata in backed mode + cross-referencing...")
import scanpy as sc

adata = sc.read_h5ad(DATA_ROOT / "HCT116_filtered_dual_guide_cells.h5ad", backed='r')
all_genes = list(adata.var_names)
all_genes_upper = [g.upper() for g in all_genes]
n_genes = len(all_genes)
gene_to_idx = {g.upper(): i for i, g in enumerate(all_genes)}
print(f"  Genes: {n_genes}, Cells: {adata.n_obs}")

guide = pd.DataFrame({'cell_barcode': adata.obs.index, 'gene_target': adata.obs['gene_target'].astype(str)})
gene_counts = guide['gene_target'].value_counts()
perturbed_genes = set(gene_counts[gene_counts >= MIN_CELLS_PER_GENE].index)
perturbed_genes.discard('Non-Targeting')

human_tfs = set(pd.read_csv(TF_FILE, header=None)[0].str.upper())
perturbed_tfs = sorted(perturbed_genes & human_tfs & set(gene_to_idx.keys()))
print(f"  Perturbed TFs in expression matrix: {len(perturbed_tfs)}")

# Map barcodes to indices
barcode_to_idx = {b: i for i, b in enumerate(adata.obs_names)}
cell_to_target = dict(zip(guide['cell_barcode'], guide['gene_target']))

# ===========================================================
# Step 2: Load control cells + per-TF cell lists
# ===========================================================
print("\n[2/4] Loading controls and indexing TF cells...")

non_targeting_barcodes = guide[guide['gene_target'] == 'Non-Targeting']['cell_barcode'].tolist()
control_barcodes = [c for c in non_targeting_barcodes if c in barcode_to_idx]
if len(control_barcodes) > 5000:
    control_barcodes = rng.choice(control_barcodes, 5000, replace=False).tolist()

# Build per-TF barcode lists
tf_barcodes = {tf: [] for tf in perturbed_tfs}
for barcode, target in cell_to_target.items():
    if target in perturbed_tfs and barcode in barcode_to_idx:
        tf_barcodes[target].append(barcode)

print(f"  TFs with >= {MIN_CELLS_PER_GENE} cells: "
      f"{sum(1 for v in tf_barcodes.values() if len(v) >= MIN_CELLS_PER_GENE)}")

# Load control expression from backed anndata (5000 cells, sparse->dense)
t_load = time.time()
X_ctrl_sparse = adata[control_barcodes].X
if sparse.issparse(X_ctrl_sparse):
    X_ctrl = X_ctrl_sparse.toarray().astype(np.float32)
else:
    X_ctrl = X_ctrl_sparse.astype(np.float32)
del X_ctrl_sparse; gc.collect()

print(f"  Control expression: {X_ctrl.shape} in {time.time()-t_load:.1f}s, "
      f"{X_ctrl.nbytes/1e6:.1f} MB")

# ===========================================================
# Step 3: DE per TF
# ===========================================================
print("\n[3/4] Differential expression per TF...")

ctrl_mean = X_ctrl.mean(axis=0)
ctrl_var = X_ctrl.var(axis=0, ddof=1)
n_ctrl = X_ctrl.shape[0]

# Gene filtering: at least 5% non-zero in controls
gene_expr_rate = (X_ctrl > 0).mean(axis=0)
valid_mask = gene_expr_rate >= 0.05
valid_indices = np.where(valid_mask)[0]
valid_gene_names = [all_genes_upper[i] for i in valid_indices]
n_valid = len(valid_indices)
print(f"  Genes with >=5% detection rate: {n_valid}")

ctrl_mean_v = ctrl_mean[valid_indices]
ctrl_var_v = ctrl_var[valid_indices]

gt_edges = {}
t_de = time.time()
tf_processed = 0

for tf_idx, tf in enumerate(perturbed_tfs):
    barcodes = tf_barcodes[tf]
    n_tf = len(barcodes)
    if n_tf < MIN_CELLS_PER_GENE:
        continue

    # Load TF cells from backed anndata
    X_tf_sparse = adata[barcodes].X
    if sparse.issparse(X_tf_sparse):
        X_tf = X_tf_sparse[:, valid_indices].toarray().astype(np.float32)
    else:
        X_tf = X_tf_sparse[:, valid_indices].astype(np.float32)
    del X_tf_sparse

    tf_mean = X_tf.mean(axis=0)
    tf_var = X_tf.var(axis=0, ddof=1)
    tf_var = np.maximum(tf_var, 1e-10)

    # Welch t-test
    se = np.sqrt(tf_var / n_tf + ctrl_var_v / n_ctrl)
    se = np.maximum(se, 1e-10)
    t_stats = (tf_mean - ctrl_mean_v) / se

    df_num = (tf_var / n_tf + ctrl_var_v / n_ctrl) ** 2
    df_den = ((tf_var / n_tf) ** 2) / max(n_tf-1, 1) + ((ctrl_var_v / n_ctrl) ** 2) / max(n_ctrl-1, 1)
    df_den = np.maximum(df_den, 1e-10)
    df_approx = df_num / df_den
    p_values = 2 * tdist.sf(np.abs(t_stats), df_approx)

    log2fc = np.log2(np.maximum(tf_mean, 0.1) / np.maximum(ctrl_mean_v, 0.1))

    significant = (p_values < P_THRESHOLD) & (np.abs(log2fc) >= LOG2FC_THRESHOLD)

    hits = []
    for gi in np.where(significant)[0]:
        hits.append((valid_gene_names[gi], float(p_values[gi]), float(log2fc[gi])))

    if len(hits) > 0:
        hits.sort(key=lambda x: x[1])
        gt_edges[tf] = hits[:100]

    del X_tf

    tf_processed += 1
    if tf_processed % 50 == 0:
        print(f"  [{tf_processed}/{len(perturbed_tfs)}] {tf}: {len(hits)} DE genes, "
              f"total edges: {sum(len(v) for v in gt_edges.values())}")

print(f"  DE done in {time.time()-t_de:.1f}s: {len(gt_edges)} TFs with edges")

# ===========================================================
# Step 4: Select genes and run model
# ===========================================================
print("\n[4/4] Selecting genes and running model...")

top_tfs = sorted(gt_edges.keys(), key=lambda t: len(gt_edges[t]), reverse=True)[:50]
sel_genes = set(top_tfs)
for tf in top_tfs:
    for g, _, _ in gt_edges[tf]:
        sel_genes.add(g)
sel_genes = sorted(sel_genes)[:G]
print(f"  Selected genes: {len(sel_genes)}")

# Map to expression matrix indices
sel_idx = [gene_to_idx[g.upper()] for g in sel_genes]
G_act = len(sel_genes)

# Extract control expression for selected genes
X_model = X_ctrl[:, sel_idx]
del X_ctrl; gc.collect()

# Compute P and D
n_cells = X_model.shape[0]
lw = LedoitWolf()
lw.fit(X_model)
P_mat = lw.precision_.astype(np.float32)
print(f"  P done: {P_mat.shape}")

def fast_dcor(X):
    C, Gv = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    A_flat = np.zeros((Gv, C * C), dtype=np.float64)
    for i in range(Gv):
        xi = X_c[:, i]
        d = np.abs(xi[:, None].astype(np.float64) - xi[None, :].astype(np.float64))
        A = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
        A_flat[i] = A.ravel()
        del d, A
    dcov2 = A_flat @ A_flat.T / (C * C)
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    dcor = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(dcor, 1.0)
    dcor = np.clip(dcor, 0, 1)
    del A_flat, dcov2, dvar, dvp
    gc.collect()
    return dcor.astype(np.float32)

D_mat = fast_dcor(X_model)
print(f"  D done: {D_mat.shape}")

# Pad to G
if G_act < G:
    pad = G - G_act
    P_full = np.pad(P_mat, ((0, pad), (0, pad)), constant_values=0)
    D_full = np.pad(D_mat, ((0, pad), (0, pad)), constant_values=0)
else:
    P_full, D_full = P_mat, D_mat

P_t = torch.from_numpy(P_full).unsqueeze(0).to(device)
D_t = torch.from_numpy(D_full).unsqueeze(0).to(device)

# Load models
print("  Loading models...")
enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, dropout=dropout, sd_prob=sd_prob).to(device)
edge_head = EdgeHeadV3(d_model=d_model, d_k=128).to(device)
ckpt = torch.load(CKPT_ROOT / "main" / "edge_v3_seed0.pt",
                  map_location=device, weights_only=True)
enc.load_state_dict(ckpt['encoder'])
edge_head.load_state_dict(ckpt['edge_head'])
enc.eval(); edge_head.eval()

dir_enc_nt = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, sd_prob=sd_prob).to(device)
dir_head_nt = AsymmetricDirHead(d_model=d_model).to(device)
dckpt = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_non_tf_seed0.pt",
                   map_location=device, weights_only=True)
dir_enc_nt.load_state_dict(dckpt['encoder'])
dir_head_nt.load_state_dict(dckpt['dir_head'])
dir_enc_nt.eval(); dir_head_nt.eval()

dir_enc_tt = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, sd_prob=sd_prob).to(device)
dir_head_tt = AsymmetricDirHead(d_model=d_model).to(device)
dckpt2 = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt",
                    map_location=device, weights_only=True)
dir_enc_tt.load_state_dict(dckpt2['encoder'])
dir_head_tt.load_state_dict(dckpt2['dir_head'])
dir_enc_tt.eval(); dir_head_tt.eval()

# Inference
with torch.no_grad():
    h_edge = enc(P_t, D_t)
    edge_p = torch.sigmoid(edge_head(h_edge, P_t, D_t).float())[0].cpu().numpy()

    h_nt = dir_enc_nt(P_t, D_t)
    dir_p_nt = torch.sigmoid(dir_head_nt(h_nt, P_t, D_t).float())[0].cpu().numpy()

    h_tt = dir_enc_tt(P_t, D_t)
    dir_p_tt = torch.sigmoid(dir_head_tt(h_tt, P_t, D_t).float())[0].cpu().numpy()

# Evaluate cross-specialist
gene_to_gidx = {g: i for i, g in enumerate(sel_genes)}
tf_model_indices = set(gene_to_gidx[t] for t in top_tfs if t in gene_to_gidx)

results = {k: {'correct': 0, 'total': 0} for k in
           ['nt_on_nontf', 'nt_on_tftf', 'tt_on_nontf', 'tt_on_tftf',
            'nt_all', 'tt_all', 'matched_all']}

for tf, edges in gt_edges.items():
    if tf not in gene_to_gidx:
        continue
    tf_idx = gene_to_gidx[tf]
    for target, pval, l2fc in edges:
        if target not in gene_to_gidx:
            continue
        tgt_idx = gene_to_gidx[target]
        if tf_idx == tgt_idx:
            continue

        is_tf_tf = (tf_idx in tf_model_indices) and (tgt_idx in tf_model_indices)

        nt_pred = 1 if dir_p_nt[tf_idx, tgt_idx] > dir_p_nt[tgt_idx, tf_idx] else 0
        tt_pred = 1 if dir_p_tt[tf_idx, tgt_idx] > dir_p_tt[tgt_idx, tf_idx] else 0
        matched_pred = tt_pred if is_tf_tf else nt_pred

        if is_tf_tf:
            results['nt_on_tftf']['correct'] += nt_pred
            results['nt_on_tftf']['total'] += 1
            results['tt_on_tftf']['correct'] += tt_pred
            results['tt_on_tftf']['total'] += 1
        else:
            results['nt_on_nontf']['correct'] += nt_pred
            results['nt_on_nontf']['total'] += 1
            results['tt_on_nontf']['correct'] += tt_pred
            results['tt_on_nontf']['total'] += 1

        results['nt_all']['correct'] += nt_pred
        results['nt_all']['total'] += 1
        results['tt_all']['correct'] += tt_pred
        results['tt_all']['total'] += 1
        results['matched_all']['correct'] += matched_pred
        results['matched_all']['total'] += 1

# Print results
print("\n" + "=" * 60)
print("HCT116 G=200 CROSS-SPECIALIST RESULTS")
print("=" * 60)
out = {}
for k, v in results.items():
    acc = v['correct'] / max(v['total'], 1)
    out[k] = acc
    print(f"  {k:20s}: {acc:.4f} ({v['correct']}/{v['total']})")

# Save
out['dataset'] = 'hct116'
out['coverage'] = 'G=200'
out['n_eval'] = results['nt_all']['total']
out_path = RESULT_ROOT / "3_full_gene" / "hct116_g200_cross_specialist.json"
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Total time: {time.time() - t0_total:.1f}s")
