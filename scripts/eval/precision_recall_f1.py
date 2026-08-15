#!/usr/bin/env python
"""Precision / Recall / F1 evaluation across BEELINE, HCT116, and K562.

Computes edge-level metrics at threshold 0.2 (and sweeps thresholds):
  - AUROC, AUPRC (threshold-free)
  - Precision, Recall, F1 at threshold=0.2
  - Precision-Recall curve summary

Uses TT specialist for all evaluations.
"""
import numpy as np
import pandas as pd
import torch, pickle, time, gc
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
sys_path = str(Path(__file__).resolve().parent.parent / "train")

import sys; sys.path.insert(0, sys_path)
from train_gt_g200_edge_v3 import (
    GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads,
    n_layers, dropout, sd_prob, device
)
from train_gt_g200_dir_specialist import (
    GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead
)

THRESHOLDS = [0.1, 0.2, 0.3, 0.5]

# ======= Load models =======
print("Loading models (TT specialist)...")
enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                n_layers=n_layers, sd_prob=0.0).to(device)
edge_head = EdgeHeadV3(d_model=d_model).to(device)
ckpt = torch.load(CKPT_ROOT / "main" / "edge_v3_seed0.pt",
                  map_location=device, weights_only=True)
enc.load_state_dict(ckpt['encoder']); edge_head.load_state_dict(ckpt['edge_head'])
enc.eval(); edge_head.eval()
print("  Ready.")


def compute_P_D(X):
    from sklearn.covariance import LedoitWolf
    n_cells, Gv = X.shape
    lw = LedoitWolf(); lw.fit(X)
    P = lw.precision_.astype(np.float32)
    if n_cells > 200:
        rng = np.random.RandomState(42)
        X = X[rng.choice(n_cells, 200, replace=False)]
        n_cells = 200
    X_c = X - X.mean(axis=0, keepdims=True)
    A = np.zeros((Gv, n_cells * n_cells), dtype=np.float64)
    for g in range(Gv):
        d = np.abs(X_c[:, g:g+1] - X_c[:, g:g+1].T)
        A[g] = (d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()).ravel()
    dcov2 = A @ A.T / (n_cells * n_cells)
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    D = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(D, 0); D = np.clip(D, 0, 1)
    return P, D.astype(np.float32)


@torch.no_grad()
def predict(P, D):
    P_t = torch.from_numpy(P).unsqueeze(0).to(device)
    D_t = torch.from_numpy(D).unsqueeze(0).to(device)
    h = enc(P_t, D_t)
    probs = torch.sigmoid(edge_head(h, P_t, D_t).float())[0].cpu().numpy()
    np.fill_diagonal(probs, 0)
    return probs


def eval_metrics(probs, gt, mask, label="", thresholds=THRESHOLDS):
    """Compute all metrics for a prediction vs ground truth."""
    probs_m = probs[mask]
    gt_m = gt[mask]

    # Threshold-free
    try: auroc = roc_auc_score(gt_m, probs_m)
    except: auroc = 0.5
    try: auprc = average_precision_score(gt_m, probs_m)
    except: auprc = 0.0

    results = {'label': label, 'AUROC': auroc, 'AUPRC': auprc, 'gt_pos': int(gt_m.sum())}

    # At each threshold
    for thr in thresholds:
        pred = (probs_m > thr).astype(int)
        tp = int(((pred == 1) & (gt_m == 1)).sum())
        fp = int(((pred == 1) & (gt_m == 0)).sum())
        fn = int(((pred == 0) & (gt_m == 1)).sum())
        tn = int(((pred == 0) & (gt_m == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        results[f'P@{thr}'] = prec
        results[f'R@{thr}'] = rec
        results[f'F1@{thr}'] = f1
        results[f'n_pred@{thr}'] = tp + fp

    return results


# ======= Part 1: BEELINE =======
print("\n" + "="*60)
print("PART 1: BEELINE Benchmark (TT specialist)")
print("="*60)

BEELINE_DIR = DATA_ROOT / "BEELINE"
DATASETS = {
    'mDC': (BEELINE_DIR / "Networks" / "mouse" / "mDC-ChIP-seq-network.csv",
            BEELINE_DIR / "mouse-tfs.csv"),
    'mHSC-E': (BEELINE_DIR / "Networks" / "mouse" / "mHSC-ChIP-seq-network.csv",
               BEELINE_DIR / "mouse-tfs.csv"),
    'mHSC-GM': (BEELINE_DIR / "Networks" / "mouse" / "mHSC-ChIP-seq-network.csv",
                BEELINE_DIR / "mouse-tfs.csv"),
    'mHSC-L': (BEELINE_DIR / "Networks" / "mouse" / "mHSC-ChIP-seq-network.csv",
               BEELINE_DIR / "mouse-tfs.csv"),
    'hESC': (BEELINE_DIR / "Networks" / "human" / "hESC-ChIP-seq-network.csv",
             BEELINE_DIR / "human-tfs.csv"),
    'hHep': (BEELINE_DIR / "Networks" / "human" / "HepG2-ChIP-seq-network.csv",
             BEELINE_DIR / "human-tfs.csv"),
}

beeline_results = []
for ds_name, (net_path, tf_path) in DATASETS.items():
    expr_path = BEELINE_DIR / "BEELINE-data" / "inputs" / "scRNA-Seq" / ds_name / "ExpressionData.csv"
    if not expr_path.exists():
        print(f"  [SKIP] {ds_name}: expression not found"); continue

    print(f"\n  {ds_name}...", end='', flush=True)
    t0 = time.time()

    # Load expression
    expr_df = pd.read_csv(expr_path, index_col=0)
    genes = list(expr_df.columns)
    X = expr_df.values.astype(np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Load GT
    tfs_df = pd.read_csv(tf_path, header=None)
    tf_set = set(str(t).upper() for t in tfs_df[0])
    tf_in = [g for g in genes if g.upper() in tf_set]
    net_df = pd.read_csv(net_path)
    nti = {g: i for i, g in enumerate(genes)}
    G_size = len(genes)
    gt = np.zeros((G_size, G_size), dtype=np.float32)
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in nti and t in nti:
            gt[nti[s], nti[t]] = 1.0

    # Standardize
    X_std = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    # Handle datasets larger than G=200: truncate to first G genes
    if G_size > G:
        genes = genes[:G]
        X_std = X_std[:, :G]
        G_size = G
        nti = {g: i for i, g in enumerate(genes)}

    # Build GT with correct gene-name matching
    gt = np.zeros((G_size, G_size), dtype=np.float32)
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in nti and t in nti:
            gt[nti[s], nti[t]] = 1.0

    # Pad to G=200 if needed
    if G_size < G:
        pad = G - G_size
        X_pad = np.zeros((X.shape[0], G), dtype=np.float32)
        X_pad[:, :G_size] = X_std
        P, D = compute_P_D(X_pad)
        P = P[:G_size, :G_size]
        D = D[:G_size, :G_size]
    else:
        P, D = compute_P_D(X_std)

    probs = predict(P, D)
    mask = ~np.eye(G_size, dtype=bool)

    r = eval_metrics(probs, gt, mask, label=ds_name)
    r['n_genes'] = G_size
    r['n_tfs'] = len(tf_in)
    beeline_results.append(r)
    print(f" AUROC={r['AUROC']:.4f} AUPRC={r['AUPRC']:.4f} "
          f"P@0.2={r['P@0.2']:.4f} R@0.2={r['R@0.2']:.4f} F1@0.2={r['F1@0.2']:.4f} "
          f"({time.time()-t0:.1f}s)")

    del X, X_std, P, D, probs; gc.collect()

df_bl = pd.DataFrame(beeline_results)
df_bl.to_csv(RESULT_ROOT / "2_benchmark_g200" / "beeline_prf1.csv", index=False)


# ======= Part 2: HCT116 Perturb-seq =======
print("\n" + "="*60)
print("PART 2: HCT116 Perturb-seq (TT specialist)")
print("="*60)

HCT_DIR = DATA_ROOT
TF_FILE = DATA_ROOT / "BEELINE" / "human-tfs.csv"

import scanpy as sc, h5py
from scipy import sparse
from scipy.stats import t as tdist

print("  Loading data...", end='', flush=True)
adata = sc.read_h5ad(HCT_DIR / "HCT116_filtered_dual_guide_cells.h5ad", backed='r')
all_genes_h = list(adata.var_names)
gene_to_idx_h = {g.upper(): i for i, g in enumerate(all_genes_h)}
guide = pd.DataFrame({'barcode': adata.obs.index, 'target': adata.obs['gene_target'].astype(str)})
gene_counts = guide['target'].value_counts()
perturbed_genes = set(gene_counts[gene_counts >= 50].index) - {'Non-Targeting'}
human_tfs = set(pd.read_csv(TF_FILE, header=None)[0].str.upper())
perturbed_tfs = sorted(perturbed_genes & human_tfs & set(gene_to_idx_h.keys()))
print(f" {len(all_genes_h)} genes, {len(perturbed_tfs)} perturbed TFs")

# Load controls
nt_barcodes = guide[guide['target'] == 'Non-Targeting']['barcode'].tolist()
control_barcodes = [b for b in nt_barcodes if b in {b: i for i, b in enumerate(adata.obs_names)}]
rng = np.random.RandomState(42)
if len(control_barcodes) > 5000:
    control_barcodes = rng.choice(control_barcodes, 5000, replace=False).tolist()

X_ctrl_sparse = adata[control_barcodes].X
X_ctrl = X_ctrl_sparse.toarray().astype(np.float32) if sparse.issparse(X_ctrl_sparse) else np.asarray(X_ctrl_sparse, dtype=np.float32)
ctrl_mean = X_ctrl.mean(axis=0)
ctrl_var = X_ctrl.var(axis=0, ddof=1)
n_ctrl = X_ctrl.shape[0]

# DE: Welch t-test per TF
print("  Computing DE ground truth...", end='', flush=True)
t0 = time.time()
gt_hct = np.zeros((len(all_genes_h), len(all_genes_h)), dtype=np.float32)
gene_expr_rate = (X_ctrl > 0).mean(axis=0)
valid_mask = gene_expr_rate >= 0.05
valid_idx = np.where(valid_mask)[0]

for tf in perturbed_tfs:
    tf_idx = gene_to_idx_h[tf]
    barcodes = guide[guide['target'] == tf]['barcode'].tolist()
    barcodes = [b for b in barcodes if b in {b: i for i, b in enumerate(adata.obs_names)}]
    if len(barcodes) < 50: continue
    X_tf_sparse = adata[barcodes].X
    X_tf = X_tf_sparse[:, valid_idx].toarray().astype(np.float32) if sparse.issparse(X_tf_sparse) else np.asarray(X_tf_sparse[:, valid_idx], dtype=np.float32)
    tf_mean = X_tf.mean(axis=0)
    tf_var = X_tf.var(axis=0, ddof=1)
    se = np.sqrt(np.maximum(tf_var, 1e-10) / len(barcodes) + np.maximum(ctrl_var[valid_idx], 1e-10) / n_ctrl)
    t_stats = (tf_mean - ctrl_mean[valid_idx]) / np.maximum(se, 1e-10)
    df_num = (tf_var / len(barcodes) + ctrl_var[valid_idx] / n_ctrl) ** 2
    df_den = np.maximum((tf_var / len(barcodes))**2 / max(len(barcodes)-1, 1) +
                        (ctrl_var[valid_idx] / n_ctrl)**2 / max(n_ctrl-1, 1), 1e-10)
    p_vals = 2 * tdist.sf(np.abs(t_stats), df_num / df_den)
    l2fc = np.log2(np.maximum(tf_mean, 0.1) / np.maximum(ctrl_mean[valid_idx], 0.1))
    sig = (p_vals < 0.01) & (np.abs(l2fc) >= 0.25)
    for gi in np.where(sig)[0]:
        gt_hct[tf_idx, valid_idx[gi]] = 1.0
print(f" {time.time()-t0:.1f}s, GT edges: {int(gt_hct.sum())}")

# Use top-50 TFs for fast eval
top_tfs = perturbed_tfs[:50]
sel_idx = [gene_to_idx_h[t] for t in top_tfs]
X_sel = X_ctrl[rng.choice(X_ctrl.shape[0], min(500, X_ctrl.shape[0]), replace=False)][:, sel_idx]
X_sel_std = (X_sel - X_sel.mean(0, keepdims=True)) / (X_sel.std(0, keepdims=True) + 1e-8)

# Pad to G=200
X_pad = np.zeros((X_sel_std.shape[0], G), dtype=np.float32)
X_pad[:, :len(sel_idx)] = X_sel_std
P, D = compute_P_D(X_pad)
probs_full = predict(P, D)[:len(sel_idx), :len(sel_idx)]

mask_h = ~np.eye(len(sel_idx), dtype=bool)
gt_sub = gt_hct[np.ix_(sel_idx, sel_idx)]
r_hct = eval_metrics(probs_full, gt_sub, mask_h, label='HCT116')
print(f"  HCT116 (top-50 TFs): AUROC={r_hct['AUROC']:.4f} AUPRC={r_hct['AUPRC']:.4f} "
      f"P@0.2={r_hct['P@0.2']:.4f} R@0.2={r_hct['R@0.2']:.4f} F1@0.2={r_hct['F1@0.2']:.4f}")

del adata, X_ctrl; gc.collect()


# ======= Part 3: K562 Perturb-seq =======
print("\n" + "="*60)
print("PART 3: K562 Perturb-seq (TT specialist)")
print("="*60)

print("  Loading data...", end='', flush=True)
import h5py as hf
with hf.File(DATA_ROOT / "K562_gwps_normalized_bulk_01.h5ad", 'r') as f:
    gene_cats = f["var"]["__categories"]["gene_name"][:]
    gene_idx_arr = f["var"]["gene_name"][:]
    gene_symbols = np.array([g.decode('utf-8').upper() for g in [gene_cats[i] for i in gene_idx_arr]])
    X_k562 = f['X'][:]
    cc = f['obs']['core_control'][:]

# Standardize
X_k562 = np.nan_to_num(np.asarray(X_k562, dtype=np.float32))
X_ctrl_k = X_k562[cc]
ctrl_mu = X_ctrl_k.mean(0, keepdims=True)
ctrl_std = np.maximum(X_ctrl_k.std(0, ddof=1, keepdims=True), 1e-8)
X_ctrl_k_std = ((X_ctrl_k - ctrl_mu) / ctrl_std).astype(np.float32)
del X_k562; gc.collect()
print(f" {len(gene_symbols)} genes, {int(cc.sum())} controls")

# Load GT
rt = pd.read_parquet(DATA_ROOT / "truthseq_k562" / "replogle_knockdown_effects.parquet")
rt = rt[rt['z_score'].abs() >= 2.0]
gene_to_idx_k = {g: i for i, g in enumerate(gene_symbols)}
gt_k562 = np.zeros((len(gene_symbols), len(gene_symbols)), dtype=np.float32)
for _, r in rt.iterrows():
    s = str(r['knocked_down_gene']).upper()
    t = str(r['affected_gene']).upper()
    if s in gene_to_idx_k and t in gene_to_idx_k:
        gt_k562[gene_to_idx_k[s], gene_to_idx_k[t]] = 1.0
print(f"  GT edges: {int(gt_k562.sum()):,}")

# Use top-50 TFs for fast eval
rt_tf = rt[rt['knocked_down_gene'].str.upper().isin(human_tfs)]
tf_counts = rt_tf['knocked_down_gene'].str.upper().value_counts()
top_tfs_k = list(tf_counts.head(50).index)
sel_idx_k = [gene_to_idx_k[t] for t in top_tfs_k if t in gene_to_idx_k]

X_sub_k = X_ctrl_k_std[rng.choice(X_ctrl_k_std.shape[0], min(500, X_ctrl_k_std.shape[0]), replace=False)][:, sel_idx_k]
X_pad_k = np.zeros((X_sub_k.shape[0], G), dtype=np.float32)
X_pad_k[:, :len(sel_idx_k)] = X_sub_k
P_k, D_k = compute_P_D(X_pad_k)
probs_k = predict(P_k, D_k)[:len(sel_idx_k), :len(sel_idx_k)]

mask_k = ~np.eye(len(sel_idx_k), dtype=bool)
gt_sub_k = gt_k562[np.ix_(sel_idx_k, sel_idx_k)]
r_k562 = eval_metrics(probs_k, gt_sub_k, mask_k, label='K562')
print(f"  K562 (top-50 TFs): AUROC={r_k562['AUROC']:.4f} AUPRC={r_k562['AUPRC']:.4f} "
      f"P@0.2={r_k562['P@0.2']:.4f} R@0.2={r_k562['R@0.2']:.4f} F1@0.2={r_k562['F1@0.2']:.4f}")


# ======= Summary =======
print("\n" + "="*60)
print("SUMMARY: Precision / Recall / F1 @ threshold=0.2")
print("="*60)

all_results = [df_bl] if len(df_bl) > 0 else []
all_results.append(pd.DataFrame([r_hct]))
all_results.append(pd.DataFrame([r_k562]))
df_all = pd.concat(all_results, ignore_index=True)

print(f"\n{'Dataset':<12s} {'AUROC':>7s} {'AUPRC':>7s} {'Prec@.2':>8s} {'Rec@.2':>8s} {'F1@.2':>7s} {'#pred':>7s} {'GT+':>7s}")
print("-" * 75)
for _, r in df_all.iterrows():
    print(f"  {r['label']:<12s} {r['AUROC']:>7.4f} {r['AUPRC']:>7.4f} "
          f"{r['P@0.2']:>8.4f} {r['R@0.2']:>8.4f} {r['F1@0.2']:>7.4f} "
          f"{r.get('n_pred@0.2', 0):>7d} {r['gt_pos']:>7d}")

# BEELINE mean
if len(df_bl) > 0:
    print(f"\n  BEELINE mean: AUROC={df_bl['AUROC'].mean():.4f} AUPRC={df_bl['AUPRC'].mean():.4f} "
          f"P@0.2={df_bl['P@0.2'].mean():.4f} R@0.2={df_bl['R@0.2'].mean():.4f} "
          f"F1@0.2={df_bl['F1@0.2'].mean():.4f}")

df_all.to_csv(RESULT_ROOT / "3_full_gene" / "precision_recall_f1.csv", index=False)
print(f"\nSaved: results/3_full_gene/precision_recall_f1.csv")
print("DONE")
