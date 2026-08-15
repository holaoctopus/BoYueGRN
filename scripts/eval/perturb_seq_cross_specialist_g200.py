#!/usr/bin/env python
"""HCT116/K562 G=200 single-window cross-specialist evaluation.

Evaluates BOTH direction specialists on BOTH edge types:
  - NT specialist on TF->non-TF (matched)
  - NT specialist on TF->TF (cross)
  - TT specialist on TF->non-TF (cross)
  - TT specialist on TF->TF (matched)
  - Each specialist on ALL edges

Usage:
  python scripts/eval/perturb_seq_cross_specialist_g200.py --dataset hct116
  python scripts/eval/perturb_seq_cross_specialist_g200.py --dataset k562
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as tdist
import torch
from pathlib import Path
import sys, time, gc, argparse, json
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['hct116', 'k562'])
args = parser.parse_args()

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import DATA_ROOT, CKPT_ROOT, RESULT_ROOT
TF_FILE = DATA_ROOT / "BEELINE" / "human-tfs.csv"

print(f"G={G}, device={device}, dataset={args.dataset}")
t0_total = time.time()
rng = np.random.RandomState(42)

# ===========================================================
# Load data
# ===========================================================
print("\n[1/4] Loading data...")

if args.dataset == 'hct116':
    import scanpy as sc
    adata = sc.read_h5ad(DATA_ROOT / "HCT116_filtered_dual_guide_cells.h5ad", backed='r')
    all_genes = list(adata.var_names)
    all_genes_upper = [g.upper() for g in all_genes]
    n_genes = len(all_genes)
    gene_to_idx = {g.upper(): i for i, g in enumerate(all_genes)}
    print(f"  HCT116: {n_genes} genes")
else:
    import h5py as hf
    with hf.File(DATA_ROOT / "K562_gwps_normalized_bulk_01.h5ad", 'r') as f:
        gc_arr = f["var"]["__categories"]["gene_name"][:]
        gi_arr = f["var"]["gene_name"][:]
        gene_syms = np.array([g.decode('utf-8').upper() for g in [gc_arr[i] for i in gi_arr]])
        X_k = f['X'][:]
        cc = f['obs']['core_control'][:]
    X_k = np.nan_to_num(np.asarray(X_k, dtype=np.float32))
    X_ctrl = X_k[cc]
    mu = X_ctrl.mean(0, keepdims=True)
    std = np.maximum(X_ctrl.std(0, ddof=1, keepdims=True), 1e-8)
    X_ctrl = ((X_ctrl - mu) / std).astype(np.float32)
    del X_k
    gc.collect()
    all_genes = list(gene_syms)
    all_genes_upper = list(gene_syms)
    n_genes = len(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    print(f"  K562: {n_genes} genes")

# Load TF list
tfs = set(pd.read_csv(TF_FILE, header=None)[0].str.upper())
print(f"  Human TFs: {len(tfs)}")

# ===========================================================
# Build GT and select genes
# ===========================================================
print("\n[2/4] Building ground truth...")

if args.dataset == 'hct116':
    # HCT116: use Perturb-seq DE results
    MIN_CELLS_PER_GENE = 50
    P_THRESHOLD = 0.01
    LOG2FC_THRESHOLD = 0.25

    # Cross-reference perturbed TFs
    guide = pd.DataFrame({'cell_barcode': adata.obs.index, 'gene_target': adata.obs['gene_target'].astype(str)})
    gene_counts = guide['gene_target'].value_counts()
    perturbed_genes = set(gene_counts[gene_counts >= MIN_CELLS_PER_GENE].index)
    perturbed_genes.discard('Non-Targeting')
    target_genes = sorted(perturbed_genes & tfs & set(gene_to_idx.keys()))
    print(f"  Perturbed TFs: {len(target_genes)}")

    # For each TF, run Welch t-test vs controls
    gt_edges = {}
    ctrl_mask = guide['gene_target'] == 'Non-Targeting'
    X_ctrl = adata[ctrl_mask].X
    if sparse.issparse(X_ctrl):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = np.asarray(X_ctrl, dtype=np.float32)
    ctrl_mu = X_ctrl.mean(0)
    ctrl_var = X_ctrl.var(0, ddof=1)
    n_ctrl = X_ctrl.shape[0]

    for tf in target_genes:
        tf_mask = guide['gene_target'] == tf
        n_cells = tf_mask.sum()
        if n_cells < MIN_CELLS_PER_GENE:
            continue
        X_tf = adata[tf_mask].X
        if sparse.issparse(X_tf):
            X_tf = X_tf.toarray()
        X_tf = np.asarray(X_tf, dtype=np.float32)
        tf_mu = X_tf.mean(0)
        tf_var = X_tf.var(0, ddof=1)
        n_tf = X_tf.shape[0]

        # Welch t-test
        se = np.sqrt(tf_var / n_tf + ctrl_var / n_ctrl + 1e-10)
        t_stat = (tf_mu - ctrl_mu) / se
        df = (tf_var / n_tf + ctrl_var / n_ctrl + 1e-10) ** 2 / \
             ((tf_var / n_tf) ** 2 / (n_tf - 1) + (ctrl_var / n_ctrl) ** 2 / (n_ctrl - 1) + 1e-10)
        pvals = 2 * tdist.sf(np.abs(t_stat), df)
        l2fc = np.log2((tf_mu + 1e-6) / (ctrl_mu + 1e-6))

        edges = []
        for i, g in enumerate(all_genes):
            if pvals[i] < P_THRESHOLD and abs(l2fc[i]) > LOG2FC_THRESHOLD:
                edges.append((g, pvals[i], l2fc[i]))
        if edges:
            gt_edges[tf] = edges

    print(f"  TFs with edges: {len(gt_edges)}")

    # Select top TFs and genes
    top_tfs = sorted(gt_edges.keys(), key=lambda t: len(gt_edges[t]), reverse=True)[:50]
    sel_genes = set(top_tfs)
    for tf in top_tfs:
        for g, _, _ in gt_edges[tf]:
            sel_genes.add(g)
    sel_genes = sorted(sel_genes)[:G]
    print(f"  Selected genes: {len(sel_genes)}")

    # Load control expression for selected genes
    sel_idx = [all_genes_upper.index(g.upper()) for g in sel_genes]
    X_model = X_ctrl[:, sel_idx]
    G_act = len(sel_genes)

else:
    # K562
    rt = pd.read_parquet(DATA_ROOT / "truthseq_k562" / "replogle_knockdown_effects.parquet")
    rt = rt[rt['z_score'].abs() >= 2.0]
    g2i = {g: i for i, g in enumerate(all_genes)}
    gt_k = np.zeros((n_genes, n_genes), dtype=np.float32)
    for _, r in rt.iterrows():
        s, t = str(r['knocked_down_gene']).upper(), str(r['affected_gene']).upper()
        if s in g2i and t in g2i:
            gt_k[g2i[s], g2i[t]] = 1.0

    human_tfs = set(pd.read_csv(TF_FILE, header=None)[0].str.upper())
    top50 = list(rt[rt['knocked_down_gene'].str.upper().isin(human_tfs)]['knocked_down_gene'].str.upper().value_counts().head(50).index)
    sel = [g2i[t] for t in top50 if t in g2i]
    sel_genes = [all_genes[i] for i in sel]
    G_act = len(sel_genes)

    # Build GT edges dict for consistency
    gt_edges = {}
    for tf in top50:
        if tf not in g2i:
            continue
        tf_idx = g2i[tf]
        edges = []
        for j, g in enumerate(all_genes):
            if gt_k[tf_idx, j] > 0:
                edges.append((g, 0.0, 1.0))  # dummy pval/l2fc
        if edges:
            gt_edges[tf] = edges

    # Subsample control cells
    rng = np.random.RandomState(42)
    X_sub = X_ctrl[rng.choice(X_ctrl.shape[0], min(500, X_ctrl.shape[0]), replace=False)][:, sel]
    X_model = X_sub

print(f"  Model genes: {G_act}")

# ===========================================================
# Compute P and D
# ===========================================================
print("\n[3/4] Computing P and D...")

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

# ===========================================================
# Load models and predict
# ===========================================================
print("\n[4/4] Loading models and evaluating...")

# Edge model
enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, dropout=dropout, sd_prob=sd_prob).to(device)
edge_head = EdgeHeadV3(d_model=d_model, d_k=128).to(device)
ckpt = torch.load(CKPT_ROOT / "main" / "edge_v3_seed0.pt",
                  map_location=device, weights_only=True)
enc.load_state_dict(ckpt['encoder'])
edge_head.load_state_dict(ckpt['edge_head'])
enc.eval(); edge_head.eval()

# NT specialist
dir_enc_nt = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, sd_prob=sd_prob).to(device)
dir_head_nt = AsymmetricDirHead(d_model=d_model).to(device)
dckpt = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_non_tf_seed0.pt",
                   map_location=device, weights_only=True)
dir_enc_nt.load_state_dict(dckpt['encoder'])
dir_head_nt.load_state_dict(dckpt['dir_head'])
dir_enc_nt.eval(); dir_head_nt.eval()

# TT specialist
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
tf_set = set(top_tfs)
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

        # NT specialist prediction
        nt_pred = 1 if dir_p_nt[tf_idx, tgt_idx] > dir_p_nt[tgt_idx, tf_idx] else 0
        # TT specialist prediction
        tt_pred = 1 if dir_p_tt[tf_idx, tgt_idx] > dir_p_tt[tgt_idx, tf_idx] else 0
        # Matched routing
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
print(f"{args.dataset.upper()} G=200 CROSS-SPECIALIST RESULTS")
print("=" * 60)
out = {}
for k, v in results.items():
    acc = v['correct'] / max(v['total'], 1)
    out[k] = acc
    print(f"  {k:20s}: {acc:.4f} ({v['correct']}/{v['total']})")

# Save
out['dataset'] = args.dataset
out['coverage'] = 'G=200'
out['n_eval'] = results['nt_all']['total']
out_path = RESULT_ROOT / "3_full_gene" / f"{args.dataset}_g200_cross_specialist.json"
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Total time: {time.time() - t0_total:.1f}s")
