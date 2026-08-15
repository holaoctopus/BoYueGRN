#!/usr/bin/env python
"""DEG-Expanded + TT Specialist core module (shared across case studies).

Reusable functions: model loading, DEG computation (Wilcoxon stage-vs-baseline,
no cap), sliding-window inference, fast dCor. Each case script imports these
and supplies its own data loading + config.

Strategy (from NAFLD optimal):
  - DEG selection: ALL DEGs per cell type (no 350 cap)
  - Direction: TT specialist (outperforms NT on both edge types)
  - Sliding window: 5 TFs + 195 targets + target rotation
  - Merge: edge_prob=max, dir_score=mean across windows
"""
import numpy as np
import pandas as pd
import torch, time, gc, os, sys
from pathlib import Path
from scipy.stats import ranksums
from sklearn.covariance import LedoitWolf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Use _config for env-var-driven paths (BOYUE_ROOT, BOYUE_DATA, BOYUE_CKPT)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT


def load_known_tfs(species='human'):
    csv = DATA_ROOT / "BEELINE" / f"{species}-tfs.csv"
    tfs = set(pd.read_csv(csv, header=None)[0].str.upper().tolist())
    return {t for t in tfs if t and len(t) > 1}


def load_models(device):
    """Load edge_v3 (seed_0) + TT direction specialist."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
    from train_gt_g200_edge_v3 import (
        GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads,
        n_layers, dropout, sd_prob)
    from train_gt_g200_dir_specialist import (
        GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead)

    # Submission checkpoint layout: main/edge_v3_seed0.pt, main/dir_specialist_tf_tf_seed0.pt
    edge_ckpt = CKPT_ROOT / "main" / "edge_v3_seed0.pt"
    dir_ckpt = CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt"

    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                    n_layers=n_layers, sd_prob=0.0).to(device)
    edge_head = EdgeHeadV3(d_model=d_model).to(device)
    st = torch.load(edge_ckpt, map_location=device, weights_only=True)
    enc.load_state_dict(st['encoder']); edge_head.load_state_dict(st['edge_head'])
    enc.eval(); edge_head.eval()

    dir_enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, sd_prob=0.0).to(device)
    dir_head = AsymmetricDirHead(d_model=d_model).to(device)
    dst = torch.load(dir_ckpt, map_location=device, weights_only=True)
    dir_enc.load_state_dict(dst['encoder']); dir_head.load_state_dict(dst['dir_head'])
    dir_enc.eval(); dir_head.eval()

    G_val = G
    return enc, edge_head, dir_enc, dir_head, G_val


def compute_degs(expr_ct, all_genes, obs_condition, deg_comparisons,
                 baseline, min_cells=50, fdr_thr=0.05, l2fc_thr=0.5,
                 known_tfs=None):
    """Wilcoxon stage-vs-baseline, FDR<0.05, |log2FC|>0.5, UNION (no cap).

    expr_ct: (n_cells, n_genes) dense array
    obs_condition: pd.Series aligned with expr_ct rows
    deg_comparisons: list of (stage, baseline) tuples
    Returns deg_df with columns [gene, max_abs_log2fc, min_fdr, n_stages_sig, is_tf].
    """
    base_mask = (obs_condition == baseline).values
    n_base = int(base_mask.sum())
    if n_base < min_cells:
        return pd.DataFrame()
    per_stage = []
    for stage, _ in deg_comparisons:
        stage_mask = (obs_condition == stage).values
        n_stage = int(stage_mask.sum())
        if n_stage < min_cells:
            continue
        X_b = expr_ct[base_mask]
        X_s = expr_ct[stage_mask]
        mean_b = X_b.mean(axis=0) + 1e-6
        mean_s = X_s.mean(axis=0) + 1e-6
        log2fc = np.log2(mean_s / mean_b)
        p_vals = np.ones(len(all_genes))
        for gi in range(len(all_genes)):
            if np.all(X_b[:, gi] == X_b[0, gi]) and np.all(X_s[:, gi] == X_s[0, gi]):
                continue
            _, p = ranksums(X_s[:, gi], X_b[:, gi])
            p_vals[gi] = p
        valid = ~np.isnan(p_vals)
        fdr = np.ones(len(p_vals))
        if valid.sum() > 0:
            _, fdr_v, _, _ = multipletests(p_vals[valid], method='fdr_bh')
            fdr[valid] = fdr_v
        per_stage.append({'log2fc': log2fc, 'fdr': fdr})
    if not per_stage:
        return pd.DataFrame()
    n_genes = len(all_genes)
    is_deg = np.zeros(n_genes, dtype=bool)
    max_abs_l2fc = np.zeros(n_genes)
    min_fdr = np.ones(n_genes)
    n_stages_sig = np.zeros(n_genes, dtype=int)
    for r in per_stage:
        sig = (r['fdr'] < fdr_thr) & (np.abs(r['log2fc']) > l2fc_thr)
        is_deg |= sig
        max_abs_l2fc = np.maximum(max_abs_l2fc, np.abs(r['log2fc']))
        min_fdr = np.minimum(min_fdr, r['fdr'])
        n_stages_sig += sig.astype(int)
    deg_df = pd.DataFrame({
        'gene': all_genes, 'max_abs_log2fc': max_abs_l2fc,
        'min_fdr': min_fdr, 'n_stages_sig': n_stages_sig,
        'is_tf': [g in known_tfs for g in all_genes] if known_tfs else False,
    })
    return deg_df[is_deg].copy()


def fast_dcor(X):
    """Distance correlation matrix (G x G)."""
    C, Gv = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    A_flat = np.zeros((Gv, C * C), dtype=np.float64)
    for i in range(Gv):
        xi = X_c[:, i]
        d = np.abs(xi[:, None].astype(np.float64) - xi[None, :].astype(np.float64))
        A = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
        A_flat[i] = A.ravel()
    dcov2 = A_flat @ A_flat.T / (C * C)
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    dcor = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(dcor, 1.0); dcor = np.clip(dcor, 0, 1)
    del A_flat, dcov2, dvar, dvp; gc.collect()
    return dcor.astype(np.float32)


def make_infer_window(enc, edge_head, dir_enc, dir_head, device):
    @torch.no_grad()
    def infer_window(P_t, D_t):
        h_e = enc(P_t, D_t)
        ep = torch.sigmoid(edge_head(h_e, P_t, D_t).float())[0].cpu().numpy()
        h_d = dir_enc(P_t, D_t)
        ds = torch.sigmoid(dir_head(h_d, P_t, D_t).float())[0].cpu().numpy()
        np.fill_diagonal(ep, 0); np.fill_diagonal(ds, 0)
        return ep, ds
    return infer_window


def expanded_deg_grn(X, all_deg_genes, tf_genes_in_data, gene_to_idx,
                     enc, edge_head, dir_enc, dir_head, device, G_val,
                     top_k=50, tf_per_win=5, max_cells=500, max_tfs=150, verbose=False):
    """Sliding window GRN on ALL DEG genes (no cap). Returns (ep_max, ds_avg, n_windows).

    max_tfs: cap on number of TFs used (top-N by caller's ordering, which is
             |log2FC| descending). Keeps window count manageable for cell types
             with hundreds of DEG TFs (e.g. 514 -> 150 -> ~600 windows vs ~1854).
             Set to None for no cap.
    """
    infer_window = make_infer_window(enc, edge_head, dir_enc, dir_head, device)
    if max_tfs is not None and len(tf_genes_in_data) > max_tfs:
        if verbose:
            print(f"    Capping TFs: {len(tf_genes_in_data)} -> {max_tfs} (top by |log2FC|)")
        tf_genes_in_data = list(tf_genes_in_data)[:max_tfs]
    tf_set = set(tf_genes_in_data)
    n_genes = len(all_deg_genes)
    deg2local = {g: i for i, g in enumerate(all_deg_genes)}

    # Pre-screen: correlation on DEG genes only
    rng_ps = np.random.RandomState(42)
    n_sub = min(500, X.shape[0])
    deg_global_idx = [gene_to_idx[g] for g in all_deg_genes if g in gene_to_idx]
    X_deg = X[:, deg_global_idx]
    X_sub_ps = X_deg[rng_ps.choice(X_deg.shape[0], n_sub, replace=False)]
    if verbose:
        print(f"    Pre-screen (n={n_sub}, {n_genes} DEGs)...", end='', flush=True)
    corr_mat = np.corrcoef(X_sub_ps.T).astype(np.float32)
    if verbose:
        print(" done", flush=True)

    candidates = {}
    for tf in tf_genes_in_data:
        if tf not in deg2local:
            continue
        ti = deg2local[tf]
        corrs = np.abs(corr_mat[ti])
        sorted_idx = np.argsort(corrs)[::-1]
        cands = []
        for idx in sorted_idx:
            if idx == ti:
                continue
            g = all_deg_genes[idx]
            if g in tf_set:
                continue
            cands.append((g, float(corrs[idx])))
            if len(cands) >= top_k:
                break
        candidates[tf] = cands
    del corr_mat; gc.collect()

    tf_list = sorted(candidates.keys(), key=lambda t: len(candidates[t]), reverse=True)
    target_set = set()
    for cs in candidates.values():
        for g, _ in cs:
            target_set.add(g)
    targets = list(target_set)
    target_slots = G_val - tf_per_win
    windows = []
    for bs in range(0, len(tf_list), tf_per_win):
        batch_tfs = tf_list[bs:bs + tf_per_win]
        if len(batch_tfs) < 2:
            continue
        batch_targets = targets[:]
        n_rot = -(-len(batch_targets) // target_slots) if len(batch_targets) > target_slots else 1
        for rot in range(n_rot):
            rs = rot * target_slots
            rot_targets = batch_targets[rs:rs + target_slots]
            wg = list(batch_tfs) + list(rot_targets)
            ws = set(wg)
            for g in all_deg_genes:
                if len(wg) >= G_val:
                    break
                if g not in ws:
                    wg.append(g); ws.add(g)
            windows.append(wg[:G_val])
    if verbose:
        print(f"    Windows: {len(windows)}")

    ep_max = np.zeros((n_genes, n_genes), dtype=np.float32)
    ds_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
    ds_cnt = np.zeros((n_genes, n_genes), dtype=np.float32)

    rng = np.random.RandomState(42)
    X_sub = X_deg[rng.choice(X_deg.shape[0], min(max_cells, X_deg.shape[0]), replace=False)]

    for wi, wgenes in enumerate(windows):
        if verbose and wi % 50 == 0:
            print(f"      Window {wi+1}/{len(windows)}...", flush=True)
        win_local = [deg2local[g] for g in wgenes if g in deg2local]
        win_names = [g for g in wgenes if g in deg2local]
        if len(win_local) < 10:
            continue
        X_win = X_sub[:, win_local]
        mu = X_win.mean(0, keepdims=True)
        std = np.maximum(X_win.std(0, keepdims=True), 1e-8)
        X_std = (X_win - mu) / std
        try:
            lw = LedoitWolf(); lw.fit(X_std)
            P_mat = lw.precision_.astype(np.float32)
            D_mat = fast_dcor(X_std)
        except Exception:
            continue
        G_act = len(win_local)
        if G_act < G_val:
            pad = G_val - G_act
            P_full = np.pad(P_mat, ((0, pad), (0, pad)))
            D_full = np.pad(D_mat, ((0, pad), (0, pad)))
        else:
            P_full, D_full = P_mat, D_mat
        P_t = torch.from_numpy(P_full).unsqueeze(0).to(device)
        D_t = torch.from_numpy(D_full).unsqueeze(0).to(device)
        ep_w, ds_w = infer_window(P_t, D_t)
        for i, gi in enumerate(win_names):
            li = deg2local[gi]
            for j, gj in enumerate(win_names):
                lj = deg2local[gj]
                ep_max[li, lj] = max(ep_max[li, lj], ep_w[i, j])
                ds_sum[li, lj] += ds_w[i, j]
                ds_cnt[li, lj] += 1
        del P_t, D_t; gc.collect()

    valid = ds_cnt > 0
    ds_avg = np.zeros_like(ds_sum)
    ds_avg[valid] = ds_sum[valid] / ds_cnt[valid]
    return ep_max, ds_avg, len(windows)
