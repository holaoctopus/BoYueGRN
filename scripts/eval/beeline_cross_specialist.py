#!/usr/bin/env python
"""BEELINE full-gene cross-specialist evaluation.

Evaluates BOTH direction specialists on BOTH edge types for BEELINE datasets:
  - NT specialist on TF->non-TF (matched)
  - NT specialist on TF->TF (cross)
  - TT specialist on TF->non-TF (cross)
  - TT specialist on TF->TF (matched)
  - Each specialist on ALL edges

Uses the SAME multi-window TF-focused strategy as full_evaluation_matrix.py.

Usage:
  python scripts/eval/beeline_cross_specialist.py
"""
import numpy as np
import pandas as pd
import torch
import time
import gc
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
from train_gt_g200_edge_v3 import (
    GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers,
    dropout, sd_prob, device
)
from train_gt_g200_dir_specialist import (
    GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import DATA_ROOT, CKPT_ROOT, RESULT_ROOT

TF_PER_WINDOW = 5
TOP_K_TARGETS = 50
THRESHOLD = 0.2

# ======= Load models =======
print("Loading edge models (4-seed ensemble)...")
encoders, edge_heads = [], []
for seed in range(4):
    p = CKPT_ROOT / "main" / f"edge_v3_seed{seed}.pt"
    if not p.exists():
        continue
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                    n_layers=n_layers, sd_prob=0.0).to(device)
    head = EdgeHeadV3(d_model=d_model).to(device)
    ck = torch.load(p, map_location=device, weights_only=True)
    enc.load_state_dict(ck['encoder'])
    head.load_state_dict(ck['edge_head'])
    enc.eval(); head.eval()
    encoders.append(enc); edge_heads.append(head)
print(f"  Edge: {len(encoders)} seeds")

print("Loading NT direction specialist...")
dir_enc_nt = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, sd_prob=sd_prob).to(device)
dir_head_nt = AsymmetricDirHead(d_model=d_model).to(device)
dstate = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_non_tf_seed0.pt",
                    map_location=device, weights_only=True)
dir_enc_nt.load_state_dict(dstate['encoder'])
dir_head_nt.load_state_dict(dstate['dir_head'])
dir_enc_nt.eval(); dir_head_nt.eval()
print("  NT direction ready.")

print("Loading TT direction specialist...")
dir_enc_tt = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, sd_prob=sd_prob).to(device)
dir_head_tt = AsymmetricDirHead(d_model=d_model).to(device)
dstate2 = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt",
                     map_location=device, weights_only=True)
dir_enc_tt.load_state_dict(dstate2['encoder'])
dir_head_tt.load_state_dict(dstate2['dir_head'])
dir_enc_tt.eval(); dir_head_tt.eval()
print("  TT direction ready.")


def compute_P_D(X):
    n_cells, Gv = X.shape
    lw = LedoitWolf(); lw.fit(X)
    P = lw.precision_.astype(np.float32)
    if n_cells > 200:
        rng = np.random.RandomState(42)
        X = X[rng.choice(n_cells, 200, replace=False)]
    X_c = X - X.mean(axis=0, keepdims=True)
    A = np.zeros((Gv, X.shape[0]**2), dtype=np.float64)
    for g in range(Gv):
        d = np.abs(X_c[:, g:g+1] - X_c[:, g:g+1].T)
        A[g] = (d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()).ravel()
    dcov2 = A @ A.T / X.shape[0]**2
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    D = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(D, 0); D = np.clip(D, 0, 1)
    return P, D.astype(np.float32)


@torch.no_grad()
def predict_edge(P, D):
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    probs = None
    for enc, head in zip(encoders, edge_heads):
        h = enc(Pt, Dt)
        p = torch.sigmoid(head(h, Pt, Dt).float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    probs = (probs / len(encoders)).numpy()
    np.fill_diagonal(probs, 0)
    return np.maximum(probs, probs.T).astype(np.float32)


@torch.no_grad()
def predict_dir_both(P, D):
    """Return both NT and TT direction predictions."""
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    h_nt = dir_enc_nt(Pt, Dt)
    ds_nt = torch.sigmoid(dir_head_nt(h_nt, Pt, Dt).float()).squeeze(0).cpu().numpy()
    h_tt = dir_enc_tt(Pt, Dt)
    ds_tt = torch.sigmoid(dir_head_tt(h_tt, Pt, Dt).float()).squeeze(0).cpu().numpy()
    np.fill_diagonal(ds_nt, 0)
    np.fill_diagonal(ds_tt, 0)
    return ds_nt.astype(np.float32), ds_tt.astype(np.float32)


def build_tf_windows(genes, tf_names, X_std, top_k=TOP_K_TARGETS):
    g2i = {g: i for i, g in enumerate(genes)}
    tf_in = [g for g in genes if g.upper() in set(t.upper() for t in tf_names)]
    if len(tf_in) < 2:
        return [], tf_in
    rng = np.random.RandomState(42)
    n_sub = min(500, X_std.shape[0])
    corr = np.corrcoef(X_std[rng.choice(X_std.shape[0], n_sub, replace=False)].T).astype(np.float32)
    cands = {}
    for tf in tf_in:
        ti = g2i[tf]
        co = np.abs(corr[ti]); si = np.argsort(co)[::-1]
        c = []
        for idx in si:
            if idx == ti:
                continue
            g = genes[idx]
            if g.upper() in set(t.upper() for t in tf_names):
                continue
            c.append(g)
            if len(c) >= top_k:
                break
        cands[tf] = c
    del corr; gc.collect()
    tf_list = sorted(cands.keys(), key=lambda t: len(cands[t]), reverse=True)
    tgt_set = set()
    for cs in cands.values():
        for g in cs:
            tgt_set.add(g)
    targets = list(tgt_set)
    ts = G - TF_PER_WINDOW
    windows = []
    for bs in range(0, len(tf_list), TF_PER_WINDOW):
        bt = tf_list[bs:bs+TF_PER_WINDOW]
        if len(bt) < 2:
            continue
        btg = targets[:]
        nr = -(-len(btg) // ts) if len(btg) > ts else 1
        for rot in range(nr):
            rs = rot * ts
            rt = btg[rs:rs+ts]
            wg = list(bt) + list(rt)
            ws = set(wg)
            v = X_std.var(axis=0); fo = np.argsort(v)[::-1]
            for idx in fo:
                if len(wg) >= G:
                    break
                g = genes[idx]
                if g not in ws:
                    wg.append(g); ws.add(g)
            windows.append(wg[:G])
    return windows, tf_in


def eval_dir_acc_by_type(ep, ds_nt, ds_tt, gt, tf_set, thr=0.2):
    """Evaluate direction accuracy split by edge type.

    Returns dict with:
      - nt_on_nontf, nt_on_tftf, tt_on_nontf, tt_on_tftf
      - nt_all, tt_all, matched_all
      - high-conf versions (edge_prob > thr)
    """
    pred_mask = ep > thr
    gt_mask = gt > 0
    both = pred_mask & gt_mask
    n_both = int(both.sum())
    if n_both == 0:
        return {k: 0.0 for k in ['nt_on_nontf', 'nt_on_tftf', 'tt_on_nontf', 'tt_on_tftf',
                                  'nt_all', 'tt_all', 'matched_all',
                                  'nt_on_nontf_hc', 'nt_on_tftf_hc', 'tt_on_nontf_hc', 'tt_on_tftf_hc',
                                  'nt_all_hc', 'tt_all_hc', 'matched_all_hc', 'n_eval']}

    # Build TF index set for edge type classification
    tf_idx_set = set()
    for i, g in enumerate(genes):
        if g.upper() in tf_set:
            tf_idx_set.add(i)

    results = {k: {'correct': 0, 'total': 0} for k in
               ['nt_on_nontf', 'nt_on_tftf', 'tt_on_nontf', 'tt_on_tftf',
                'nt_all', 'tt_all', 'matched_all',
                'nt_on_nontf_hc', 'nt_on_tftf_hc', 'tt_on_nontf_hc', 'tt_on_tftf_hc',
                'nt_all_hc', 'tt_all_hc', 'matched_all_hc']}

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if not both[i, j]:
                continue
            is_tf_tf = (i in tf_idx_set) and (j in tf_idx_set)
            is_hc = ep[i, j] > thr

            # GT direction: gt[i,j]>0 means i->j
            # Model direction: ds[i,j] > ds[j,i] means i->j
            nt_correct = 1 if ds_nt[i, j] >= ds_nt[j, i] else 0
            tt_correct = 1 if ds_tt[i, j] >= ds_tt[j, i] else 0
            matched_correct = tt_correct if is_tf_tf else nt_correct

            # All edges
            if is_tf_tf:
                results['nt_on_tftf']['correct'] += nt_correct
                results['nt_on_tftf']['total'] += 1
                results['tt_on_tftf']['correct'] += tt_correct
                results['tt_on_tftf']['total'] += 1
            else:
                results['nt_on_nontf']['correct'] += nt_correct
                results['nt_on_nontf']['total'] += 1
                results['tt_on_nontf']['correct'] += tt_correct
                results['tt_on_nontf']['total'] += 1

            results['nt_all']['correct'] += nt_correct
            results['nt_all']['total'] += 1
            results['tt_all']['correct'] += tt_correct
            results['tt_all']['total'] += 1
            results['matched_all']['correct'] += matched_correct
            results['matched_all']['total'] += 1

            # High-conf
            if is_hc:
                if is_tf_tf:
                    results['nt_on_tftf_hc']['correct'] += nt_correct
                    results['nt_on_tftf_hc']['total'] += 1
                    results['tt_on_tftf_hc']['correct'] += tt_correct
                    results['tt_on_tftf_hc']['total'] += 1
                else:
                    results['nt_on_nontf_hc']['correct'] += nt_correct
                    results['nt_on_nontf_hc']['total'] += 1
                    results['tt_on_nontf_hc']['correct'] += tt_correct
                    results['tt_on_nontf_hc']['total'] += 1

                results['nt_all_hc']['correct'] += nt_correct
                results['nt_all_hc']['total'] += 1
                results['tt_all_hc']['correct'] += tt_correct
                results['tt_all_hc']['total'] += 1
                results['matched_all_hc']['correct'] += matched_correct
                results['matched_all_hc']['total'] += 1

    out = {}
    for k, v in results.items():
        out[k] = v['correct'] / max(v['total'], 1)
    out['n_eval'] = n_both
    return out


# ======= Main: BEELINE =======
print("\n" + "=" * 70)
print("BEELINE CROSS-SPECIALIST EVALUATION (full-gene multi-window)")
print("=" * 70)

BEELINE_DIR = DATA_ROOT / "BEELINE"
BL = {
    'mDC': ("mouse/mDC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mDC/ExpressionData.csv"),
    'mHSC-E': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv"),
    'mHSC-GM': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv"),
    'mHSC-L': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv"),
    'hESC': ("human/hESC-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv"),
    'hHep': ("human/HepG2-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hHep/ExpressionData.csv"),
}

all_results = []
for ds, (net_rel, tf_rel, expr_rel) in BL.items():
    expr_path = BEELINE_DIR / expr_rel
    if not expr_path.exists():
        print(f"  [SKIP] {ds}: expression not found")
        continue

    print(f"\n  {ds}:")
    t0 = time.time()

    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes = list(expr_df.columns)
    X = expr_df.values.astype(np.float32)
    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = list(tf_df[0])
    tf_set = set(t.upper() for t in tf_names)

    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    gt = np.zeros((len(genes), len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in gene_to_idx and t in gene_to_idx:
            gt[gene_to_idx[s], gene_to_idx[t]] = 1.0

    X_std = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    # Multi-window only
    windows, tf_in = build_tf_windows(genes, tf_names, X_std)
    if not windows:
        print(f"    no windows")
        continue

    ep_full = np.zeros((len(genes), len(genes)), dtype=np.float32)
    ds_nt_sum = np.zeros((len(genes), len(genes)), dtype=np.float32)
    ds_tt_sum = np.zeros((len(genes), len(genes)), dtype=np.float32)
    ds_cnt = np.zeros((len(genes), len(genes)), dtype=np.float32)

    for wi, wg in enumerate(windows):
        if wi % 50 == 0:
            print(f"    W{wi+1}/{len(windows)}...", end='', flush=True)
        widx = [gene_to_idx[g] for g in wg if g in gene_to_idx]
        wn = [g for g in wg if g in gene_to_idx]
        nw = len(widx)
        if nw < 10:
            continue
        Xw = X_std[:, widx]
        Xp = np.zeros((Xw.shape[0], G), dtype=np.float32)
        Xp[:, :nw] = Xw
        P, D = compute_P_D(Xp)
        ep_w = predict_edge(P, D)[:nw, :nw]
        ds_nt_w, ds_tt_w = predict_dir_both(P, D)
        ds_nt_w = ds_nt_w[:nw, :nw]
        ds_tt_w = ds_tt_w[:nw, :nw]
        for i, gi in enumerate(widx):
            for j, gj in enumerate(widx):
                ep_full[gi, gj] = max(ep_full[gi, gj], ep_w[i, j])
                ds_nt_sum[gi, gj] += ds_nt_w[i, j]
                ds_tt_sum[gi, gj] += ds_tt_w[i, j]
                ds_cnt[gi, gj] += 1

    valid = ds_cnt > 0
    ds_nt_full = np.zeros_like(ds_nt_sum)
    ds_tt_full = np.zeros_like(ds_tt_sum)
    ds_nt_full[valid] = ds_nt_sum[valid] / ds_cnt[valid]
    ds_tt_full[valid] = ds_tt_sum[valid] / ds_cnt[valid]

    dt = time.time() - t0

    # Evaluate
    r = eval_dir_acc_by_type(ep_full, ds_nt_full, ds_tt_full, gt, tf_set, thr=THRESHOLD)
    r['dataset'] = ds
    r['n_windows'] = len(windows)
    r['time_s'] = dt
    all_results.append(r)

    print(f"    NT->nonTF={r['nt_on_nontf']:.4f} NT->TF={r['nt_on_tftf']:.4f} "
          f"TT->nonTF={r['tt_on_nontf']:.4f} TT->TF={r['tt_on_tftf']:.4f} "
          f"NTall={r['nt_all']:.4f} TTall={r['tt_all']:.4f} ({dt:.1f}s)")

    del X, X_std, ep_full, ds_nt_full, ds_tt_full, ds_cnt, gt
    gc.collect()

# ======= Save =======
df = pd.DataFrame(all_results)
print("\n" + "=" * 70)
print("BEELINE CROSS-SPECIALIST SUMMARY")
print("=" * 70)
cols = ['dataset', 'nt_on_nontf', 'nt_on_tftf', 'tt_on_nontf', 'tt_on_tftf',
        'nt_all', 'tt_all', 'matched_all', 'n_eval', 'n_windows']
print(df[cols].to_string(index=False))

out_path = RESULT_ROOT / "3_full_gene" / "beeline_cross_specialist.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("DONE")
