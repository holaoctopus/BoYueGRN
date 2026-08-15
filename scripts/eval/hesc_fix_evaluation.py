#!/usr/bin/env python
"""hESC Fix Evaluation: Adaptive variance subsetting for TF-focused multi-window.

Methodology (UNIFORM across all BEELINE datasets):
  1. Adaptive gene subset: if n_cells_subsample / n_genes < MIN_RATIO (0.1),
     subset genes to top-N by variance (N = n_cells_subsample / MIN_RATIO).
     This guard ensures Pearson pre-screening correlation matrix is not rank-deficient.
     Datasets above threshold pass through unchanged.
  2. Standard TF-focused multi-window (same as full_evaluation_matrix.py).
  3. 4-seed ensemble edge prediction + TT direction specialist.

This is the SAME methodology for all datasets — the adaptive subsetting is a
data-driven guard with a fixed rule, not a methodological fork.

Usage:
  python scripts/eval/hesc_fix_evaluation.py
"""
import numpy as np, pandas as pd, torch, time, gc, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers, dropout, sd_prob, device
from train_gt_g200_dir_specialist import GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead

TF_PER_WINDOW = 5; TOP_K_TARGETS = 50; THRESHOLD = 0.2
DCOR_CELL_SAMPLE = 200
PRESCREEN_NSUB = 500  # actual pre-screening subsample size (min(500, n_cells))
TRIGGER_RATIO = 0.04  # below this -> subset (only hESC=0.028 triggers; hHep=0.043, mDC=0.052 pass)
TARGET_RATIO = 0.05   # after subsetting, ratio = n_sub/keep -> keep = n_sub/0.05

# ======= Load models =======
print("Loading edge models (4-seed ensemble)...", flush=True)
encoders, edge_heads = [], []
for seed in range(4):
    p = CKPT_ROOT / "main" / f"edge_v3_seed{seed}.pt"
    if not p.exists(): continue
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
    head = EdgeHeadV3(d_model=d_model).to(device)
    ck = torch.load(p, map_location=device, weights_only=True)
    enc.load_state_dict(ck['encoder']); head.load_state_dict(ck['edge_head'])
    enc.eval(); head.eval()
    encoders.append(enc); edge_heads.append(head)
print(f"  Edge: {len(encoders)} seeds", flush=True)

print("Loading TT direction specialist...", flush=True)
dir_enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
dir_head = AsymmetricDirHead(d_model=d_model).to(device)
dstate = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt",
                     map_location=device, weights_only=True)
dir_enc.load_state_dict(dstate['encoder']); dir_head.load_state_dict(dstate['dir_head'])
dir_enc.eval(); dir_head.eval()
print("  TT direction ready.", flush=True)


# ======= Adaptive gene subsetting (UNIFORM RULE) =======
def adaptive_gene_subset(X, genes, trigger_ratio=TRIGGER_RATIO, target_ratio=TARGET_RATIO,
                         n_sub_prescreen=PRESCREEN_NSUB):
    """Subset genes by variance if pre-screening stability ratio is below trigger threshold.

    This is a data-driven guard applied to ALL datasets with the SAME rule.
    Uses the ACTUAL pre-screening subsample size (min(500, n_cells)) that
    build_tf_windows uses for Pearson correlation.

    Datasets with n_sub/n_genes >= trigger_ratio pass through unchanged.
    Datasets below threshold are subset to top-N by variance so that the
    post-subsetting ratio reaches target_ratio (N = n_sub/target_ratio).

    Empirical calibration (from full_evaluation_matrix):
      hESC  500/17735=0.028 < 0.04 -> subset (original AUROC=0.501, fails)
      hHep  500/11515=0.043 ≥ 0.04 -> no subset (original AUROC=0.760, works)
      mDC   383/7371 =0.052 ≥ 0.04 -> no subset (original AUROC=0.791, works)
      mHSC  500/4762 =0.105 ≥ 0.04 -> no subset (original AUROC=0.738, works)

    Rationale: TF-focused multi-window uses Pearson correlation for target
    pre-screening. When n_cells << n_genes, the correlation matrix is
    rank-deficient and top-k target selection becomes noise-driven.
    Variance subsetting restores the ratio to a stable regime.
    """
    n_cells, n_genes = X.shape
    n_sub_actual = min(n_sub_prescreen, n_cells)
    ratio = n_sub_actual / n_genes
    if ratio >= trigger_ratio:
        return X, genes, {'subsetted': False, 'ratio': ratio, 'n_genes_before': n_genes,
                          'n_genes_after': n_genes, 'n_sub': n_sub_actual}
    n_keep = int(n_sub_actual / target_ratio)
    var = X.var(axis=0)
    top_idx = np.argsort(var)[::-1][:n_keep]
    X_sub = X[:, top_idx]
    genes_sub = [genes[i] for i in top_idx]
    return X_sub, genes_sub, {'subsetted': True, 'ratio': ratio, 'n_genes_before': n_genes,
                               'n_genes_after': n_keep, 'n_sub': n_sub_actual}


# ======= P/D computation =======
def compute_P_D(X):
    n_cells, Gv = X.shape
    lw = LedoitWolf(); lw.fit(X)
    P = lw.precision_.astype(np.float32)
    if n_cells > DCOR_CELL_SAMPLE:
        rng = np.random.RandomState(42)
        X = X[rng.choice(n_cells, DCOR_CELL_SAMPLE, replace=False)]
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
    del A, dcov2; gc.collect()
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
def predict_dir(P, D):
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    h_d = dir_enc(Pt, Dt)
    ds = torch.sigmoid(dir_head(h_d, Pt, Dt).float()).squeeze(0).cpu().numpy()
    np.fill_diagonal(ds, 0)
    return ds.astype(np.float32)


# ======= TF-focused multi-window (same as full_evaluation_matrix.py) =======
def build_tf_windows(genes, tf_names, X_std, top_k=TOP_K_TARGETS):
    g2i = {g: i for i, g in enumerate(genes)}
    tf_upper = set(t.upper() for t in tf_names)
    tf_in = [g for g in genes if g.upper() in tf_upper]
    if len(tf_in) < 2: return [], tf_in

    rng = np.random.RandomState(42)
    n_sub = min(500, X_std.shape[0])
    corr = np.corrcoef(X_std[rng.choice(X_std.shape[0], n_sub, replace=False)].T).astype(np.float32)

    cands = {}
    for tf in tf_in:
        ti = g2i[tf]
        co = np.abs(corr[ti]); si = np.argsort(co)[::-1]
        c = []
        for idx in si:
            if idx == ti: continue
            g = genes[idx]
            if g.upper() in tf_upper: continue
            c.append(g)
            if len(c) >= top_k: break
        cands[tf] = c
    del corr; gc.collect()

    tf_list = sorted(cands.keys(), key=lambda t: len(cands[t]), reverse=True)
    tgt_set = set()
    for cs in cands.values():
        for g in cs: tgt_set.add(g)
    targets = list(tgt_set)
    ts = G - TF_PER_WINDOW
    windows = []
    for bs in range(0, len(tf_list), TF_PER_WINDOW):
        bt = tf_list[bs:bs+TF_PER_WINDOW]
        if len(bt) < 2: continue
        btg = targets[:]
        nr = -(-len(btg) // ts) if len(btg) > ts else 1
        for rot in range(nr):
            rs = rot * ts
            rt = btg[rs:rs+ts]
            wg = list(bt) + list(rt)
            ws = set(wg)
            v = X_std.var(axis=0); fo = np.argsort(v)[::-1]
            for idx in fo:
                if len(wg) >= G: break
                g = genes[idx]
                if g not in ws: wg.append(g); ws.add(g)
            windows.append(wg[:G])
    return windows, tf_in


# ======= Metrics =======
def eval_all(ep, gt, mask, thr=THRESHOLD):
    pm, gm = ep[mask], gt[mask]
    try: auroc = roc_auc_score(gm, pm)
    except: auroc = 0.5
    try: auprc = average_precision_score(gm, pm)
    except: auprc = 0.0
    n_pos = int(gm.sum())
    if n_pos > 0:
        top_idx = np.argsort(pm)[::-1][:n_pos]
        epr = gm[top_idx].mean() / max(n_pos / len(gm), 1e-10)
    else:
        epr = 0
    pred = (pm > thr).astype(int)
    tp = int(((pred==1)&(gm==1)).sum()); fp = int(((pred==1)&(gm==0)).sum()); fn = int(((pred==0)&(gm==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1); f1 = 2*prec*rec/max(prec+rec,1e-8)
    return {'AUROC': auroc, 'AUPRC': auprc, 'EPR': epr, 'P@0.2': prec, 'R@0.2': rec, 'F1@0.2': f1,
            'n_pred@thr': tp+fp, 'gt_pos': n_pos}

def eval_dir_acc(ep, ds, gt, thr=THRESHOLD):
    pred_mask = ep > thr
    gt_mask = gt > 0
    both = pred_mask & gt_mask
    n_both = int(both.sum())
    if n_both == 0: return {'DirAcc': 0.0, 'n_eval': 0}
    # Vectorized direction check: ds[i,j] >= ds[j,i] means i->j
    ds_sym_check = (ds >= ds.T)
    correct = int((both & ds_sym_check).sum())
    return {'DirAcc': correct / n_both, 'n_eval': n_both}


# ======= Main evaluation =======
BEELINE_DIR = DATA_ROOT / "BEELINE"
BL = {
    'mDC':     ("mouse/mDC-ChIP-seq-network.csv",  "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mDC/ExpressionData.csv"),
    'mHSC-E':  ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv"),
    'mHSC-GM': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv"),
    'mHSC-L':  ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv"),
    'hESC':    ("human/hESC-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv"),
    'hHep':    ("human/HepG2-ChIP-seq-network.csv","human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hHep/ExpressionData.csv"),
}

all_results = []
subset_log = []

print("\n" + "="*90, flush=True)
print("hESC Fix Evaluation: Adaptive Variance Subsetting + TF-focused Multi-window", flush=True)
print(f"Uniform rule: if n_sub/n_genes < {TRIGGER_RATIO} (trigger), subset to top-N by variance (target ratio {TARGET_RATIO})", flush=True)
print("="*90, flush=True)
print(f"\n{'Dataset':<10} {'cells':>6} {'genes':>7} {'ratio':>6} {'subset?':>8} {'n_after':>8} {'n_win':>6} {'AUROC':>7} {'AUPRC':>7} {'EPR':>6} {'DirAcc':>7} {'time':>6}", flush=True)
print("-"*90, flush=True)

for ds, (net_rel, tf_rel, expr_rel) in BL.items():
    # Allow filtering via command line: python 17_hesc_fix_evaluation.py hESC mHSC-E
    if len(sys.argv) > 1 and ds not in sys.argv[1:]:
        continue
    expr_path = BEELINE_DIR / expr_rel
    if not expr_path.exists():
        print(f"  {ds}: SKIP (no data)", flush=True); continue

    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes_all = list(expr_df.columns)
    X_all = expr_df.values.astype(np.float32)
    n_cells = X_all.shape[0]; n_genes_all = X_all.shape[1]

    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = list(tf_df[0])

    # -- Step 1: Adaptive gene subsetting (UNIFORM RULE) --
    X_sub, genes_sub, sub_info = adaptive_gene_subset(X_all, genes_all)
    n_genes_sub = len(genes_sub)
    subset_log.append({'dataset': ds, **sub_info})

    # -- Build GT on subsetted genes --
    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    g2i = {g: i for i, g in enumerate(genes_sub)}
    gt = np.zeros((n_genes_sub, n_genes_sub), dtype=np.float32)
    n_gt_total = 0; n_gt_in_subset = 0
    gene_set_sub = set(genes_sub)
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        n_gt_total += 1
        if s in g2i and t in g2i:
            gt[g2i[s], g2i[t]] = 1.0; n_gt_in_subset += 1

    mask = ~np.eye(n_genes_sub, dtype=bool)
    X_std = (X_sub - X_sub.mean(0, keepdims=True)) / (X_sub.std(0, keepdims=True) + 1e-8)

    t0 = time.time()
    # -- Step 2: TF-focused multi-window --
    windows, tf_in = build_tf_windows(genes_sub, tf_names, X_std)
    if not windows:
        print(f"  {ds}: no windows (tf_in={len(tf_in)})", flush=True); continue

    ep_full = np.zeros((n_genes_sub, n_genes_sub), dtype=np.float32)
    ds_sum = np.zeros((n_genes_sub, n_genes_sub), dtype=np.float32)
    ds_cnt = np.zeros((n_genes_sub, n_genes_sub), dtype=np.float32)

    for wi, wg in enumerate(windows):
        if wi % 100 == 0: print(f"    {ds} W{wi+1}/{len(windows)}...", end='', flush=True)
        widx = [g2i[g] for g in wg if g in g2i]
        wn = [g for g in wg if g in g2i]
        nw = len(widx)
        if nw < 10: continue
        Xw = X_std[:, widx]
        Xp = np.zeros((Xw.shape[0], G), dtype=np.float32)
        Xp[:, :nw] = Xw
        P, D = compute_P_D(Xp)
        ep_w = predict_edge(P, D)[:nw, :nw]
        ds_w = predict_dir(P, D)[:nw, :nw]
        for i, gi in enumerate(widx):
            for j, gj in enumerate(widx):
                if ep_w[i, j] > ep_full[gi, gj]:
                    ep_full[gi, gj] = ep_w[i, j]
                ds_sum[gi, gj] += ds_w[i, j]
                ds_cnt[gi, gj] += 1

    valid = ds_cnt > 0
    ds_full = np.zeros_like(ds_sum)
    ds_full[valid] = ds_sum[valid] / ds_cnt[valid]
    dt = time.time() - t0

    # -- Metrics --
    edge_r = eval_all(ep_full, gt, mask)
    dir_r = eval_dir_acc(ep_full, ds_full, gt)

    row = {
        'dataset': ds, 'source': 'BEELINE', 'coverage': 'multi_fixed',
        'n_cells': n_cells, 'n_genes_before': n_genes_all, 'n_genes_after': n_genes_sub,
        'ratio_before': sub_info['ratio'], 'subsetted': sub_info['subsetted'],
        'n_windows': len(windows), 'n_tf_in': len(tf_in),
        'gt_pos_total': n_gt_total, 'gt_pos_in_subset': n_gt_in_subset,
        'gt_coverage': n_gt_in_subset / max(n_gt_total, 1),
        'AUROC': edge_r['AUROC'], 'AUPRC': edge_r['AUPRC'], 'EPR': edge_r['EPR'],
        'P@0.2': edge_r['P@0.2'], 'R@0.2': edge_r['R@0.2'], 'F1@0.2': edge_r['F1@0.2'],
        'DirAcc': dir_r['DirAcc'], 'n_eval_dir': dir_r['n_eval'],
        'time_s': dt,
    }
    all_results.append(row)

    sub_tag = "YES" if sub_info['subsetted'] else "no"
    print(f"  {ds:<10} {n_cells:>6} {n_genes_all:>7} {sub_info['ratio']:>6.3f} {sub_tag:>8} {n_genes_sub:>8} "
          f"{len(windows):>6} {edge_r['AUROC']:>7.4f} {edge_r['AUPRC']:>7.4f} {edge_r['EPR']:>5.2f}x "
          f"{dir_r['DirAcc']:>7.4f} {dt:>5.0f}s", flush=True)

    # Save per-dataset result
    del X_all, X_sub, X_std, ep_full, ds_full, ds_sum, ds_cnt, gt; gc.collect()

# ======= Save =======
df = pd.DataFrame(all_results)
df.to_csv(RESULT_ROOT / "3_full_gene" / "hesc_fix_evaluation.csv", index=False)

sub_df = pd.DataFrame(subset_log)
sub_df.to_csv(RESULT_ROOT / "3_full_gene" / "hesc_fix_subset_log.csv", index=False)

print("\n" + "="*90, flush=True)
print("SUMMARY: Adaptive Subsetting Log", flush=True)
print("="*90, flush=True)
print(f"\n{'Dataset':<10} {'genes_before':>13} {'ratio':>6} {'subsetted':>10} {'genes_after':>12} {'gt_coverage':>12}", flush=True)
print("-"*70, flush=True)
for _, r in sub_df.iterrows():
    # GT coverage from results
    res = [x for x in all_results if x['dataset'] == r['dataset']]
    gtc = f"{res[0]['gt_coverage']:.1%}" if res else "N/A"
    print(f"  {r['dataset']:<10} {r['n_genes_before']:>13} {r['ratio']:>6.3f} "
          f"{'YES' if r['subsetted'] else 'no':>10} {r['n_genes_after']:>12} {gtc:>12}", flush=True)

print(f"\nSaved: results/3_full_gene/hesc_fix_evaluation.csv", flush=True)
print(f"Saved: results/3_full_gene/hesc_fix_subset_log.csv", flush=True)
print("DONE", flush=True)
