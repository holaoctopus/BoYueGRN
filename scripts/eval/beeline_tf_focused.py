#!/usr/bin/env python
"""BEELINE TF-focused sliding window: 5 TFs + 195 targets per window.
Same strategy as genome-wide pipeline. Compare vs existing benchmark AUROC.
"""
import numpy as np, pandas as pd, torch, time, gc
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers, dropout, sd_prob, device

BEELINE_DIR = DATA_ROOT / "BEELINE"
TF_PER_WINDOW = 5
TOP_K_TARGETS = 50

DATASETS = {
    'mDC': ("mouse/mDC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mDC/ExpressionData.csv"),
    'mHSC-E': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv"),
    'mHSC-GM': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv"),
    'mHSC-L': ("mouse/mHSC-ChIP-seq-network.csv", "mouse-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv"),
    'hESC': ("human/hESC-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv"),
    'hHep': ("human/HepG2-ChIP-seq-network.csv", "human-tfs.csv", "BEELINE-data/inputs/scRNA-Seq/hHep/ExpressionData.csv"),
}

print("Loading model (4-seed ensemble)...")
encoders, edge_heads = [], []
for seed in range(4):
    ckpt_path = CKPT_ROOT / "main" / f"edge_v3_seed{seed}.pt"
    if not ckpt_path.exists(): continue
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
    head = EdgeHeadV3(d_model=d_model).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    enc.load_state_dict(ck['encoder']); head.load_state_dict(ck['edge_head'])
    enc.eval(); head.eval()
    encoders.append(enc); edge_heads.append(head)
print(f"  Loaded {len(encoders)} seeds")


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
def predict_ensemble(P, D):
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


def build_tf_focused_windows(gene_names, tf_names, X_std, top_k=TOP_K_TARGETS):
    """TF-focused windows: 5 TFs + top-k correlated targets per TF."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    tf_in = sorted(set(g.upper() for g in gene_names) & set(t.upper() for t in tf_names))
    tf_in = [g for g in gene_names if g.upper() in set(t.upper() for t in tf_names)]

    if len(tf_in) < 2:
        return [], tf_in

    # Pre-screen: correlation matrix on subsampled cells
    rng_ps = np.random.RandomState(42)
    n_sub = min(500, X_std.shape[0])
    X_sub = X_std[rng_ps.choice(X_std.shape[0], n_sub, replace=False)]
    corr_mat = np.corrcoef(X_sub.T).astype(np.float32)

    # Per-TF candidates
    candidates = {}
    for tf in tf_in:
        ti = gene_to_idx[tf]
        corrs = np.abs(corr_mat[ti])
        sorted_idx = np.argsort(corrs)[::-1]
        cands = []
        for idx in sorted_idx:
            if idx == ti: continue
            g = gene_names[idx]
            if g.upper() in set(t.upper() for t in tf_in): continue  # skip TF targets
            cands.append(g)
            if len(cands) >= top_k: break
        candidates[tf] = cands
    del corr_mat; gc.collect()

    # Build windows
    tf_list = sorted(candidates.keys(), key=lambda t: len(candidates[t]), reverse=True)
    target_set = set()
    for cs in candidates.values():
        for g in cs: target_set.add(g)
    targets = list(target_set)

    target_slots = G - TF_PER_WINDOW
    windows = []
    for bs in range(0, len(tf_list), TF_PER_WINDOW):
        batch_tfs = tf_list[bs:bs + TF_PER_WINDOW]
        if len(batch_tfs) < 2: continue
        batch_targets = targets[:]
        n_rot = -(-len(batch_targets) // target_slots) if len(batch_targets) > target_slots else 1
        for rot in range(n_rot):
            rs = rot * target_slots
            rot_targets = batch_targets[rs:rs + target_slots]
            wg = list(batch_tfs) + list(rot_targets)
            ws = set(wg)
            # Fill to G with high-variance genes
            gene_vars = X_std.var(axis=0)
            filler_order = np.argsort(gene_vars)[::-1]
            for idx in filler_order:
                if len(wg) >= G: break
                g = gene_names[idx]
                if g not in ws: wg.append(g); ws.add(g)
            windows.append(wg[:G])

    return windows, tf_in


def eval_metrics(probs, gt, mask):
    pm, gm = probs[mask], gt[mask]
    try: auroc = roc_auc_score(gm, pm)
    except: auroc = 0.5
    try: auprc = average_precision_score(gm, pm)
    except: auprc = 0.0
    n_pos = int(gm.sum())
    if n_pos > 0:
        top_indices = np.argsort(pm)[::-1][:n_pos]
        prec_top = gm[top_indices].mean()
        random_prec = n_pos / len(gm)
        epr = prec_top / max(random_prec, 1e-10)
    else:
        prec_top = 0; epr = 0
    return {'AUROC': auroc, 'AUPRC': auprc, 'EPR': epr, 'gt_pos': n_pos}


# ======= Main =======
print("\n" + "="*70)
print("BEELINE TF-Focused Window Evaluation (4-seed ensemble)")
print("="*70)

all_results = []
for ds, (net_rel, tf_rel, expr_rel) in DATASETS.items():
    expr_path = BEELINE_DIR / expr_rel
    net_path = BEELINE_DIR / "Networks" / net_rel
    tf_path = BEELINE_DIR / tf_rel
    if not expr_path.exists(): print(f"[SKIP] {ds}"); continue

    print(f"\n{'='*50}\n  {ds}\n{'='*50}")
    t0 = time.time()

    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes = list(expr_df.columns)
    X_raw = expr_df.values.astype(np.float32)
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    tf_df = pd.read_csv(tf_path, header=None)
    tf_names = list(tf_df[0])

    net_df = pd.read_csv(net_path)
    gt = np.zeros((n_genes, n_genes), dtype=np.float32)
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in gene_to_idx and t in gene_to_idx:
            gt[gene_to_idx[s], gene_to_idx[t]] = 1.0

    n_gt = int(gt.sum())
    print(f"  Genes: {n_genes}, Cells: {X_raw.shape[0]}, GT edges: {n_gt}")

    X_std = (X_raw - X_raw.mean(0, keepdims=True)) / (X_raw.std(0, keepdims=True) + 1e-8)

    windows, tf_in = build_tf_focused_windows(genes, tf_names, X_std)
    print(f"  TFs in data: {len(tf_in)}, Windows: {len(windows)}")

    if not windows:
        print("  [SKIP] no windows"); continue

    ep_max = np.zeros((n_genes, n_genes), dtype=np.float32)
    for wi, wgenes in enumerate(windows):
        if wi % 10 == 0:
            print(f"    Window {wi+1}/{len(windows)}...", end='', flush=True)
        win_idx = [gene_to_idx[g] for g in wgenes if g in gene_to_idx]
        win_names = [g for g in wgenes if g in gene_to_idx]
        n_win = len(win_idx)
        if n_win < 10: continue

        X_win = X_std[:, win_idx]
        X_pad = np.zeros((X_win.shape[0], G), dtype=np.float32)
        X_pad[:, :n_win] = X_win
        P_pad, D_pad = compute_P_D(X_pad)
        probs_win = predict_ensemble(P_pad, D_pad)[:n_win, :n_win]

        for i, gi in enumerate(win_idx):
            for j, gj in enumerate(win_idx):
                ep_max[gi, gj] = max(ep_max[gi, gj], probs_win[i, j])
        print(f" done", flush=True)

    r = eval_metrics(ep_max, gt, ~np.eye(n_genes, dtype=bool))
    r['dataset'] = ds
    r['n_genes'] = n_genes
    r['n_windows'] = len(windows)
    r['time_s'] = time.time() - t0
    all_results.append(r)
    print(f"  AUROC={r['AUROC']:.4f} AUPRC={r['AUPRC']:.4f} EPR={r['EPR']:.2f}x ({r['time_s']:.1f}s)")

    del X_raw, X_std, ep_max; gc.collect()

print("\n" + "="*70)
print("SUMMARY: TF-Focused Window BEELINE Results")
print("="*70)

df = pd.DataFrame(all_results)
print(f"\n{'Dataset':<10s} {'Genes':>6s} {'Wins':>5s} {'AUROC':>7s} {'AUPRC':>7s} {'EPR':>7s} {'Time':>7s}")
print("-"*55)
for _, r in df.iterrows():
    print(f"  {r['dataset']:<10s} {r['n_genes']:>6d} {r['n_windows']:>5d} {r['AUROC']:>7.4f} {r['AUPRC']:>7.4f} {r['EPR']:>6.2f}x {r['time_s']:>6.1f}s")

valid = df[df['AUROC'] != 0.5]
if len(valid) > 0:
    print(f"\n  Mean (excl. random): AUROC={valid['AUROC'].mean():.4f} AUPRC={valid['AUPRC'].mean():.4f} EPR={valid['EPR'].mean():.2f}x")

# Compare with existing benchmark
print(f"\n  Existing benchmark (4-seed, prepare_dataset):")
print(f"    mHSC-E: 0.749, mHSC-GM: 0.760, mHSC-L: 0.751, hESC: 0.736, hHep: 0.604")
print(f"    Mean: 0.720")

df.to_csv(RESULT_ROOT / "2_benchmark_g200" / "beeline_tf_focused.csv", index=False)
print(f"\nSaved: results/2_benchmark_g200/beeline_tf_focused.csv")
print("DONE")
