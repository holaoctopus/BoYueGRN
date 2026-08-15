#!/usr/bin/env python
"""Full Evaluation Matrix: 8 datasets × 2 coverage × 2 models.

Datasets: mDC, mHSC-E, mHSC-GM, mHSC-L, hESC, hHep (BEELINE) + HCT116, K562
Coverage: single-window (G=200, pad) + TF-focused multi-window
Models: edge prediction (4-seed ensemble) + TT direction prediction
Metrics: AUROC, AUPRC, EPR, Precision, Recall, F1, DirAcc
"""
import sys
import numpy as np, pandas as pd, torch, time, gc, json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers, dropout, sd_prob, device
from train_gt_g200_dir_specialist import GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead

TF_PER_WINDOW = 5; TOP_K_TARGETS = 50; THRESHOLDS = [0.1, 0.2, 0.3, 0.5]

# ======= Load models =======
print("Loading edge models (4-seed ensemble)...")
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
print(f"  Edge: {len(encoders)} seeds")

print("Loading TT direction specialist...")
dir_enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
dir_head = AsymmetricDirHead(d_model=d_model).to(device)
dstate = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt",
                     map_location=device, weights_only=True)
dir_enc.load_state_dict(dstate['encoder']); dir_head.load_state_dict(dstate['dir_head'])
dir_enc.eval(); dir_head.eval()
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
    """4-seed ensemble edge prediction."""
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
    """TT specialist direction prediction."""
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    h_d = dir_enc(Pt, Dt)
    ds = torch.sigmoid(dir_head(h_d, Pt, Dt).float()).squeeze(0).cpu().numpy()
    np.fill_diagonal(ds, 0)
    return ds.astype(np.float32)


def eval_all(ep, gt, mask, thr=0.2):
    """Compute all metrics."""
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
    return {'AUROC': auroc, 'AUPRC': auprc, 'EPR': epr, f'P@{thr}': prec, f'R@{thr}': rec, f'F1@{thr}': f1,
            'n_pred@thr': tp+fp, 'gt_pos': n_pos}

def eval_dir_acc(ep, ds, gt, thr=0.2):
    """Direction accuracy on GT edges above threshold."""
    pred_mask = ep > thr
    gt_mask = gt > 0
    both = pred_mask & gt_mask
    n_both = int(both.sum())
    if n_both == 0: return {'DirAcc': 0.0, 'n_eval': 0}
    correct = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if both[i, j]:
                # GT direction: gt[i,j]>0 means i->j
                # Model direction: ds[i,j] > ds[j,i] means i->j
                if ds[i, j] >= ds[j, i]:
                    correct += 1
    return {'DirAcc': correct / n_both, 'n_eval': n_both}


def build_tf_windows(genes, tf_names, X_std, top_k=TOP_K_TARGETS):
    """TF-focused windows."""
    g2i = {g: i for i, g in enumerate(genes)}
    tf_in = [g for g in genes if g.upper() in set(t.upper() for t in tf_names)]
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
            if g.upper() in set(t.upper() for t in tf_names): continue
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
            # fill
            v = X_std.var(axis=0); fo = np.argsort(v)[::-1]
            for idx in fo:
                if len(wg) >= G: break
                g = genes[idx]
                if g not in ws: wg.append(g); ws.add(g)
            windows.append(wg[:G])
    return windows, tf_in


# ======= Evaluate single dataset =======
def evaluate_dataset(name, X_raw, genes, tf_names, gt, tf_names_in_data=None):
    """Run all 4 combinations on one dataset."""
    n_genes = len(genes)
    g2i = {g: i for i, g in enumerate(genes)}
    mask = ~np.eye(n_genes, dtype=bool)

    X_std = (X_raw - X_raw.mean(0, keepdims=True)) / (X_raw.std(0, keepdims=True) + 1e-8)
    results = {}

    for coverage in ['single', 'multi']:
        print(f"    {coverage}...", end='', flush=True)
        t0 = time.time()

        if coverage == 'single':
            rng = np.random.RandomState(42)
            n_win = min(n_genes, G)
            sel_idx = rng.choice(n_genes, n_win, replace=False)
            sel_genes = [genes[i] for i in sel_idx]
            X_win = X_std[:, sel_idx]
            X_pad = np.zeros((X_win.shape[0], G), dtype=np.float32)
            X_pad[:, :n_win] = X_win
            P, D = compute_P_D(X_pad)
            ep = predict_edge(P, D)[:n_win, :n_win]
            ds = predict_dir(P, D)[:n_win, :n_win]
            ep_full = np.zeros((n_genes, n_genes), dtype=np.float32)
            ds_full = np.zeros((n_genes, n_genes), dtype=np.float32)
            for i, gi in enumerate(sel_idx):
                for j, gj in enumerate(sel_idx):
                    ep_full[gi, gj] = ep[i, j]
                    ds_full[gi, gj] = ds[i, j]
            gt_sub = gt; mask_sub = mask
            n_wins = 1
        else:
            # TF-focused multi-window
            windows, tf_in = build_tf_windows(genes, tf_names, X_std)
            if not windows:
                print(f" no windows"); continue
            ep_full = np.zeros((n_genes, n_genes), dtype=np.float32)
            ds_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
            ds_cnt = np.zeros((n_genes, n_genes), dtype=np.float32)
            for wi, wg in enumerate(windows):
                if wi % 50 == 0: print(f"\n      W{wi+1}/{len(windows)}...", end='', flush=True)
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
                        ep_full[gi, gj] = max(ep_full[gi, gj], ep_w[i, j])
                        ds_sum[gi, gj] += ds_w[i, j]
                        ds_cnt[gi, gj] += 1
            valid = ds_cnt > 0
            ds_full = np.zeros_like(ds_sum)
            ds_full[valid] = ds_sum[valid] / ds_cnt[valid]
            gt_sub = gt; mask_sub = mask
            n_wins = len(windows)
            print(f" done", end='')

        dt = time.time() - t0

        # Edge metrics
        edge_r = eval_all(ep_full, gt, mask)
        edge_r['coverage'] = coverage
        edge_r['windows'] = n_wins
        edge_r['time_s'] = dt

        # Direction metrics
        dir_r = eval_dir_acc(ep_full, ds_full, gt, thr=0.2)

        edge_r.update(dir_r)
        results[coverage] = edge_r
        print(f" AUROC={edge_r['AUROC']:.4f} EPR={edge_r['EPR']:.2f}x DirAcc={dir_r['DirAcc']:.4f} ({dt:.1f}s)")

    return results


# ======= Main: BEELINE =======
print("\n" + "="*70)
print("PART 1: BEELINE (6 datasets)")
print("="*70)

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
    if not expr_path.exists(): continue
    print(f"\n  {ds}:")
    expr_df = pd.read_csv(expr_path, index_col=0).T
    genes = list(expr_df.columns)
    X = expr_df.values.astype(np.float32)
    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = list(tf_df[0])
    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    gt = np.zeros((len(genes), len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
        if s in gene_to_idx and t in gene_to_idx:
            gt[gene_to_idx[s], gene_to_idx[t]] = 1.0

    r = evaluate_dataset(ds, X, genes, tf_names, gt)
    for cov, metrics in r.items():
        row = {'dataset': ds, 'source': 'BEELINE', **metrics}
        all_results.append(row)

# ======= K562 =======
print("\n" + "="*70)
print("PART 2: K562 Perturb-seq — SKIPPED (data file not found)")
print("="*70)

# ======= Save =======
df = pd.DataFrame(all_results)
print("\n" + "="*70)
print("FULL EVALUATION MATRIX")
print("="*70)
print(f"\n{'Dataset':<10s} {'Source':<12s} {'Cov':<8s} {'AUROC':>7s} {'AUPRC':>7s} {'EPR':>7s} {'P@.2':>7s} {'R@.2':>7s} {'F1@.2':>7s} {'DirAcc':>7s} {'Wins':>5s}")
print("-"*100)
for _, r in df.iterrows():
    print(f"  {r['dataset']:<10s} {r['source']:<12s} {r['coverage']:<8s} {r['AUROC']:>7.4f} {r['AUPRC']:>7.4f} "
          f"{r['EPR']:>6.2f}x {r.get('P@0.2',0):>7.4f} {r.get('R@0.2',0):>7.4f} {r.get('F1@0.2',0):>7.4f} "
          f"{r.get('DirAcc',0):>7.4f} {r.get('windows',0):>5d}")

df.to_csv(RESULT_ROOT / "3_full_gene" / "full_evaluation_matrix.csv", index=False)
print(f"\nSaved: results/3_full_gene/full_evaluation_matrix.csv")
print("DONE")
