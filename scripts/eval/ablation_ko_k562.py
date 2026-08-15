#!/usr/bin/env python
"""Ablation-KO small-scale test on K562: 10 TFs.

Concept: ablate TF expression in control cells (zero / shuffle), recompute
P/D, re-run edge+dir models. Genes whose edge_prob drops most = predicted
downstream targets of the ablated TF.

Validation: Replogle truth-seq z-scores (|z|>=2 = true responders).

Predictors compared per TF:
  1. |Δedge|   — edge_baseline[tf,j] - edge_ablated[tf,j]  (virtual KO signal)
  2. edge      — static edge_prob[tf,j]                    (static GRN)
  3. |pearson| — |corr(tf, j)| on control cells            (naive baseline)
GT: responders = truth-seq |z|>=2 (excluding self).

Metrics: AUROC, EPR, Spearman(|Δedge|, |z|).

Usage:
  python scripts/eval/ablation_ko_k562.py
"""
import numpy as np, pandas as pd, torch, time, gc, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
import h5py as hf
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, n_layers, dropout, sd_prob, device
from train_gt_g200_dir_specialist import GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead

N_TEST_TF = 10
Z_THR = 2.0
DCOR_CELLS = 200
rng = np.random.RandomState(42)

# ======= Load models =======
print("Loading models...", flush=True)
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

dir_enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
dir_head = AsymmetricDirHead(d_model=d_model).to(device)
dstate = torch.load(CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt",
                     map_location=device, weights_only=True)
dir_enc.load_state_dict(dstate['encoder']); dir_head.load_state_dict(dstate['dir_head'])
dir_enc.eval(); dir_head.eval()
print("  TT direction ready.", flush=True)

# ======= P/D + inference =======
def compute_P_D(X):
    n_cells, Gv = X.shape
    lw = LedoitWolf(); lw.fit(X)
    P = lw.precision_.astype(np.float32)
    Xd = X
    if n_cells > DCOR_CELLS:
        Xd = X[rng.choice(n_cells, DCOR_CELLS, replace=False)]
    X_c = Xd - Xd.mean(axis=0, keepdims=True)
    A = np.zeros((Gv, Xd.shape[0]**2), dtype=np.float64)
    for g in range(Gv):
        d = np.abs(X_c[:, g:g+1] - X_c[:, g:g+1].T)
        A[g] = (d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()).ravel()
    dcov2 = A @ A.T / Xd.shape[0]**2
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    D = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(D, 0); D = np.clip(D, 0, 1)
    del A, dcov2; gc.collect()
    return P, D.astype(np.float32)

@torch.no_grad()
def predict(P, D):
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    probs = None
    for enc, head in zip(encoders, edge_heads):
        h = enc(Pt, Dt)
        p = torch.sigmoid(head(h, Pt, Dt).float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    ep = (probs / len(encoders)).numpy()
    np.fill_diagonal(ep, 0)
    h_d = dir_enc(Pt, Dt)
    ds = torch.sigmoid(dir_head(h_d, Pt, Dt).float()).squeeze(0).cpu().numpy()
    np.fill_diagonal(ds, 0)
    return ep.astype(np.float32), ds.astype(np.float32)

# ======= Load K562 control + truth-seq =======
print("\nLoading K562...", flush=True)
with hf.File(DATA_ROOT / "K562_gwps_normalized_bulk_01.h5ad", 'r') as f:
    gc_arr = f["var"]["__categories"]["gene_name"][:]
    gi_arr = f["var"]["gene_name"][:]
    gene_syms = np.array([g.decode('utf-8').upper() for g in [gc_arr[i] for i in gi_arr]])
    X_k = f['X'][:]; cc = f['obs']['core_control'][:]
X_k = np.nan_to_num(np.asarray(X_k, dtype=np.float32))
X_ctrl = X_k[cc]
del X_k; gc.collect()
mu = X_ctrl.mean(0, keepdims=True); std = np.maximum(X_ctrl.std(0, ddof=1, keepdims=True), 1e-8)
X_std = ((X_ctrl - mu) / std).astype(np.float32)  # standardized control
print(f"  Control cells: {X_std.shape}", flush=True)

rt = pd.read_parquet(DATA_ROOT / "truthseq_k562" / "replogle_knockdown_effects.parquet")
rt['kd'] = rt['knocked_down_gene'].str.upper()
rt['aff'] = rt['affected_gene'].str.upper()
# Vectorized pivot: rows=kd, cols=aff, values=z (37.7M rows -> fast)
Z = rt.pivot_table(index='kd', columns='aff', values='z_score', aggfunc='first')
kd_counts = (Z.abs() >= Z_THR).sum(axis=1).sort_values(ascending=False)
del rt; gc.collect()
print(f"  Truth-seq: {Z.shape[0]} KD x {Z.shape[1]} affected", flush=True)

human_tfs = set(pd.read_csv(DATA_ROOT / "BEELINE" / "human-tfs.csv", header=None)[0].str.upper())
gene_set = set(gene_syms)

# Top TFs: most responders, must be human TF, in expression data
cand = [t for t in kd_counts.index if t in human_tfs and t in gene_set]
test_tfs = sorted(cand, key=lambda t: -kd_counts[t])[:N_TEST_TF]
print(f"  Test TFs ({len(test_tfs)}): {test_tfs}", flush=True)
print(f"  Responders per TF: {[int(kd_counts[t]) for t in test_tfs]}", flush=True)

# ======= Window: 10 TFs + 190 top-variance non-TF genes (no GT leak) =======
g2i = {g: i for i, g in enumerate(gene_syms)}
tf_idx = [g2i[t] for t in test_tfs]
ctrl_var = X_std.var(axis=0)
var_order = np.argsort(ctrl_var)[::-1]
win_genes = list(test_tfs)
win_set = set(win_genes)
for gi in var_order:
    if len(win_genes) >= G: break
    g = gene_syms[gi]
    if g not in win_set:
        win_genes.append(g); win_set.add(g)
win_genes = win_genes[:G]
widx = [g2i[g] for g in win_genes]
nw = len(widx)
print(f"  Window: {nw} genes ({len(test_tfs)} TF + {nw-len(test_tfs)} targets)", flush=True)

# GT responders per TF within window
win_gene_set = set(win_genes)
responders = {}  # tf -> set of responder genes in window
for tf in test_tfs:
    row = Z.loc[tf] if tf in Z.index else None
    resp = set()
    if row is not None:
        sub = row.reindex(list(win_gene_set))
        resp = set(sub.index[sub.abs() >= Z_THR])
    resp.discard(tf)
    responders[tf] = resp
    print(f"  {tf}: {len(resp)} responders in window", flush=True)

# Pearson baseline (control cells)
Xw = X_std[:, widx]
pear = np.corrcoef(Xw.T).astype(np.float32)
np.fill_diagonal(pear, 0)

# ======= Baseline inference =======
print("\nBaseline inference...", flush=True)
Xp = np.zeros((Xw.shape[0], G), dtype=np.float32); Xp[:, :nw] = Xw
P0, D0 = compute_P_D(Xp)
ep0, ds0 = predict(P0, D0)
ep0, ds0 = ep0[:nw, :nw], ds0[:nw, :nw]

# ======= Ablation runs =======
results = []
for abl_mode in ['zero', 'shuffle']:
    print(f"\n=== Ablation mode: {abl_mode} ===", flush=True)
    delta = np.zeros((nw, nw), dtype=np.float32)
    for ti, tf in enumerate(test_tfs):
        i = win_genes.index(tf)
        Xa = Xp.copy()
        if abl_mode == 'zero':
            Xa[:, i] = 0.0
        else:
            perm = rng.permutation(Xa.shape[0])
            Xa[:, i] = Xa[perm, i]
        Pa, Da = compute_P_D(Xa)
        epa, _ = predict(Pa, Da)
        epa = epa[:nw, :nw]
        # Δedge: drop in edge prob from tf to j (both directions, take max like eval)
        d_row = np.maximum(ep0[i, :] - epa[i, :], 0)
        d_col = np.maximum(ep0[:, i] - epa[:, i], 0)
        delta[i, :] = np.maximum(d_row, d_col)
        print(f"  {tf} ablated", flush=True)

    # ======= Evaluate per TF =======
    print(f"\nPer-TF results ({abl_mode}):", flush=True)
    print(f"  {'TF':<10} {'n_resp':>6} {'AUROC_Δe':>9} {'AUROC_e':>8} {'AUROC_p':>8} {'EPR_Δe':>7} {'SPR_Δe':>7}", flush=True)
    for tf in test_tfs:
        i = win_genes.index(tf)
        resp = responders[tf]
        if len(resp) < 3:
            print(f"  {tf:<10} {len(resp):>6}  SKIP (too few responders)", flush=True)
            continue
        genes_eval = [g for g in win_genes if g != tf]
        jidx = [win_genes.index(g) for g in genes_eval]
        y = np.array([1.0 if g in resp else 0.0 for g in genes_eval])
        s_delta = np.abs(delta[i, jidx])
        s_edge = ep0[i, jidx]
        s_pear = np.abs(pear[i, jidx])
        z_row = Z.loc[tf] if tf in Z.index else pd.Series(dtype=float)
        zvals = np.abs(z_row.reindex(genes_eval).fillna(0.0).values)

        def auroc(sc):
            try: return roc_auc_score(y, sc)
            except: return 0.5
        n_pos = int(y.sum())
        order = np.argsort(-s_delta)[:n_pos]
        epr = y[order].mean() / max(n_pos / len(y), 1e-10)
        spr, _ = spearmanr(zvals, s_delta)

        r = {'mode': abl_mode, 'tf': tf, 'n_resp': n_pos,
             'auroc_delta': auroc(s_delta), 'auroc_edge': auroc(s_edge), 'auroc_pearson': auroc(s_pear),
             'epr_delta': epr, 'spearman_delta_z': spr}
        results.append(r)
        print(f"  {tf:<10} {n_pos:>6} {r['auroc_delta']:>9.4f} {r['auroc_edge']:>8.4f} {r['auroc_pearson']:>8.4f} {epr:>6.2f}x {spr:>+7.3f}", flush=True)

# ======= Summary =======
df = pd.DataFrame(results)
df.to_csv(RESULT_ROOT / "5_ablation_controls" / "ablation_ko_k562.csv", index=False)
print("\n" + "="*70, flush=True)
print("SUMMARY (mean ± std across TFs)", flush=True)
print("="*70, flush=True)
for mode in ['zero', 'shuffle']:
    sub = df[df['mode'] == mode]
    if len(sub) == 0: continue
    print(f"\n  Mode: {mode}  (n={len(sub)} TFs)", flush=True)
    print(f"    AUROC |Δedge|  : {sub['auroc_delta'].mean():.4f} ± {sub['auroc_delta'].std():.4f}", flush=True)
    print(f"    AUROC static e : {sub['auroc_edge'].mean():.4f} ± {sub['auroc_edge'].std():.4f}", flush=True)
    print(f"    AUROC |pearson|: {sub['auroc_pearson'].mean():.4f} ± {sub['auroc_pearson'].std():.4f}", flush=True)
    print(f"    EPR |Δedge|    : {sub['epr_delta'].mean():.2f}x", flush=True)
    print(f"    Spearman(|Δe|,|z|): {sub['spearman_delta_z'].mean():+.4f}", flush=True)
print(f"\nSaved: results/5_ablation_controls/ablation_ko_k562.csv", flush=True)
print("DONE", flush=True)
