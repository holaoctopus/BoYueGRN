#!/usr/bin/env python
"""Comprehensive BEELINE Benchmark: BoYue vs 8 baseline methods.

Evaluates edge prediction (AUROC, AUPRC, EPR) across 6 BEELINE datasets.
BoYue is run live (4-seed ensemble, edge_v3) alongside all baselines
for fully reproducible, same-gene-selection comparison.

Usage:
  python scripts/benchmark_beeline.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, average_precision_score
import time, sys, zipfile, io, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config import DATA_ROOT, CKPT_ROOT

G = 200; d_model = 512; n_heads = 8; n_layers = 8
dcor_cell_sample = 200
NETWORKS_DIR = DATA_ROOT / "BEELINE" / "Networks"
NETWORKS_ZIP = DATA_ROOT / "BEELINE-Networks.zip"
DATA_DIR = DATA_ROOT / "BEELINE" / "BEELINE-data" / "inputs" / "scRNA-Seq"
CKPT_DIR = CKPT_ROOT / "main"

DATASETS = {
    'mDC':     ('mouse', 'mDC-ChIP-seq-network.csv',   'mDC'),
    'mHSC-E':  ('mouse', 'mHSC-ChIP-seq-network.csv',  'mHSC-E'),
    'mHSC-GM': ('mouse', 'mHSC-ChIP-seq-network.csv',  'mHSC-GM'),
    'mHSC-L':  ('mouse', 'mHSC-ChIP-seq-network.csv',  'mHSC-L'),
    'hESC':    ('human', 'hESC-ChIP-seq-network.csv',   'hESC'),
    'hHep':    ('human', 'HepG2-ChIP-seq-network.csv',   'hHep'),
}

device = torch.device('cuda')
RNG = np.random.RandomState(42)


# ===============================================================
# Load BoYue edge_v3 model (4-seed ensemble)
# ===============================================================

from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3

encoders, edge_heads = [], []
for s in range(4):
    ckpt = torch.load(CKPT_DIR / f"edge_v3_seed{s}.pt", map_location=device, weights_only=True)
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    head = EdgeHeadV3(d_model=d_model, d_k=128).to(device)
    head.load_state_dict(ckpt['edge_head']); head.eval()
    encoders.append(enc); edge_heads.append(head)
print(f"Loaded BoYue edge_v3: {len(encoders)} seeds", flush=True)


# ===============================================================
# Data loading
# ===============================================================

def load_network(species, net_csv):
    path = NETWORKS_DIR / species / net_csv
    if path.exists():
        ndf = pd.read_csv(path, header=None)
    elif NETWORKS_ZIP.exists():
        net_path = f"Networks/{species}/{net_csv}"
        with zipfile.ZipFile(NETWORKS_ZIP) as zf:
            with zf.open(net_path) as f:
                ndf = pd.read_csv(io.BytesIO(f.read()), header=None)
    else:
        raise FileNotFoundError(f"Network file not found: {path}")
    if str(ndf.iloc[0, 0]).strip() == "Gene1":
        ndf = ndf.iloc[1:]
    return ndf


def load_expression(ds_name):
    expr_path = DATA_DIR / ds_name / "ExpressionData.csv"
    expr = pd.read_csv(expr_path, index_col=0)
    gnames = expr.index.tolist()
    Xr = expr.values.astype(np.float64).T
    gvars = Xr.var(0)
    nc = gvars > 1e-12
    Xr = Xr[:, nc]
    gnames = [gnames[i] for i in range(len(gnames)) if nc[i]]
    return Xr, gnames


def select_genes(Xr, gnames, ndf, n_tf=50):
    """Gene selection from eval_beeline_edge_multi.py."""
    es = set(gnames)
    gni = {n: i for i, n in enumerate(gnames)}
    gnv = {g: Xr[:, gni[g]].var() for g in gnames}

    ns = set(); tt = {}
    for _, r in ndf.iterrows():
        s, t = str(r[0]).strip(), str(r[1]).strip()
        ns.add(s); tt.setdefault(s, set()).add(t)

    nt = [(g, len(tt.get(g, set()) & es)) for g in ns if g in es]
    nt.sort(key=lambda x: (-x[1], x[0]))
    st = [g for g, _ in nt[:min(n_tf, len(nt))]]
    sl = list(st); ss = set(st)

    stargets = set()
    for t in st:
        if t in tt: stargets |= (tt[t] & es)
    ct = [(g, gnv.get(g, 0)) for g in (stargets - ss) if g in gni]
    ct.sort(key=lambda x: (-x[1], x[0]))
    rem = G - len(st)
    for g, _ in ct[:rem]:
        sl.append(g); ss.add(g)

    if len(sl) < G:
        rg = [(g, v) for g, v in gnv.items() if g not in ss]
        rg.sort(key=lambda x: (-x[1], x[0]))
        for g, _ in rg:
            if len(sl) >= G: break
            sl.append(g)

    sn = sl[:G]
    nti = {n: i for i, n in enumerate(sn)}

    Xs = np.zeros((Xr.shape[0], G), dtype=np.float64)
    for j, g in enumerate(sn): Xs[:, j] = Xr[:, gni[g]]

    gt = np.zeros((G, G), dtype=np.float32)
    for _, r in ndf.iterrows():
        s, t = str(r[0]).strip(), str(r[1]).strip()
        if s in nti and t in nti: gt[nti[s], nti[t]] = 1.0

    return Xs, sn, gt


# ===============================================================
# dCor (same as eval_beeline_edge_multi.py for exact consistency)
# ===============================================================

def compute_dcor_fast(X, cell_indices=None):
    if cell_indices is not None: X = X[cell_indices]
    C, G_ = X.shape
    X_t = X.T.astype(np.float64)
    dist_mat = np.abs(X_t[:, None, :] - X_t[:, :, None])
    rm = dist_mat.mean(2, keepdims=True); cm = dist_mat.mean(1, keepdims=True)
    gm = dist_mat.mean((1, 2), keepdims=True)[:, None, None]
    A = dist_mat - rm - cm + gm
    Af = A.reshape(G_, -1).astype(np.float64)
    dsq = Af @ Af.T / (C * C)
    dv = np.diag(dsq).copy()
    dvp = np.sqrt(np.maximum(np.outer(dv, dv), 1e-30))
    d = np.sqrt(np.maximum(dsq / dvp, 0)); np.fill_diagonal(d, 0); d = np.clip(d, 0, 1)
    del dist_mat, A, Af, dsq; return d.astype(np.float32)


# ===============================================================
# BoYue method (live inference, 4-seed ensemble)
# ===============================================================

def method_boyue(Xs, gene_names):
    """BoYue edge_v3: P+D encoder, 4-seed ensemble, live inference."""
    X_std = (Xs - Xs.mean(0, keepdims=True)) / (Xs.std(0, keepdims=True) + 1e-8)

    lw = LedoitWolf(assume_centered=False)
    lw.fit(X_std)
    P = lw.precision_.astype(np.float32)

    ci = RNG.choice(Xs.shape[0], min(dcor_cell_sample, Xs.shape[0]), replace=False)
    D = compute_dcor_fast(X_std, ci)

    Pt = torch.tensor(P).unsqueeze(0).to(device)
    Dt = torch.tensor(D).unsqueeze(0).to(device)

    probs = None
    for enc, head in zip(encoders, edge_heads):
        with torch.no_grad():
            h = enc(Pt, Dt)
            edge_logits = head(h, Pt, Dt)
        p = torch.sigmoid(edge_logits.float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    probs = (probs / len(encoders)).numpy()
    np.fill_diagonal(probs, 0)

    # Symmetrize for fair undirected comparison (take max of (i,j) and (j,i))
    probs_sym = np.maximum(probs, probs.T)
    return probs_sym.astype(np.float32)


# ===============================================================
# Baseline methods
# ===============================================================

def method_genie3(X, gene_names):
    n_cells, n_genes = X.shape
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    im = np.zeros((n_genes, n_genes), dtype=np.float32)
    for j in range(n_genes):
        feat_idx = list(range(n_genes)); feat_idx.remove(j)
        X_feat = X_std[:, feat_idx]; y = X_std[:, j]
        rf = RandomForestRegressor(n_estimators=50, max_features='sqrt',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_feat, y)
        for k, fi in enumerate(rf.feature_importances_):
            im[feat_idx[k], j] = fi
    im = 0.5 * (im + im.T)
    np.fill_diagonal(im, 0)
    return im

def method_grnboost2(X, gene_names):
    n_cells, n_genes = X.shape
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    im = np.zeros((n_genes, n_genes), dtype=np.float32)
    for j in range(n_genes):
        feat_idx = list(range(n_genes)); feat_idx.remove(j)
        X_feat = X_std[:, feat_idx]; y = X_std[:, j]
        gb = GradientBoostingRegressor(n_estimators=50, max_features='sqrt',
                                        random_state=42, verbose=0)
        gb.fit(X_feat, y)
        for k, fi in enumerate(gb.feature_importances_):
            im[feat_idx[k], j] = fi
    im = 0.5 * (im + im.T)
    np.fill_diagonal(im, 0)
    return im

def method_correlation(X, gene_names):
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    corr = np.abs(np.corrcoef(X_std.T))
    np.fill_diagonal(corr, 0)
    return corr.astype(np.float32)

def method_mutual_info(X, gene_names):
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    Xd = discretizer.fit_transform(X)
    n_genes = X.shape[1]
    mi = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i in range(n_genes):
        mi[i, :] = mutual_info_regression(Xd, Xd[:, i])
    np.fill_diagonal(mi, 0)
    return mi

def method_ppcor(X, gene_names):
    from sklearn.covariance import GraphicalLassoCV
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=200, n_jobs=-1)
        gl.fit(X_std)
        P = gl.precision_
    except Exception:
        P = np.eye(G)
    diag = np.diag(P).copy()
    diag_sqrt = np.sqrt(np.maximum(np.outer(diag, diag), 1e-16))
    pcor = -P / diag_sqrt
    np.fill_diagonal(pcor, 0)
    return np.abs(pcor).astype(np.float32)

def method_glasso(X, gene_names):
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(X_std)
    P = lw.precision_.astype(np.float32)
    np.fill_diagonal(P, 0)
    return np.abs(P)

def method_dcor(X, gene_names):
    ncells, ng = X.shape
    max_n = 200
    if ncells > max_n:
        idx = RNG.choice(ncells, max_n, replace=False)
        X = X[idx]
        ncells = max_n
    X_t = torch.tensor(X.T.astype(np.float32), device=device)
    X_c = X_t - X_t.mean(dim=1, keepdim=True)
    xi = X_c.unsqueeze(1); xj = X_c.unsqueeze(2)
    D_dist = (xi - xj).abs()
    row_mean = D_dist.mean(dim=2, keepdim=True)
    col_mean = D_dist.mean(dim=1, keepdim=True)
    grand_mean = D_dist.mean(dim=(1, 2), keepdim=True)
    A_mat = D_dist - row_mean - col_mean + grand_mean
    A_flat = A_mat.reshape(ng, ncells * ncells)
    dcov2 = (A_flat @ A_flat.T) / (ncells * ncells)
    dvar = dcov2.diagonal()
    dvar_ij = dvar.unsqueeze(0) * dvar.unsqueeze(1)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0)) / torch.sqrt(torch.clamp(dvar_ij, min=1e-16))
    dcor = torch.clamp(dcor, 0, 1)
    result = dcor.cpu().numpy().astype(np.float32)
    np.fill_diagonal(result, 0)
    return result

def method_random(X, gene_names):
    rng = np.random.RandomState(42)
    mat = rng.rand(G, G).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat


# ===============================================================
# Evaluation
# ===============================================================

def early_precision(targets, scores, k):
    s = scores.flatten(); t = targets.flatten()
    order = np.argsort(-s)
    return t[order[:k]].mean()

def compute_metrics(mat, gt):
    diag_mask = ~np.eye(G, dtype=bool)
    gt_flat = gt[diag_mask].flatten()
    scores_flat = mat[diag_mask].flatten()

    valid = np.isfinite(scores_flat)
    if not valid.all():
        scores_flat = np.nan_to_num(scores_flat, nan=0.0, posinf=1.0, neginf=0.0)

    if gt_flat.sum() == 0:
        return 0.5, 0.0, 0.0

    try:
        auroc = roc_auc_score(gt_flat, scores_flat)
    except Exception:
        auroc = 0.5
    try:
        auprc = average_precision_score(gt_flat, scores_flat)
    except Exception:
        auprc = 0.0

    total_pos = int(gt_flat.sum())
    k = total_pos  # BEELINE standard: k = n_gt
    epr = early_precision(gt, mat, k)

    return auroc, auprc, epr


# ===============================================================
# Main
# ===============================================================

METHODS = {
    'GENIE3':     method_genie3,
    'GRNBoost2':  method_grnboost2,
    'Correlation': method_correlation,
    'MutualInfo': method_mutual_info,
    'PPCOR':      method_ppcor,
    'GLasso':     method_glasso,
    'dCor':       method_dcor,
    'Random':     method_random,
    'BoYue':      method_boyue,
}

METHOD_ORDER = list(METHODS.keys())

print("=" * 130, flush=True)
print("BEELINE Comprehensive Benchmark: Edge Prediction (AUROC / AUPRC / EPR)", flush=True)
print("=" * 130, flush=True)

# -- AUROC table --
print(f"\n{'Dataset':<10} {'cells':>6} {'n_gt':>7}", end="", flush=True)
for name in METHOD_ORDER:
    print(f" {name:>9}", end="", flush=True)
print(f"  {'Best':>8}", flush=True)
print("-" * 130, flush=True)

all_results = {}

for ds_name, (species, net_csv, ds_dir) in DATASETS.items():
    t0 = time.time()

    Xr, gnames = load_expression(ds_name)
    ndf = load_network(species, net_csv)
    Xs, sn, gt = select_genes(Xr, gnames, ndf)
    n_gt = int(gt.sum())

    if n_gt == 0:
        print(f"{ds_name:<10} {'SKIP':>50}", flush=True)
        continue

    results_row = {}

    for method_name, method_fn in METHODS.items():
        t1 = time.time()
        try:
            mat = method_fn(Xs, sn)
            if mat.shape != (G, G):
                results_row[method_name] = (float('nan'), float('nan'), float('nan'))
                continue
            auroc, auprc, epr = compute_metrics(mat, gt)
            results_row[method_name] = (auroc, auprc, epr)
        except Exception as e:
            print(f"\n  [!] {method_name} failed: {e}", flush=True)
            results_row[method_name] = (float('nan'), float('nan'), float('nan'))

    n_cells = Xr.shape[0]
    all_results[ds_name] = (n_cells, n_gt, results_row)

    # Print AUROC row
    print(f"{ds_name:<10} {n_cells:>6} {n_gt:>7}", end="", flush=True)
    best_auroc = 0; best_name = ""
    for name in METHOD_ORDER:
        auroc = results_row[name][0]
        print(f" {auroc:>9.4f}" if np.isfinite(auroc) else f" {'FAIL':>9}", end="", flush=True)
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc; best_name = name
    marker = f" ***{best_name}***" if best_name else ""
    print(f"  {best_auroc:.4f}{marker}  [{time.time()-t0:.0f}s]", flush=True)

print("-" * 130, flush=True)

# -- AUROC Summary --
print("\nAUROC Summary (mean across datasets):", flush=True)
print("-" * 55, flush=True)
auroc_by_method = {name: [] for name in METHOD_ORDER}
for ds_name, (n_cells, n_gt, results) in all_results.items():
    for name in METHOD_ORDER:
        auroc = results[name][0]
        if np.isfinite(auroc): auroc_by_method[name].append(auroc)
for name in METHOD_ORDER:
    vals = auroc_by_method[name]
    if vals:
        print(f"  {name:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})", flush=True)

# -- AUPRC table --
print(f"\n{'='*130}", flush=True)
print("AUPRC Table (Area Under Precision-Recall Curve)", flush=True)
print(f"{'='*130}", flush=True)
print(f"{'Dataset':<10} {'cells':>6} {'n_gt':>7}", end="", flush=True)
for name in METHOD_ORDER:
    print(f" {name:>9}", end="", flush=True)
print(f"  {'Best':>8}", flush=True)
print("-" * 130, flush=True)

for ds_name, (species, net_csv, ds_dir) in DATASETS.items():
    if ds_name not in all_results: continue
    n_cells, n_gt, results = all_results[ds_name]
    print(f"{ds_name:<10} {n_cells:>6} {n_gt:>7}", end="", flush=True)
    best_auprc = 0; best_name_ap = ""
    for name in METHOD_ORDER:
        auprc = results[name][1]
        print(f" {auprc:>9.4f}" if np.isfinite(auprc) else f" {'FAIL':>9}", end="", flush=True)
        if np.isfinite(auprc) and auprc > best_auprc:
            best_auprc = auprc; best_name_ap = name
    marker = f" ***{best_name_ap}***" if best_name_ap else ""
    print(f"  {best_auprc:.4f}{marker}", flush=True)

print("-" * 130, flush=True)

# -- AUPRC Summary --
print("\nAUPRC Summary (mean across datasets):", flush=True)
print("-" * 55, flush=True)
auprc_by_method = {name: [] for name in METHOD_ORDER}
for ds_name, (n_cells, n_gt, results) in all_results.items():
    for name in METHOD_ORDER:
        auprc = results[name][1]
        if np.isfinite(auprc): auprc_by_method[name].append(auprc)
for name in METHOD_ORDER:
    vals = auprc_by_method[name]
    if vals:
        print(f"  {name:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})", flush=True)

# -- EPR Summary --
print(f"\n{'='*130}", flush=True)
print("EPR Table (Early Precision Ratio)", flush=True)
print(f"{'='*130}", flush=True)
print(f"{'Dataset':<10} {'cells':>6} {'n_gt':>7}", end="", flush=True)
for name in METHOD_ORDER:
    print(f" {name:>9}", end="", flush=True)
print(f"  {'Best':>8}", flush=True)
print("-" * 130, flush=True)

for ds_name, (species, net_csv, ds_dir) in DATASETS.items():
    if ds_name not in all_results: continue
    n_cells, n_gt, results = all_results[ds_name]
    print(f"{ds_name:<10} {n_cells:>6} {n_gt:>7}", end="", flush=True)
    best_epr = 0; best_name_ep = ""
    for name in METHOD_ORDER:
        epr = results[name][2]
        print(f" {epr:>9.4f}" if np.isfinite(epr) else f" {'FAIL':>9}", end="", flush=True)
        if np.isfinite(epr) and epr > best_epr:
            best_epr = epr; best_name_ep = name
    marker = f" ***{best_name_ep}***" if best_name_ep else ""
    print(f"  {best_epr:.4f}{marker}", flush=True)

print("-" * 130, flush=True)

print("\nDone.", flush=True)
