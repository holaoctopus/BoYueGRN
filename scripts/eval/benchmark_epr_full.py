#!/usr/bin/env python
"""Compute true EPR ratio for all 9 BEELINE methods (consistent with AUROC).

Recomputes top-k precision (k = n_gt) for every method using the same
select_genes, method implementations, and seed as benchmark_beeline.py, then
converts to true EPR:
    EPR_ratio = precision@k / (k / (G^2 - G)),  G=200 -> G^2-G = 39800
Random should be approximately 1.

Output (new file, does not overwrite):
  results/2_benchmark_g200/all_methods_epr_ratio_full.csv
  columns: dataset, method, n_pred(=k), n_gt, topk_precision, EPR_ratio, ECR(=k/39800)

Usage:
  BOYUE_DATA=$BOYUE_ROOT/Data \
    python scripts/eval/benchmark_epr_full.py
"""
import sys, time, warnings
import numpy as np, pandas as pd, torch
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import DATA_ROOT, CKPT_ROOT, RESULT_ROOT

G = 200; GSQ = G*G - G
dcor_cell_sample = 200
NETWORKS_DIR = DATA_ROOT / "BEELINE" / "Networks"
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

# BoYue edge_v3 (4-seed ensemble)
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3
encoders, edge_heads = [], []
for s in range(4):
    ck = torch.load(CKPT_DIR / f"edge_v3_seed{s}.pt", map_location=device, weights_only=True)
    e = GraphTransformerEncoderV3(G=G, d_model=512, n_heads=8, n_layers=8).to(device)
    e.load_state_dict(ck['encoder']); e.eval()
    h = EdgeHeadV3(d_model=512, d_k=128).to(device)
    h.load_state_dict(ck['edge_head']); h.eval()
    encoders.append(e); edge_heads.append(h)
print(f"Loaded BoYue edge_v3: {len(encoders)} seeds", flush=True)

def load_network(species, net_csv):
    ndf = pd.read_csv(NETWORKS_DIR / species / net_csv, header=None)
    if str(ndf.iloc[0,0]).strip() == "Gene1": ndf = ndf.iloc[1:]
    return ndf

def load_expression(ds):
    expr = pd.read_csv(DATA_DIR / ds / "ExpressionData.csv", index_col=0)
    gnames = expr.index.tolist()
    Xr = expr.values.astype(np.float64).T
    gvars = Xr.var(0); nc = gvars > 1e-12
    Xr = Xr[:, nc]
    gnames = [gnames[i] for i in range(len(gnames)) if nc[i]]
    return Xr, gnames

def select_genes(Xr, gnames, ndf, n_tf=50):
    es = set(gnames); gni = {n:i for i,n in enumerate(gnames)}
    gnv = {g: Xr[:, gni[g]].var() for g in gnames}
    ns = set(); tt = {}
    for _, r in ndf.iterrows():
        s, t = str(r[0]).strip(), str(r[1]).strip()
        ns.add(s); tt.setdefault(s, set()).add(t)
    nt = [(g, len(tt.get(g,set()) & es)) for g in ns if g in es]
    nt.sort(key=lambda x: (-x[1], x[0]))
    st = [g for g,_ in nt[:min(n_tf, len(nt))]]
    sl = list(st); ss = set(st)
    stargets = set()
    for t in st:
        if t in tt: stargets |= (tt[t] & es)
    ct = [(g, gnv.get(g,0)) for g in (stargets - ss) if g in gni]
    ct.sort(key=lambda x: (-x[1], x[0]))
    rem = G - len(st)
    for g,_ in ct[:rem]: sl.append(g); ss.add(g)
    if len(sl) < G:
        rg = [(g,v) for g,v in gnv.items() if g not in ss]
        rg.sort(key=lambda x: (-x[1], x[0]))
        for g,_ in rg:
            if len(sl) >= G: break
            sl.append(g)
    sn = sl[:G]; nti = {n:i for i,n in enumerate(sn)}
    Xs = np.zeros((Xr.shape[0], G), dtype=np.float64)
    for j,g in enumerate(sn): Xs[:, j] = Xr[:, gni[g]]
    gt = np.zeros((G, G), dtype=np.float32)
    for _, r in ndf.iterrows():
        s, t = str(r[0]).strip(), str(r[1]).strip()
        if s in nti and t in nti: gt[nti[s], nti[t]] = 1.0
    return Xs, sn, gt

from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression

def compute_dcor_fast(X, cell_indices=None):
    if cell_indices is not None: X = X[cell_indices]
    C, G_ = X.shape; X_t = X.T.astype(np.float64)
    dist = np.abs(X_t[:, None, :] - X_t[None, :, :])
    rm = dist.mean(2, keepdims=True); cm = dist.mean(1, keepdims=True)
    gm = dist.mean((1,2), keepdims=True)[:,None,None]
    A = dist - rm - cm + gm
    Af = A.reshape(G_, -1).astype(np.float64)
    dsq = Af @ Af.T / (C*C); dv = np.diag(dsq).copy()
    dvp = np.sqrt(np.maximum(np.outer(dv,dv), 1e-30))
    d = np.sqrt(np.maximum(dsq/dvp, 0)); np.fill_diagonal(d,0); d = np.clip(d,0,1)
    return d.astype(np.float32)

def method_boyue(Xs, gn):
    X_std = (Xs - Xs.mean(0,keepdims=True)) / (Xs.std(0,keepdims=True)+1e-8)
    lw = LedoitWolf(assume_centered=False); lw.fit(X_std)
    P = lw.precision_.astype(np.float32)
    ci = RNG.choice(Xs.shape[0], min(dcor_cell_sample, Xs.shape[0]), replace=False)
    D = compute_dcor_fast(X_std, ci)
    Pt = torch.tensor(P).unsqueeze(0).to(device); Dt = torch.tensor(D).unsqueeze(0).to(device)
    probs = None
    for enc, head in zip(encoders, edge_heads):
        with torch.no_grad():
            p = torch.sigmoid(head(enc(Pt,Dt), Pt, Dt).float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    probs = (probs/len(encoders)).numpy(); np.fill_diagonal(probs, 0)
    return np.maximum(probs, probs.T).astype(np.float32)

def method_genie3(X, gn):
    n,ng = X.shape; Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    im = np.zeros((ng,ng), dtype=np.float32)
    for j in range(ng):
        fi = list(range(ng)); fi.remove(j)
        rf = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=42, n_jobs=-1)
        rf.fit(Xs[:,fi], Xs[:,j])
        for k,v in enumerate(rf.feature_importances_): im[fi[k],j] = v
    im = 0.5*(im+im.T); np.fill_diagonal(im,0); return im

def method_grnboost2(X, gn):
    n,ng = X.shape; Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    im = np.zeros((ng,ng), dtype=np.float32)
    for j in range(ng):
        fi = list(range(ng)); fi.remove(j)
        gb = GradientBoostingRegressor(n_estimators=50, max_features='sqrt', random_state=42)
        gb.fit(Xs[:,fi], Xs[:,j])
        for k,v in enumerate(gb.feature_importances_): im[fi[k],j] = v
    im = 0.5*(im+im.T); np.fill_diagonal(im,0); return im

def method_correlation(X, gn):
    c = np.corrcoef(((X-X.mean(0))/(X.std(0)+1e-8)).T).astype(np.float32)
    c = np.abs(c); np.fill_diagonal(c,0); return c

def method_mutual_info(X, gn):
    n,ng = X.shape; Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    im = np.zeros((ng,ng), dtype=np.float32)
    for j in range(ng):
        fi = list(range(ng)); fi.remove(j)
        mi = mutual_info_regression(Xs[:,fi], Xs[:,j], random_state=42)
        for k,v in enumerate(mi): im[fi[k],j] = v
    im = 0.5*(im+im.T); np.fill_diagonal(im,0); return im

def method_ppcor(X, gn):
    from sklearn.covariance import GraphicalLassoCV
    Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    lw = LedoitWolf(); lw.fit(Xs); prec = lw.precision_
    d = np.sqrt(np.maximum(np.diag(prec), 1e-30))
    pc = -prec/np.outer(d,d); np.fill_diagonal(pc,0)
    return np.abs(pc).astype(np.float32)

def method_glasso(X, gn):
    Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    lw = LedoitWolf(); lw.fit(Xs); prec = lw.precision_
    return np.abs(prec).astype(np.float32)

def method_dcor(X, gn):
    Xs = (X-X.mean(0))/(X.std(0)+1e-8)
    rng = np.random.RandomState(42)
    ci = rng.choice(Xs.shape[0], min(dcor_cell_sample, Xs.shape[0]), replace=False)
    return compute_dcor_fast(Xs, ci)

def method_random(X, gn):
    rng = np.random.RandomState(42); m = rng.rand(G,G).astype(np.float32)
    np.fill_diagonal(m,0); return m

METHODS = [
    ('GENIE3', method_genie3), ('GRNBoost2', method_grnboost2),
    ('Correlation', method_correlation), ('MutualInfo', method_mutual_info),
    ('PPCOR', method_ppcor), ('GLasso', method_glasso),
    ('dCor', method_dcor), ('Random', method_random), ('BoYue', method_boyue),
]

def topk_precision(gt, mat, k):
    s = mat.flatten(); t = gt.flatten()
    order = np.argsort(-s)
    return float(t[order[:k]].mean())

rows = []
for ds, (species, net_csv, ds_dir) in DATASETS.items():
    t0 = time.time()
    Xr, gnames = load_expression(ds_dir)
    ndf = load_network(species, net_csv)
    Xs, sn, gt = select_genes(Xr, gnames, ndf)
    n_gt = int(gt.sum())
    rand_base = n_gt / GSQ
    print(f"\n[{ds}] n_gt={n_gt} random_base={rand_base:.4f}", flush=True)
    for mname, mfn in METHODS:
        t1 = time.time()
        try:
            mat = mfn(Xs, sn)
            pk = topk_precision(gt, mat, n_gt)
            epr = pk / rand_base
            rows.append({'dataset': ds, 'method': mname, 'n_pred': n_gt, 'n_gt': n_gt,
                         'topk_precision': pk, 'EPR_ratio': epr, 'ECR': n_gt/GSQ})
            print(f"  {mname:<12} topk_p={pk:.4f} EPR={epr:.3f}  ({time.time()-t1:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {mname:<12} FAILED: {e}", flush=True)
            rows.append({'dataset': ds, 'method': mname, 'n_pred': n_gt, 'n_gt': n_gt,
                         'topk_precision': np.nan, 'EPR_ratio': np.nan, 'ECR': n_gt/GSQ})
    print(f"[{ds}] done ({time.time()-t0:.0f}s)", flush=True)

out = RESULT_ROOT / '2_benchmark_g200' / 'all_methods_epr_ratio_full.csv'
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved: {out}", flush=True)
