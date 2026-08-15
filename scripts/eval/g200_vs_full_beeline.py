"""
Quick evaluation: run BEELINE 6 datasets single (G=200) + read existing multi results.
Generate G=200 vs full-gene comparison CSV.
"""
import sys
import numpy as np, pandas as pd, torch, time, gc
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, CKPT_ROOT, RESULT_ROOT
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_gt_g200_edge_v3 import GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads, dropout, sd_prob, device
from train_gt_g200_dir_specialist import GraphTransformerEncoderV3 as DirEncoder, AsymmetricDirHead


torch.set_grad_enabled(False)

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
    for enc, head in edge_models:
        h = enc(Pt, Dt)
        p = torch.sigmoid(head(h, Pt, Dt).float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    probs = (probs / len(edge_models)).numpy()
    np.fill_diagonal(probs, 0)
    return np.maximum(probs, probs.T).astype(np.float32)

@torch.no_grad()
def predict_dir(P, D):
    """TT specialist direction prediction."""
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    h_d = tt_enc(Pt, Dt)
    ds = torch.sigmoid(tt_head(h_d, Pt, Dt).float()).squeeze(0).cpu().numpy()
    return ds

print("Loading edge models (4-seed ensemble)...")
edge_models = []
for s in range(4):
    ck = CKPT_ROOT / "main" / f"edge_v3_seed{s}.pt"
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=8, sd_prob=0.0).to(device)
    head = EdgeHeadV3(d_model=d_model).to(device)
    state = torch.load(ck, map_location=device, weights_only=False)
    enc.load_state_dict(state['encoder']); head.load_state_dict(state['edge_head'])
    enc.eval(); head.eval(); edge_models.append((enc, head))
print(f"  Edge: {len(edge_models)} seeds")

print("Loading TT direction specialist...")
tt_ck = CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt"
tt_enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=8, sd_prob=0.0).to(device)
tt_head = AsymmetricDirHead(d_model=d_model).to(device)
state = torch.load(tt_ck, map_location=device, weights_only=False)
tt_enc.load_state_dict(state['encoder']); tt_head.load_state_dict(state['dir_head'])
tt_enc.eval(); tt_head.eval()
print(f"  TT direction ready.")

@torch.no_grad()
def predict_edge(P, D):
    """4-seed ensemble edge prediction."""
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    probs = None
    for enc, head in edge_models:
        h = enc(Pt, Dt)
        p = torch.sigmoid(head(h, Pt, Dt).float()).detach().cpu().squeeze(0)
        probs = p if probs is None else probs + p
    probs = (probs / len(edge_models)).numpy()
    np.fill_diagonal(probs, 0)
    return np.maximum(probs, probs.T).astype(np.float32)

@torch.no_grad()
def predict_dir(P, D):
    """TT specialist direction prediction."""
    Pt = torch.from_numpy(P).unsqueeze(0).to(device)
    Dt = torch.from_numpy(D).unsqueeze(0).to(device)
    h_d = tt_enc(Pt, Dt)
    ds = torch.sigmoid(tt_head(h_d, Pt, Dt).float()).squeeze(0).cpu().numpy()
    return ds

def eval_all(ep, gt, mask):
    flat_s, flat_t = ep[mask], gt[mask]
    n_pos = flat_t.sum()
    if n_pos == 0:
        return {'AUROC':0.5,'AUPRC':0.0,'EPR':0.0,'P@0.2':0.0,'R@0.2':0.0,'F1@0.2':0.0,
                'n_pred@thr':0,'gt_pos':0,'gt_total':int(mask.sum())}
    auroc = roc_auc_score(flat_t, flat_s)
    auprc = average_precision_score(flat_t, flat_s)
    k = max(1, int(0.2 * flat_t.sum()))
    top_idx = np.argsort(flat_s)[::-1][:k]
    p_at = flat_t[top_idx].mean()
    r_at = flat_t[top_idx].sum() / max(n_pos, 1)
    f1_at = 2*p_at*r_at/max(p_at+r_at,1e-10)
    base_rate = n_pos / len(flat_t)
    epr = p_at / max(base_rate, 1e-10)
    return {'AUROC':auroc,'AUPRC':auprc,'EPR':epr,'P@0.2':p_at,'R@0.2':r_at,'F1@0.2':f1_at,
            'n_pred@thr':int(k),'gt_pos':int(n_pos),'gt_total':int(mask.sum())}

def eval_dir_acc(ep, ds, gt, thr=0.2):
    valid = (ep > thr) & (gt > 0)
    n = valid.sum()
    if n == 0: return {'DirAcc':0.0,'DirAcc_n':0}
    return {'DirAcc':float((ds[valid] > 0.5).mean()),'DirAcc_n':int(n)}

BL = {
    'mDC':    ('mouse/mDC-ChIP-seq-network.csv', 'mouse-tfs.csv', 'mDC/ExpressionData.csv'),
    'mHSC-E': ('mouse/mHSC-ChIP-seq-network.csv', 'mouse-tfs.csv', 'mHSC-E/ExpressionData.csv'),
    'mHSC-GM': ('mouse/mHSC-ChIP-seq-network.csv', 'mouse-tfs.csv', 'mHSC-GM/ExpressionData.csv'),
    'mHSC-L': ('mouse/mHSC-ChIP-seq-network.csv', 'mouse-tfs.csv', 'mHSC-L/ExpressionData.csv'),
    'hESC':   ('human/hESC-ChIP-seq-network.csv', 'human-tfs.csv', 'hESC/ExpressionData.csv'),
    'hHep':   ('human/HepG2-ChIP-seq-network.csv', 'human-tfs.csv', 'hHep/ExpressionData.csv'),
}

BEELINE_DIR = DATA_ROOT / "BEELINE"
EXPR_BASE = BEELINE_DIR / "BEELINE-data" / "inputs" / "scRNA-Seq"
all_results = []

for ds, (net_rel, tf_rel, expr_rel) in BL.items():
    print(f"\n  {ds}:")
    expr_path = EXPR_BASE / expr_rel
    if not expr_path.exists():
        expr_path = EXPR_BASE / "ExpressionData.csv"
    expr_df = pd.read_csv(expr_path, index_col=0)
    genes = expr_df.index.tolist()
    X_raw = expr_df.values.T.astype(np.float32)
    tf_df = pd.read_csv(BEELINE_DIR / tf_rel, header=None)
    tf_names = set(tf_df[0].str.upper())
    net_df = pd.read_csv(BEELINE_DIR / "Networks" / net_rel)
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    gt = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for _, r in net_df.iterrows():
        s, t = str(r.iloc[0]).upper(), str(r.iloc[1]).upper()
        if s in g2i and t in g2i: gt[g2i[s], g2i[t]] = 1.0

    n_genes = len(genes)
    mask = ~np.eye(n_genes, dtype=bool)
    X_std = (X_raw - X_raw.mean(0, keepdims=True)) / (X_raw.std(0, keepdims=True) + 1e-8)

    print(f"    single...", end='', flush=True)
    t0 = time.time()
    rng = np.random.RandomState(42)
    n_win = min(n_genes, G)
    sel_idx = rng.choice(n_genes, n_win, replace=False)
    X_win = X_std[:, sel_idx]
    X_pad = np.zeros((X_win.shape[0], G), dtype=np.float32)
    X_pad[:, :n_win] = X_win
    P, D = compute_P_D(X_pad)
    ep = predict_edge(P, D)[:n_win, :n_win]
    ds_pred = predict_dir(P, D)[:n_win, :n_win]
    ep_full = np.zeros((n_genes, n_genes), dtype=np.float32)
    ds_full = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i, gi in enumerate(sel_idx):
        for j, gj in enumerate(sel_idx):
            ep_full[gi, gj] = ep[i, j]
            ds_full[gi, gj] = ds_pred[i, j]
    edge_r = eval_all(ep_full, gt, mask)
    dir_r = eval_dir_acc(ep_full, ds_full, gt, thr=0.2)
    edge_r.update(dir_r)
    edge_r['dataset'] = ds; edge_r['source'] = 'BEELINE'; edge_r['coverage'] = 'single'; edge_r['windows'] = 1
    all_results.append(edge_r)
    dt = time.time() - t0
    print(f" AUROC={edge_r['AUROC']:.4f} EPR={edge_r['EPR']:.2f}x DirAcc={dir_r['DirAcc']:.4f} ({dt:.1f}s)")

print("\n  Loading existing multi results...")
multi_df = pd.read_csv(RESULT_ROOT / "3_full_gene" / "full_evaluation_matrix.csv")
beeline_multi = multi_df[(multi_df['source'] == 'BEELINE') & (multi_df['coverage'] == 'multi')].copy()
for _, row in beeline_multi.iterrows():
    r = row.to_dict()
    r['dataset'] = row['dataset']; r['source'] = 'BEELINE'; r['coverage'] = 'multi'
    all_results.append(r)

df = pd.DataFrame(all_results)
print("\n" + "="*70)
print("G=200 vs Full-gene (BEELINE)")
print("="*70)
print(f"\n{'Dataset':<10s} {'Cov':<8s} {'AUROC':>7s} {'AUPRC':>7s} {'EPR':>7s} {'DirAcc':>7s}")
print("-"*60)
for _, r in df.iterrows():
    print(f"  {r['dataset']:<10s} {r['coverage']:<8s} {r['AUROC']:>7.4f} {r['AUPRC']:>7.4f} {r['EPR']:>6.2f}x {r.get('DirAcc',0):>7.4f}")

df.to_csv(RESULT_ROOT / "3_full_gene" / "g200_vs_full_beeline.csv", index=False)
print(f"\nSaved: results/3_full_gene/g200_vs_full_beeline.csv")
print("DONE")
