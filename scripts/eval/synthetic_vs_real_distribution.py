#!/usr/bin/env python
"""Synthetic-vs-real expression-distribution comparison (supports Fig. 2a).

Quantifies how well the synthetic SCM expression distribution matches the six
BEELINE scRNA-seq datasets. Compares POSITIVE (x>0) log1p expression values,
since zero counts are dominated by capture/sequencing depth rather than GRN
biology; the zero-fraction is reported separately as a sparsity statistic.

For each BEELINE dataset it reports, against the synthetic pool:
  - pos_mean      : mean of positive log1p expression
  - zero_frac     : fraction of zero entries (sparsity)
  - KS_pos        : Kolmogorov-Smirnov distance on positive values
  - OVL_pos       : histogram overlap coefficient on positive values

Inputs
------
Synthetic  : regenerated from the phase1_pdn_cache_g200 .npz seeds via
             regenerate_X (identical copy of data_gen/generate_pdn_g200.py),
             i.e. clip(0)+log1p SCM expression.
Real       : $BOYUE_DATA/BEELINE/BEELINE-data/inputs/scRNA-Seq/{ds}/ExpressionData.csv
             (gene x cell log-normalised matrix; transposed to cell x gene).

Outputs
-------
CSV  : $BOYUE_ROOT/results/synthetic/synthetic_vs_real_distribution.csv
Fig  : $BOYUE_ROOT/results/synthetic/synthetic_vs_real_distribution.{png,pdf}
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT, DATA_ROOT

DATASETS = ['mDC', 'mHSC-E', 'mHSC-GM', 'mHSC-L', 'hESC', 'hHep']
COLORS6 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
N_SYNTH = 200          # number of synthetic datasets to pool
EXPR_DIR = DATA_ROOT / "BEELINE" / "BEELINE-data" / "inputs" / "scRNA-Seq"
CACHE = Path(os.environ.get("BOYUE_CACHE",
            PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200"))
OUT_DIR = PROJECT_ROOT / "results" / "synthetic"


def regenerate_X(G, C, seed):
    """Identical copy of data_gen/generate_pdn_g200.py:regenerate_X (CPU only).

    Regenerates the SCM expression matrix for a cached dataset from its seed.
    Preferential-attachment DAG (parents from earlier-indexed nodes with
    probability proportional to in-degree+1), additive-noise structural
    equations (~50% linear / 50% GELU), then clip(0)+log1p.
    """
    rng = np.random.RandomState(seed)
    A = np.zeros((G, G), dtype=np.float32)
    for i in range(1, 5):
        A[rng.randint(0, i), i] = 1.0
    indeg = A.sum(axis=0).copy()
    for i in range(5, G):
        total = indeg.sum() + 0.01
        probs = (indeg + 1) / total
        probs[i:] = 0
        np_ = max(1, min(rng.poisson(5), i))
        parents = rng.choice(G, size=np_, replace=False, p=probs / probs.sum())
        A[parents, i] = 1.0
        indeg = A.sum(axis=0)
    in_deg = A.sum(axis=0).copy()
    q = [int(i) for i in range(G) if in_deg[i] == 0]
    topo = []
    while q:
        n = q.pop(0); topo.append(n)
        for c in np.where(A[n] > 0)[0]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(int(c))
    W = rng.uniform(-1.5, 2.0, (G, G)).astype(np.float32) * A
    is_linear = rng.rand(G) < 0.5
    X = np.zeros((C, G), dtype=np.float32)
    for node in topo:
        parents = np.where(A[:, node] > 0)[0]
        if len(parents) == 0:
            X[:, node] = rng.randn(C).astype(np.float32) * 0.5
        else:
            z = (W[parents, node] * X[:, parents]).sum(axis=1)
            z = z + rng.randn(C).astype(np.float32) * 0.3
            if is_linear[node]:
                h = z
            else:
                h = 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))
            X[:, node] = h + rng.randn(C).astype(np.float32) * 2.0
    X = X.clip(0, None)
    X = np.log1p(X)
    return X


def overlap_coef(a, b, lo, hi, nbins=500):
    """Histogram overlap coefficient (area under min of the two densities)."""
    e = np.linspace(lo, hi, nbins + 1)
    ha, _ = np.histogram(a, bins=e, density=True)
    hb, _ = np.histogram(b, bins=e, density=True)
    return float((np.minimum(ha, hb) * (e[1] - e[0])).sum())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- synthetic pool ---
    npz_files = sorted(CACHE.glob("dataset_*.npz"))[:N_SYNTH]
    if not npz_files:
        raise FileNotFoundError(f"No synthetic cache .npz under {CACHE}")
    syn_all = []
    for f in npz_files:
        d = np.load(f)
        syn_all.append(regenerate_X(int(d['G']), int(d['C']), int(d['seed'])).ravel())
    syn_all = np.concatenate(syn_all)
    syn_pos = syn_all[syn_all > 0]
    print(f"Synthetic ({len(npz_files)} datasets): "
          f"pos-mean={syn_pos.mean():.3f} zero-frac={100*(syn_all==0).mean():.0f}%")

    # --- real datasets ---
    rows, real_pos = [], {}
    for ds in DATASETS:
        p = EXPR_DIR / ds / "ExpressionData.csv"
        if not p.exists():
            print(f"  [skip] {ds}: {p} not found"); continue
        Xr = pd.read_csv(p, index_col=0).values.astype(np.float64).T  # cell x gene
        pos = Xr[Xr > 0].ravel()
        real_pos[ds] = pos
        ks = stats.ks_2samp(syn_pos, pos).statistic
        ov = overlap_coef(syn_pos, pos, 0.0, max(syn_pos.max(), pos.max()))
        rows.append(dict(dataset=ds,
                         pos_mean=round(float(pos.mean()), 3),
                         zero_frac=round(float((Xr <= 0).mean()), 3),
                         KS_pos=round(float(ks), 3),
                         OVL_pos=round(float(ov), 3)))
        print(f"  {ds:8s} pos-mean={pos.mean():.3f} zero-frac={100*(Xr<=0).mean():.0f}% "
              f"KS_pos={ks:.3f} OVL_pos={ov:.3f}")

    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(dataset='Synthetic',
                           pos_mean=round(float(syn_pos.mean()), 3),
                           zero_frac=round(float((syn_all == 0).mean()), 3),
                           KS_pos=np.nan, OVL_pos=np.nan)
    csv_path = OUT_DIR / "synthetic_vs_real_distribution.csv"
    df.to_csv(csv_path, index=False)

    # --- figure (positive-expression densities) ---
    rng = np.random.RandomState(0)
    x = np.linspace(0, 13, 400)
    sp = syn_pos[rng.choice(len(syn_pos), min(80000, len(syn_pos)), replace=False)]
    kde_s = gaussian_kde(sp)(x)
    fig, ax = plt.subplots(figsize=(100 / 25.4, 60 / 25.4))
    ax.fill_between(x, kde_s, alpha=0.35, color='#333333', zorder=1)
    ax.plot(x, kde_s, color='#333333', lw=2.2, zorder=6,
            label=f'Synthetic (mean={syn_pos.mean():.2f})')
    for ds, c in zip(DATASETS, COLORS6):
        if ds not in real_pos:
            continue
        pos = real_pos[ds]
        ps = pos[rng.choice(len(pos), min(80000, len(pos)), replace=False)]
        ax.plot(x, gaussian_kde(ps)(x), color=c, lw=1.0,
                label=f'{ds} ({pos.mean():.2f})')
    ax.set_xlabel('Expression (log1p, positive)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title('Synthetic vs real expression distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, frameon=False, loc='upper right')
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "synthetic_vs_real_distribution.png"), dpi=300)
    fig.savefig(str(OUT_DIR / "synthetic_vs_real_distribution.pdf"))

    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved fig  -> {OUT_DIR}/synthetic_vs_real_distribution.png/.pdf")


if __name__ == "__main__":
    main()
