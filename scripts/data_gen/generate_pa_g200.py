#!/usr/bin/env python
"""Quick: Generate G=200 P+A cache for fast P+D+N validation."""
import numpy as np, time, sys
from sklearn.covariance import LedoitWolf
from pathlib import Path

import os
G, C, N_DATASETS = 200, 1000, 10000
OUTPUT_DIR = Path(os.environ.get("BOYUE_CACHE", Path(__file__).resolve().parent.parent.parent / "processed_data" / "phase1_cache_g200"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _make_scm(seed):
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
            if in_deg[c] == 0: q.append(int(c))
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
                h = 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (z + 0.044715 * z**3)))
            X[:, node] = h + rng.randn(C).astype(np.float32) * 2.0
    X = X.clip(0, None); X = np.log1p(X)
    return X, A

def compute_P(X):
    lw = LedoitWolf(assume_centered=False)
    lw.fit(X)
    prec = lw.precision_
    diag_abs = np.abs(np.diag(prec))
    denom = np.sqrt(np.maximum(np.outer(diag_abs, diag_abs), 1e-16))
    P = -prec / denom
    np.fill_diagonal(P, 0.0)
    return P.astype(np.float32)

t0 = time.time()
skipped = 0
for idx in range(N_DATASETS):
    out_file = OUTPUT_DIR / f"dataset_{idx:05d}.npz"
    if out_file.exists():
        skipped += 1
        continue
    seed_val = idx + 10000
    X, A = _make_scm(seed_val)
    P = compute_P(X)
    np.savez_compressed(out_file,
                        P=P, A=A, G=G, C=C, seed=seed_val)
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - t0
        done = idx + 1 - skipped
        eta = elapsed / max(done, 1) * (N_DATASETS - idx - 1)
        print(f"[{idx+1}/{N_DATASETS}] {elapsed/60:.0f}min elapsed, ETA {eta/60:.0f}min, skipped={skipped}")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nDone! {N_DATASETS} datasets (G={G}, C={C}) in {elapsed/60:.1f}min")
