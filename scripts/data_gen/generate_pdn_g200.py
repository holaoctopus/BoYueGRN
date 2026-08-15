#!/usr/bin/env python
"""Phase 1.5: Generate P+D+N cache from existing P+A cache (G=200).
Regenerates X from seed, computes D (dCor) and N (ANM) on GPU.
Output: dataset_XXXXX.npz with P, D, N, A
"""
import torch, numpy as np, time, sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(os.environ.get("BOYUE_CACHE_PA", _PROJECT_ROOT / "processed_data" / "phase1_cache_g200"))
OUTPUT_DIR = Path(os.environ.get("BOYUE_CACHE", _PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda')
BATCH_SIZE = 10  # process N datasets at once on GPU

# -- SCM Generator (must match original) ------------------
def regenerate_X(G, C, seed):
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
    return X

# -- GPU dCor (N=1000 subsample) --------------------------
def compute_D_gpu(X_tensor):
    """X_tensor: (N, G) on GPU. Returns D: (G, G) distance correlation."""
    N, G = X_tensor.shape
    # Subsample to N=1000 for speed
    if N > 1000:
        idx = torch.randperm(N, device=device)[:1000]
        X = X_tensor[idx]
        N = 1000
    else:
        X = X_tensor
    
    X_c = X - X.mean(dim=0, keepdim=True)
    
    # Pairwise absolute differences: (N, N, G)
    xi = X_c.unsqueeze(0)
    xj = X_c.unsqueeze(1)
    D_dist = (xi - xj).abs()
    
    # Double-center
    row_mean = D_dist.mean(dim=0, keepdim=True)
    col_mean = D_dist.mean(dim=1, keepdim=True)
    grand_mean = D_dist.mean(dim=(0,1), keepdim=True).unsqueeze(0)
    A_mat = D_dist - row_mean - col_mean + grand_mean
    
    # dCov matrix: A_flat^T @ A_flat / N^2
    A_flat = A_mat.reshape(N*N, G)
    dcov2 = (A_flat.T @ A_flat) / (N * N)
    
    dvar = dcov2.diagonal()
    dvar_ij = dvar.unsqueeze(0) * dvar.unsqueeze(1)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0)) / torch.sqrt(torch.clamp(dvar_ij, min=1e-16))
    
    # Fix diagonal
    dcor.fill_diagonal_(1.0)
    return dcor.cpu().numpy().astype(np.float32)

# -- GPU ANM (cubic polynomial, fully batched) ------------
def compute_N_gpu(X_tensor):
    """X_tensor: (N, G) on GPU. Returns N: (G, G) skew-symmetric asymmetry."""
    N, G = X_tensor.shape
    if N > 1000:
        idx = torch.randperm(N, device=device)[:1000]
        X = X_tensor[idx]
        N = 1000
    else:
        X = X_tensor
    
    X_c = X - X.mean(dim=0, keepdim=True)
    
    # Polynomial features for all genes at once
    x3 = X_c ** 3; x2 = X_c ** 2; x1 = X_c
    ones = torch.ones(N, 1, device=device)
    D_all = torch.stack([x3, x2, x1, ones.expand(-1, G)], dim=1)  # (N, 4, G)
    
    tss = (X_c ** 2).sum(dim=0)  # (G,)
    
    # Batch: D^T D for all genes: (G, 4, 4)
    DTD = torch.einsum('nag,nbg->gab', D_all, D_all)
    # Batch: D^T X for all genes: (G, 4, G)
    DTX = torch.einsum('nag,nc->gac', D_all, X_c)
    
    # Batch solve: all G linear systems at once
    reg = 1e-6 * torch.eye(4, device=device).unsqueeze(0)
    coeffs = torch.linalg.solve(DTD + reg, DTX)  # (G, 4, G)
    
    # Batch prediction: regressor i predicting all targets j
    pred = torch.einsum('nfi,ift->int', D_all, coeffs)  # (G, N, G)
    
    # RSS for all pairs
    diff = X_c.unsqueeze(0) - pred  # (G, N, G)
    rss = (diff ** 2).sum(dim=1)  # (G, G)
    
    r2_forward = 1 - rss / (tss.unsqueeze(0) + 1e-8)
    N_mat = r2_forward.T - r2_forward  # skew-symmetric
    return N_mat.cpu().numpy().astype(np.float32)

# -- Main --------------------------------------------------
def main():
    files = sorted(CACHE_DIR.glob("dataset_*.npz"))
    total = len(files)
    print(f"Found {total} cached datasets")
    print(f"Generating P+D+N for all...")
    
    skipped = 0
    t_start = time.time()
    for idx, f in enumerate(files):
        out_file = OUTPUT_DIR / f"dataset_{idx:05d}.npz"
        if out_file.exists():
            skipped += 1
            continue
        d = np.load(f)
        P = d['P']
        A = d['A']
        G_val = int(d['G'])
        C_val = int(d['C'])
        seed_val = int(d['seed'])
        
        # Regenerate X using G_val and C_val from npz (not hardcoded)
        X_np = regenerate_X(G_val, C_val, seed_val)
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        
        # Compute D and N
        D = compute_D_gpu(X_t)
        N = compute_N_gpu(X_t)
        
        # Save
        np.savez_compressed(out_file, P=P, D=D, N=N, A=A, G=G_val, C=C_val, seed=seed_val)
        
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            done = idx + 1 - skipped
            eta = elapsed / max(done, 1) * (total - idx - 1)
            print(f"[{idx+1}/{total}] {elapsed/60:.0f}min elapsed, ETA {eta/60:.0f}min, skipped={skipped}")
            sys.stdout.flush()
    
    elapsed = time.time() - t_start
    print(f"\nDone! {total} datasets in {elapsed/60:.1f}min ({elapsed/total:.2f}s each)")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
