"""Statistical preprocessing: Ledoit-Wolf precision (P) + distance correlation (D).

These functions convert a raw expression matrix (cells x genes) into the
P+D input representation consumed by the GraphTransformerEncoderV3.

Reference: see scripts/data_gen/generate_pdn_g200.py for the canonical
implementation used during training.
"""
import numpy as np
import torch
from sklearn.covariance import LedoitWolf


def compute_P(X, n_jobs=1):
    """Ledoit-Wolf shrinkage precision matrix.

    Args:
        X: (n_cells, n_genes) expression matrix (log1p normalized recommended).
        n_jobs: parallel jobs for LedoitWolf fitting.

    Returns:
        P: (n_genes, n_genes) precision matrix (inverse of shrunk covariance).
    """
    lw = LedoitWolf()
    lw.fit(X)
    cov = lw.covariance_
    P = np.linalg.inv(cov)
    # Symmetrize to correct for numerical asymmetry
    P = (P + P.T) / 2.0
    return P.astype(np.float32)


def compute_D(X, device='cpu', max_n=1000):
    """Distance correlation matrix.

    Captures arbitrary (including nonlinear) dependencies between gene pairs.

    Args:
        X: (n_cells, n_genes) expression matrix.
        device: 'cpu' or 'cuda' for GPU acceleration.
        max_n: subsample cells to this number for speed (default 1000).

    Returns:
        D: (n_genes, n_genes) distance correlation matrix.
    """
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    N, G = Xt.shape
    if N > max_n:
        idx = torch.randperm(N, device=device)[:max_n]
        Xt = Xt[idx]
        N = max_n

    Xc = Xt - Xt.mean(dim=0, keepdim=True)

    # Pairwise absolute differences: (N, N, G)
    xi = Xc.unsqueeze(0)
    xj = Xc.unsqueeze(1)
    D_dist = (xi - xj).abs()

    # Double-centering
    row_mean = D_dist.mean(dim=0, keepdim=True)
    col_mean = D_dist.mean(dim=1, keepdim=True)
    grand_mean = D_dist.mean(dim=(0, 1), keepdim=True).unsqueeze(0)
    A_mat = D_dist - row_mean - col_mean + grand_mean

    # dCov matrix
    A_flat = A_mat.reshape(N * N, G)
    dcov2 = (A_flat.T @ A_flat) / (N * N)

    dvar = dcov2.diagonal()
    dvar_ij = dvar.unsqueeze(0) * dvar.unsqueeze(1)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0)) / torch.sqrt(
        torch.clamp(dvar_ij, min=1e-16))
    dcor.fill_diagonal_(1.0)

    return dcor.cpu().numpy().astype(np.float32)


def compute_PDN(X, device='cpu', max_n=1000, n_jobs=1, compute_N=False):
    """Compute P + D (+ optional N) statistics.

    Args:
        X: (n_cells, n_genes) expression matrix.
        device: 'cpu' or 'cuda'.
        max_n: cell subsample limit for D/N.
        n_jobs: parallel jobs for LedoitWolf.
        compute_N: if True, also compute ANM asymmetry feature N.
                   (N is only needed as direction loss target during training;
                    not required for inference.)

    Returns:
        dict with keys 'P', 'D' (and 'N' if compute_N=True).
    """
    P = compute_P(X, n_jobs=n_jobs)
    D = compute_D(X, device=device, max_n=max_n)

    out = {'P': P, 'D': D}

    if compute_N:
        out['N'] = _compute_N(X, device=device, max_n=max_n)

    return out


def _compute_N(X, device='cpu', max_n=1000):
    """ANM (Additive Noise Model) asymmetry feature.

    Used only as direction loss target during training. Computes R^2 of a
    cubic polynomial regression i->j minus j->i, yielding a skew-symmetric
    matrix. Not needed for inference with trained checkpoints.
    """
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    N, G = Xt.shape
    if N > max_n:
        idx = torch.randperm(N, device=device)[:max_n]
        Xt = Xt[idx]
        N = max_n

    Xc = Xt - Xt.mean(dim=0, keepdim=True)

    x3 = Xc ** 3
    x2 = Xc ** 2
    x1 = Xc
    ones = torch.ones(N, 1, device=device)
    D_all = torch.stack([x3, x2, x1, ones.expand(-1, G)], dim=1)  # (N, 4, G)

    tss = (Xc ** 2).sum(dim=0)

    DTD = torch.einsum('nag,nbg->gab', D_all, D_all)
    DTX = torch.einsum('nag,nc->gac', D_all, Xc)

    reg = 1e-6 * torch.eye(4, device=device).unsqueeze(0)
    coeffs = torch.linalg.solve(DTD + reg, DTX)

    pred = torch.einsum('nfi,ift->int', D_all, coeffs)
    diff = Xc.unsqueeze(0) - pred
    rss = (diff ** 2).sum(dim=1)

    r2_forward = 1 - rss / (tss.unsqueeze(0) + 1e-8)
    N_mat = r2_forward.T - r2_forward
    return N_mat.cpu().numpy().astype(np.float32)
