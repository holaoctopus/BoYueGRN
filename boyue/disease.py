"""DEG-Expanded disease pipeline (Results 3 & 4 of the paper).

End-to-end flow:
    1. Load expression data (h5ad, or directory with count.txt.gz + metadata.txt.gz)
    2. QC + normalization (scanpy: filter -> normalize_total -> log1p)
    3. Auto-select viable cell types (>= min_cells_per_cond cells in every condition)
    4. Per cell type: Wilcoxon stage-vs-baseline DEGs with NO gene cap
    5. Per cell type x stage: sliding-window directed GRN (TT direction specialist)
    6. Export directed edge lists (CSV) + sparse edge-prob matrices (npz)
       for downstream GO enrichment (see boyue.enrich).

Strategy (the DEG-Expanded strategy of the paper, optimized on NAFLD):
    - Gene selection: ALL DEGs per cell type (no 350-gene cap)
    - Direction: TT specialist on every edge type (outperforms NT)
    - Sliding window: 5 TFs + 195 targets with target rotation
    - Merge: edge_prob = max, dir_score = mean across windows

Data formats:
    - h5ad:        --celltype-key / --stage-key name the obs columns
    - directory:   count.txt.gz (genes x cells, tab-separated, gene names in index)
                   + metadata.txt.gz with columns Cell / celltype / site
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp_sparse
from scipy.stats import rankdata, norm
from sklearn.covariance import LedoitWolf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

from .model import (
    GraphTransformerEncoderV3,
    EdgeHeadV3,
    AsymmetricDirHead,
    DEFAULT_CONFIG,
)


# -- TF / model loading -----------------------------------------

def load_known_tfs(tf_csv):
    """Load known TF symbols from a one-column CSV (headerless)."""
    tfs = set(pd.read_csv(tf_csv, header=None)[0].str.upper().tolist())
    return {t for t in tfs if t and len(t) > 1}


def load_models(edge_ckpt, dir_ckpt, config=None, device=None):
    """Load edge_v3 + TT direction specialist checkpoints.

    Returns (enc, edge_head, dir_enc, dir_head, G).
    """
    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = config["G"]

    enc = GraphTransformerEncoderV3(
        G=G, d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config["dropout"],
        sd_prob=config["sd_prob"]).to(device)
    edge_head = EdgeHeadV3(d_model=config["d_model"], d_k=config["d_k"]).to(device)
    st = torch.load(edge_ckpt, map_location=device, weights_only=True)
    enc.load_state_dict(st["encoder"]); edge_head.load_state_dict(st["edge_head"])
    enc.eval(); edge_head.eval()

    dir_enc = GraphTransformerEncoderV3(
        G=G, d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config["dropout"],
        sd_prob=config["sd_prob"]).to(device)
    dir_head = AsymmetricDirHead(d_model=config["d_model"]).to(device)
    dst = torch.load(dir_ckpt, map_location=device, weights_only=True)
    dir_enc.load_state_dict(dst["encoder"]); dir_head.load_state_dict(dst["dir_head"])
    dir_enc.eval(); dir_head.eval()

    return enc, edge_head, dir_enc, dir_head, G


# -- Data loading -----------------------------------------------

def read_expression(path, celltype_key="cell_type", stage_key="condition"):
    """Load expression data into an AnnData object.

    Supports:
      - h5ad file (obs columns named by celltype_key / stage_key)
      - directory containing count.txt.gz + metadata.txt.gz
        (metadata columns: Cell / celltype / site)
    """
    import scanpy as sc

    path = Path(path)
    if path.is_dir():
        if not (path / "count.txt.gz").exists():
            raise FileNotFoundError(
                f"No count.txt.gz in {path}; provide an h5ad file or a "
                "directory with count.txt.gz + metadata.txt.gz")
        cell_to_idx = {}
        gene_names = []
        blocks = []
        for chunk in pd.read_csv(path / "count.txt.gz", sep="\t",
                                 compression="gzip", index_col=0,
                                 chunksize=2000):
            if not cell_to_idx:
                cell_to_idx = {c: i for i, c in enumerate(chunk.columns)}
            gene_names.extend(list(chunk.index))
            blocks.append(sp_sparse.csr_matrix(chunk.values.astype(np.float32)))
        X = sp_sparse.vstack(blocks).T.tocsr()
        meta = pd.read_csv(path / "metadata.txt.gz", sep="\t",
                           compression="gzip")
        meta = meta[meta["Cell"].isin(cell_to_idx)].copy()
        meta["_idx"] = meta["Cell"].map(cell_to_idx)
        meta = meta.sort_values("_idx").reset_index(drop=True)
        obs = pd.DataFrame({
            "barcode": meta["Cell"].values,
            celltype_key: meta["celltype"].astype(str).values,
            stage_key: meta["site"].astype(str).values,
        }, index=meta["Cell"].values)
        var = pd.DataFrame({"gene": gene_names}, index=gene_names)
        return sc.AnnData(X=X, obs=obs, var=var)

    adata = sc.read_h5ad(path)
    if celltype_key not in adata.obs.columns:
        raise KeyError(
            f"celltype key '{celltype_key}' not in obs columns: "
            f"{list(adata.obs.columns)}")
    if stage_key not in adata.obs.columns:
        raise KeyError(
            f"stage key '{stage_key}' not in obs columns: "
            f"{list(adata.obs.columns)}")
    adata.obs[celltype_key] = adata.obs[celltype_key].astype(str)
    adata.obs[stage_key] = adata.obs[stage_key].astype(str)
    return adata


# -- DEG computation --------------------------------------------

def _ranksums_vec(X_s, X_b):
    """Vectorized two-sided Wilcoxon rank-sum test across all genes.

    Matches scipy.stats.ranksums exactly (same rank-average tie handling and
    normal approximation), but computes all genes in one pass:
        ranked = rankdata of the pooled (n_s + n_b) x G matrix per column
        s      = sum of ranks in the stage group
        z      = (s - expected) / sqrt(n_s * n_b * (n_s + n_b + 1) / 12)
        p      = 2 * Phi(-|z|)

    Args:
        X_s: (n_s, G) expression of the stage group.
        X_b: (n_b, G) expression of the baseline group.
    Returns:
        (z, p) arrays of length G.
    """
    n1, n2 = X_s.shape[0], X_b.shape[0]
    pooled = np.concatenate([X_s, X_b], axis=0)
    ranked = rankdata(pooled, axis=0)
    s = ranked[:n1].sum(axis=0)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    p = 2 * norm.sf(np.abs(z))
    return z, p


def compute_degs(expr_ct, all_genes, obs_condition, deg_comparisons,
                 baseline, min_cells=50, fdr_thr=0.05, l2fc_thr=0.5,
                 known_tfs=None):
    """Wilcoxon stage-vs-baseline, FDR<0.05, |log2FC|>0.5, UNION (no cap).

    expr_ct: (n_cells, n_genes) dense array
    obs_condition: pd.Series aligned with expr_ct rows
    deg_comparisons: list of (stage, baseline) tuples
    Returns deg_df with columns [gene, max_abs_log2fc, min_fdr, n_stages_sig, is_tf].
    """
    base_mask = (obs_condition == baseline).values
    n_base = int(base_mask.sum())
    if n_base < min_cells:
        return pd.DataFrame()
    per_stage = []
    for stage, _ in deg_comparisons:
        stage_mask = (obs_condition == stage).values
        n_stage = int(stage_mask.sum())
        if n_stage < min_cells:
            continue
        X_b = expr_ct[base_mask]
        X_s = expr_ct[stage_mask]
        mean_b = X_b.mean(axis=0) + 1e-6
        mean_s = X_s.mean(axis=0) + 1e-6
        log2fc = np.log2(mean_s / mean_b)
        # Vectorized Wilcoxon rank-sum across all genes at once.
        # Constant genes yield z=0 -> p=1, matching the old per-gene
        # behavior where such columns were skipped (p kept at 1).
        _, p_vals = _ranksums_vec(X_s, X_b)
        valid = ~np.isnan(p_vals)
        fdr = np.ones(len(p_vals))
        if valid.sum() > 0:
            _, fdr_v, _, _ = multipletests(p_vals[valid], method="fdr_bh")
            fdr[valid] = fdr_v
        per_stage.append({"log2fc": log2fc, "fdr": fdr})
    if not per_stage:
        return pd.DataFrame()
    n_genes = len(all_genes)
    is_deg = np.zeros(n_genes, dtype=bool)
    max_abs_l2fc = np.zeros(n_genes)
    min_fdr = np.ones(n_genes)
    n_stages_sig = np.zeros(n_genes, dtype=int)
    for r in per_stage:
        sig = (r["fdr"] < fdr_thr) & (np.abs(r["log2fc"]) > l2fc_thr)
        is_deg |= sig
        max_abs_l2fc = np.maximum(max_abs_l2fc, np.abs(r["log2fc"]))
        min_fdr = np.minimum(min_fdr, r["fdr"])
        n_stages_sig += sig.astype(int)
    deg_df = pd.DataFrame({
        "gene": all_genes, "max_abs_log2fc": max_abs_l2fc,
        "min_fdr": min_fdr, "n_stages_sig": n_stages_sig,
        "is_tf": [g in known_tfs for g in all_genes] if known_tfs else False,
    })
    return deg_df[is_deg].copy()


# -- Sliding-window inference -----------------------------------

def fast_dcor(X):
    """Distance correlation matrix (G x G)."""
    C, Gv = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    A_flat = np.zeros((Gv, C * C), dtype=np.float64)
    for i in range(Gv):
        xi = X_c[:, i]
        d = np.abs(xi[:, None].astype(np.float64) - xi[None, :].astype(np.float64))
        A = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
        A_flat[i] = A.ravel()
    dcov2 = A_flat @ A_flat.T / (C * C)
    dvar = dcov2.diagonal().copy()
    dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
    dcor = np.sqrt(np.maximum(dcov2, 0)) / dvp
    np.fill_diagonal(dcor, 1.0); dcor = np.clip(dcor, 0, 1)
    del A_flat, dcov2, dvar, dvp; gc.collect()
    return dcor.astype(np.float32)


def make_infer_window(enc, edge_head, dir_enc, dir_head, device):
    @torch.no_grad()
    def infer_window(P_t, D_t):
        h_e = enc(P_t, D_t)
        ep = torch.sigmoid(edge_head(h_e, P_t, D_t).float())[0].cpu().numpy()
        h_d = dir_enc(P_t, D_t)
        ds = torch.sigmoid(dir_head(h_d, P_t, D_t).float())[0].cpu().numpy()
        np.fill_diagonal(ep, 0); np.fill_diagonal(ds, 0)
        return ep, ds
    return infer_window


def expanded_deg_grn(X, all_deg_genes, tf_genes_in_data, gene_to_idx,
                     enc, edge_head, dir_enc, dir_head, device, G_val,
                     top_k=50, tf_per_win=5, max_cells=500, max_tfs=150,
                     verbose=False):
    """Sliding-window GRN on ALL DEG genes (no cap).

    Returns (ep_max, ds_avg, n_windows).
    max_tfs caps the number of TFs used (top-N by the caller's ordering,
    |log2FC| descending) to keep window counts manageable.
    """
    infer_window = make_infer_window(enc, edge_head, dir_enc, dir_head, device)
    if max_tfs is not None and len(tf_genes_in_data) > max_tfs:
        if verbose:
            print(f"    Capping TFs: {len(tf_genes_in_data)} -> {max_tfs} "
                  f"(top by |log2FC|)")
        tf_genes_in_data = list(tf_genes_in_data)[:max_tfs]
    tf_set = set(tf_genes_in_data)
    n_genes = len(all_deg_genes)
    deg2local = {g: i for i, g in enumerate(all_deg_genes)}

    # Pre-screen: correlation on DEG genes only
    rng_ps = np.random.RandomState(42)
    n_sub = min(500, X.shape[0])
    deg_global_idx = [gene_to_idx[g] for g in all_deg_genes if g in gene_to_idx]
    X_deg = X[:, deg_global_idx]
    X_sub_ps = X_deg[rng_ps.choice(X_deg.shape[0], n_sub, replace=False)]
    if verbose:
        print(f"    Pre-screen (n={n_sub}, {n_genes} DEGs)...", end="", flush=True)
    corr_mat = np.corrcoef(X_sub_ps.T).astype(np.float32)
    if verbose:
        print(" done", flush=True)

    candidates = {}
    for tf in tf_genes_in_data:
        if tf not in deg2local:
            continue
        ti = deg2local[tf]
        corrs = np.abs(corr_mat[ti])
        sorted_idx = np.argsort(corrs)[::-1]
        cands = []
        for idx in sorted_idx:
            if idx == ti:
                continue
            g = all_deg_genes[idx]
            if g in tf_set:
                continue
            cands.append((g, float(corrs[idx])))
            if len(cands) >= top_k:
                break
        candidates[tf] = cands
    del corr_mat; gc.collect()

    tf_list = sorted(candidates.keys(), key=lambda t: len(candidates[t]),
                     reverse=True)
    target_set = set()
    for cs in candidates.values():
        for g, _ in cs:
            target_set.add(g)
    targets = list(target_set)
    target_slots = G_val - tf_per_win
    windows = []
    for bs in range(0, len(tf_list), tf_per_win):
        batch_tfs = tf_list[bs:bs + tf_per_win]
        if len(batch_tfs) < 2:
            continue
        batch_targets = targets[:]
        n_rot = (-(-len(batch_targets) // target_slots)
                 if len(batch_targets) > target_slots else 1)
        for rot in range(n_rot):
            rs = rot * target_slots
            rot_targets = batch_targets[rs:rs + target_slots]
            wg = list(batch_tfs) + list(rot_targets)
            ws = set(wg)
            for g in all_deg_genes:
                if len(wg) >= G_val:
                    break
                if g not in ws:
                    wg.append(g); ws.add(g)
            windows.append(wg[:G_val])
    if verbose:
        print(f"    Windows: {len(windows)}")

    ep_max = np.zeros((n_genes, n_genes), dtype=np.float32)
    ds_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
    ds_cnt = np.zeros((n_genes, n_genes), dtype=np.float32)

    rng = np.random.RandomState(42)
    X_sub = X_deg[rng.choice(X_deg.shape[0],
                             min(max_cells, X_deg.shape[0]), replace=False)]

    for wi, wgenes in enumerate(windows):
        if verbose and wi % 50 == 0:
            print(f"      Window {wi + 1}/{len(windows)}...", flush=True)
        win_local = [deg2local[g] for g in wgenes if g in deg2local]
        win_names = [g for g in wgenes if g in deg2local]
        if len(win_local) < 10:
            continue
        X_win = X_sub[:, win_local]
        mu = X_win.mean(0, keepdims=True)
        std = np.maximum(X_win.std(0, keepdims=True), 1e-8)
        X_std = (X_win - mu) / std
        try:
            lw = LedoitWolf(); lw.fit(X_std)
            P_mat = lw.precision_.astype(np.float32)
            D_mat = fast_dcor(X_std)
        except Exception:
            continue
        G_act = len(win_local)
        if G_act < G_val:
            pad = G_val - G_act
            P_full = np.pad(P_mat, ((0, pad), (0, pad)))
            D_full = np.pad(D_mat, ((0, pad), (0, pad)))
        else:
            P_full, D_full = P_mat, D_mat
        P_t = torch.from_numpy(P_full).unsqueeze(0).to(device)
        D_t = torch.from_numpy(D_full).unsqueeze(0).to(device)
        ep_w, ds_w = infer_window(P_t, D_t)
        for i, gi in enumerate(win_names):
            li = deg2local[gi]
            for j, gj in enumerate(win_names):
                lj = deg2local[gj]
                ep_max[li, lj] = max(ep_max[li, lj], ep_w[i, j])
                ds_sum[li, lj] += ds_w[i, j]
                ds_cnt[li, lj] += 1
        del P_t, D_t; gc.collect()

    valid = ds_cnt > 0
    ds_avg = np.zeros_like(ds_sum)
    ds_avg[valid] = ds_sum[valid] / ds_cnt[valid]
    return ep_max, ds_avg, len(windows)


# -- High-level pipeline ----------------------------------------

def _slug(s):
    return str(s).lower().replace("/", "_").replace(" ", "_")


def run_disease(
    path,
    out_dir,
    *,
    celltype_key="cell_type",
    stage_key="condition",
    baseline=None,
    stages=None,
    tf_csv=None,
    edge_ckpt=None,
    dir_ckpt=None,
    device=None,
    config=None,
    normalize=True,
    min_genes=200,
    min_cells_gene=3,
    min_cells_per_cond=100,
    min_cells_deg=50,
    min_deg=50,
    fdr_thr=0.05,
    l2fc_thr=0.5,
    edge_threshold=0.2,
    top_k=50,
    tf_per_win=5,
    max_cells=500,
    max_tfs=150,
    max_cells_input=1000,
    case_label="case",
    verbose=True,
):
    """Run the full DEG-Expanded pipeline and export results to out_dir.

    Output layout:
        out_dir/grn/{ct}/{stage}_edges.csv       directed edge list
        out_dir/grn/{ct}/{stage}_edge_prob.npz   sparse edge-prob matrix
        out_dir/grn/{ct}/genes.json              DEG gene set metadata
        out_dir/summary/deg_summary.csv          per cell type DEG counts
        out_dir/summary/grn_size.csv             per cell type x stage edge counts
    """
    t_start = time.time()
    out_dir = Path(out_dir)
    grn_dir = out_dir / "grn"
    summary_dir = out_dir / "summary"
    grn_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device={device}")
        print("Loading models (TT specialist)...")
    known_tfs = load_known_tfs(tf_csv)
    enc, edge_head, dir_enc, dir_head, G_val = load_models(
        edge_ckpt, dir_ckpt, config=config, device=device)
    if verbose:
        print(f"  G={G_val}, {len(known_tfs)} known TFs")

    # -- 1. Load + QC + normalize --
    if verbose:
        print(f"\n[1/5] Loading data from {path} ...")
    adata = read_expression(path, celltype_key=celltype_key,
                            stage_key=stage_key)
    if baseline is None:
        raise ValueError("--baseline is required")
    all_conditions = sorted(adata.obs[stage_key].unique())
    if stages is None:
        stages = [c for c in all_conditions if c != baseline]
    adata = adata[adata.obs[stage_key].isin([baseline] + list(stages))].copy()
    if normalize:
        sc_pp = __import__("scanpy").pp
        sc_pp.filter_cells(adata, min_genes=min_genes)
        sc_pp.filter_genes(adata, min_cells=min_cells_gene)
        sc_pp.normalize_total(adata, target_sum=1e4)
        sc_pp.log1p(adata)
    if verbose:
        print(f"  {adata.shape} after {'QC+norm' if normalize else 'load'}")
        print(f"  {stage_key}: "
              f"{adata.obs[stage_key].value_counts().to_dict()}")

    # -- 2. Auto-select viable cell types --
    if verbose:
        print("\n[2/5] Cell type viability:")
    conds_all = [baseline] + list(stages)
    viable_types = []
    for ct in adata.obs[celltype_key].unique():
        if ct in ("Unknown", "nan", "NaN"):
            continue
        counts = {c: int(((adata.obs[celltype_key] == ct)
                          & (adata.obs[stage_key] == c)).sum())
                  for c in conds_all}
        ok = all(v >= min_cells_per_cond for v in counts.values())
        if verbose:
            print(f"  {ct:<16s}: " + "  ".join(
                f"{c}={counts[c]:,}" for c in conds_all) + f"  {'OK' if ok else 'skip'}")
        if ok:
            viable_types.append(ct)
    if verbose:
        print(f"  Viable: {viable_types}")

    deg_rows, size_rows = [], []
    n_ct_done = 0
    for ct in viable_types:
        ct_data = adata[adata.obs[celltype_key] == ct].copy()
        if ct_data.n_obs < 100:
            continue
        if verbose:
            print(f"\n{'=' * 60}\n[{ct}] {ct_data.n_obs} cells")
        all_genes = list(ct_data.var_names)
        gene_to_idx = {g: i for i, g in enumerate(all_genes)}
        obs_cond = ct_data.obs[stage_key]

        # -- 3. DEGs (no cap) --
        if verbose:
            print("  Computing DEGs (no cap)...")
        X = (ct_data.X.toarray() if sp_sparse.issparse(ct_data.X)
             else np.asarray(ct_data.X))
        deg_df = compute_degs(
            X, all_genes, obs_cond,
            [(s, baseline) for s in stages], baseline,
            min_cells=min_cells_deg, fdr_thr=fdr_thr, l2fc_thr=l2fc_thr,
            known_tfs=known_tfs)
        n_deg = len(deg_df)
        n_tf = int(deg_df["is_tf"].sum()) if n_deg else 0
        if verbose:
            print(f"  DEGs: {n_deg} ({n_tf} TFs)")
        if n_deg < min_deg:
            if verbose:
                print(f"  [SKIP] too few DEGs ({n_deg})")
            continue
        deg_sorted = deg_df.sort_values(["max_abs_log2fc", "n_stages_sig"],
                                        ascending=[False, False])
        all_deg_genes = deg_sorted["gene"].tolist()
        tf_in_data = [g for g in all_deg_genes if g in known_tfs]
        if verbose:
            print(f"  Selected: {len(all_deg_genes)} genes "
                  f"({len(tf_in_data)} TFs)")

        ct_slug = _slug(ct)
        ct_dir = grn_dir / ct_slug
        ct_dir.mkdir(parents=True, exist_ok=True)
        genes_meta = {
            "case": case_label, "cell_type": ct,
            "baseline": baseline, "stages": stages,
            "tf_genes": tf_in_data, "all_deg_genes": all_deg_genes,
            "n_total_genes": len(all_genes), "n_deg": n_deg, "n_tf": n_tf,
            "method": "deg_expanded_tt",
        }
        with open(ct_dir / "genes.json", "w") as f:
            json.dump(genes_meta, f)
        del X; gc.collect()
        deg_rows.append({"cell_type": ct, "n_cells": ct_data.n_obs,
                         "n_total_genes": len(all_genes), "n_deg": n_deg,
                         "n_tf": n_tf})

        # -- 4. Per-stage sliding-window GRN (baseline included) --
        for cond in [baseline] + list(stages):
            cond_mask = ct_data.obs[stage_key] == cond
            X_ct = ct_data[cond_mask].X
            if sp_sparse.issparse(X_ct):
                X_ct = X_ct.toarray().astype(np.float32)
            else:
                X_ct = np.asarray(X_ct, dtype=np.float32)
            n_cells = X_ct.shape[0]
            if n_cells < min_cells_deg:
                if verbose:
                    print(f"  {cond}: only {n_cells} cells, skip")
                continue
            if n_cells > max_cells_input:
                rng = np.random.RandomState(42)
                X_ct = X_ct[rng.choice(n_cells, max_cells_input,
                                       replace=False)]
            if verbose:
                print(f"\n  {cond}: {n_cells} cells", end="", flush=True)
            t0 = time.time()
            ep, ds, n_wins = expanded_deg_grn(
                X_ct, all_deg_genes, tf_in_data, gene_to_idx,
                enc, edge_head, dir_enc, dir_head, device, G_val,
                top_k=top_k, tf_per_win=tf_per_win, max_cells=max_cells,
                max_tfs=max_tfs, verbose=verbose)
            n_edges = int((ep > edge_threshold).sum())
            if verbose:
                print(f"    -> {n_wins} windows, {n_edges:,} edges "
                      f"({time.time() - t0:.1f}s)", flush=True)
            size_rows.append({"cell_type": ct, "stage": cond,
                              "n_cells": n_cells, "n_windows": n_wins,
                              "n_edges": n_edges})

            # -- 5. Export --
            ep_sparse = sp_sparse.csr_matrix(ep)
            sp_sparse.save_npz(ct_dir / f"{_slug(cond)}_edge_prob.npz",
                               ep_sparse)
            tf_set = set(tf_in_data)
            rows = []
            for i in range(len(all_deg_genes)):
                for j in range(len(all_deg_genes)):
                    if i == j or ep[i, j] < edge_threshold:
                        continue
                    if ds[i, j] <= ds[j, i]:
                        continue
                    rows.append({
                        "source": all_deg_genes[i],
                        "target": all_deg_genes[j],
                        "edge_prob": float(ep[i, j]),
                        "confidence": float(abs(ds[i, j] - ds[j, i])),
                        "edge_type": ("TF->TF" if all_deg_genes[i] in tf_set
                                      and all_deg_genes[j] in tf_set
                                      else "TF->non-TF"),
                    })
            edges_df = pd.DataFrame(rows).sort_values(
                "edge_prob", ascending=False).reset_index(drop=True)
            edges_df.to_csv(ct_dir / f"{_slug(cond)}_edges.csv", index=False)
            if verbose:
                print(f"    Exported {len(edges_df)} directed edges")
            del ep, ds, ep_sparse, X_ct, edges_df; gc.collect()
        n_ct_done += 1

    pd.DataFrame(deg_rows).to_csv(summary_dir / "deg_summary.csv", index=False)
    pd.DataFrame(size_rows).to_csv(summary_dir / "grn_size.csv", index=False)
    if verbose:
        print(f"\n{'=' * 60}\nDone in {(time.time() - t_start) / 60:.1f} min")
        print(f"  Cell types processed: {n_ct_done}")
        print(f"  Output: {out_dir}/grn, {out_dir}/summary")
    return out_dir
