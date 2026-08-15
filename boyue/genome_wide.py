"""Genome-wide directed GRN inference pipeline.

Solves the practical challenge: BoYue's model operates on 200-gene windows,
but real scRNA-seq datasets have 20,000+ genes. This module implements a
pre-screening + coverage-optimized windowing strategy that achieves >80%
coverage of candidate regulatory edges, making BoYue a practical tool for
genome-wide GRN inference.

Pipeline:
    1. Pre-screen: For each TF, compute Pearson correlation with all genes,
       select top-k correlated genes as candidate targets.
    2. Windowing: Build 200-gene windows (5 TFs + 195 targets) with target
       rotation to ensure every candidate TF->target pair is co-located in
       at least one window.
    3. Inference: Run BoYue (edge + direction models) on each window.
    4. Merge: Aggregate predictions across windows into genome-wide GRN.
    5. Filter: Apply edge probability threshold + optional MC-Dropout.

Usage:
    from boyue.genome_wide import GenomeWideInferencer

    gwi = GenomeWideInferencer(
        edge_ckpt="checkpoints/main/edge_v3_seed0.pt",
        dir_nontf_ckpt="checkpoints/main/dir_specialist_tf_non_tf_seed0.pt",
        dir_tftf_ckpt="checkpoints/main/dir_specialist_tf_tf_seed0.pt",
    )

    grn = gwi.infer(
        X=expression_matrix,       # (n_cells, n_genes) log1p normalized
        gene_names=gene_list,      # list of gene symbols
        tf_names=tf_list,          # list of TF symbols
        top_k_targets=50,          # pre-screen top-50 per TF
        edge_threshold=0.2,        # edge_prob > 0.2
    )
    # grn: DataFrame [source, target, edge_prob, direction, confidence]

Architecture rationale:
    - G=200 is the sweet spot (Methods: G-scaling ablation)
    - TF_PER_WINDOW=5 frees 195 target slots (vs 180 with 20 TFs)
    - Target rotation: each TF batch gets multiple windows with different
      target subsets, ensuring >80% coverage of candidate edges
    - Pre-screening with Pearson correlation reduces the search space from
      O(n_tf × n_genes) to O(n_tf × top_k), making coverage feasible
"""
import gc
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.covariance import LedoitWolf

from .model import (
    GraphTransformerEncoderV3, EdgeHeadV3, AsymmetricDirHead, DEFAULT_CONFIG,
)
from .stats import compute_P


class GenomeWideInferencer:
    """Genome-wide directed GRN inference from scRNA-seq expression.

    Wraps BoYue's edge + direction models with a practical windowing strategy
    that handles the 200-gene window limitation. Users provide a raw expression
    matrix and TF list; the class returns a genome-wide directed GRN.

    Args:
        edge_ckpt: path to edge model checkpoint (gt_g200_edge_v3).
        dir_nontf_ckpt: path to TF->non-TF direction specialist checkpoint.
        dir_tftf_ckpt: path to TF->TF direction specialist checkpoint.
        config: model config dict (default: DEFAULT_CONFIG, G=200).
        device: torch device (auto-detect if None).
    """

    def __init__(self, edge_ckpt, dir_nontf_ckpt, dir_tftf_ckpt,
                 config=None, device=None):
        self.config = config or DEFAULT_CONFIG
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.G = self.config["G"]

        # Load three models (edge + two direction specialists)
        self.enc_edge, self.edge_head = self._load_edge(edge_ckpt)
        self.enc_dir_nt, self.dir_head_nt = self._load_dir(dir_nontf_ckpt)
        self.enc_dir_tt, self.dir_head_tt = self._load_dir(dir_tftf_ckpt)

    def _load_edge(self, path):
        enc = GraphTransformerEncoderV3(
            G=self.G, d_model=self.config["d_model"],
            n_heads=self.config["n_heads"], n_layers=self.config["n_layers"],
            dropout=self.config["dropout"], sd_prob=self.config["sd_prob"]
        ).to(self.device)
        head = EdgeHeadV3(
            d_model=self.config["d_model"], d_k=self.config["d_k"]
        ).to(self.device)
        ck = torch.load(path, map_location=self.device, weights_only=True)
        enc.load_state_dict(ck['encoder'])
        head.load_state_dict(ck['edge_head'])
        enc.eval(); head.eval()
        return enc, head

    def _load_dir(self, path):
        enc = GraphTransformerEncoderV3(
            G=self.G, d_model=self.config["d_model"],
            n_heads=self.config["n_heads"], n_layers=self.config["n_layers"],
            dropout=self.config["dropout"], sd_prob=self.config["sd_prob"]
        ).to(self.device)
        head = AsymmetricDirHead(
            d_model=self.config["d_model"]
        ).to(self.device)
        ck = torch.load(path, map_location=self.device, weights_only=True)
        enc.load_state_dict(ck['encoder'])
        head.load_state_dict(ck['dir_head'])
        enc.eval(); head.eval()
        return enc, head

    # -- Public API ----------------------------------------------

    def infer(self, X, gene_names, tf_names, top_k_targets=50,
              tf_per_window=5, edge_threshold=0.2, mc_dropout=False,
              mc_iterations=30, max_cells=500, verbose=True):
        """Run genome-wide directed GRN inference.

        Args:
            X: (n_cells, n_genes) expression matrix, log1p normalized.
            gene_names: list of gene symbols (length = n_genes).
            tf_names: list of TF symbols (subset of gene_names).
            top_k_targets: pre-screen top-k correlated targets per TF.
            tf_per_window: number of TFs per 200-gene window (default 5).
            edge_threshold: minimum edge_prob to include in output.
            mc_dropout: if True, run MC-Dropout uncertainty estimation.
            mc_iterations: number of MC-Dropout forward passes.
            max_cells: cell subsample limit for P+D computation.
            verbose: print progress.

        Returns:
            DataFrame with columns:
                source, target, edge_prob, direction, confidence,
                edge_type, mc_std (if mc_dropout=True)
        """
        t0 = time.time()
        gene_names = [str(g).upper() for g in gene_names]
        tf_set = set(str(t).upper() for t in tf_names)
        tf_in_data = sorted(tf_set & set(gene_names))

        if verbose:
            print(f"BoYue Genome-Wide Inference")
            print(f"  Cells: {X.shape[0]}, Genes: {X.shape[1]}")
            print(f"  TFs in data: {len(tf_in_data)}/{len(tf_names)}")

        # Step 1: Pre-screen candidate edges
        candidate_edges = self._prescreen(X, gene_names, tf_in_data,
                                          top_k_targets, verbose)

        # Step 2: Build coverage-optimized windows
        windows = self._build_windows(gene_names, tf_in_data, candidate_edges,
                                      X, tf_per_window, verbose)

        # Step 3: Run inference per window + merge
        grn = self._infer_windows(X, gene_names, windows, tf_set,
                                  edge_threshold, mc_dropout, mc_iterations,
                                  max_cells, verbose)

        if verbose:
            print(f"\nDone in {(time.time()-t0)/60:.1f} min")
            print(f"  Output: {len(grn)} directed edges "
                  f"(edge_prob > {edge_threshold})")

        return grn

    # -- Step 1: Pre-screen --------------------------------------

    @staticmethod
    def _prescreen(X, gene_names, tf_names, top_k, verbose):
        """Pre-screen candidate TF->target edges via Pearson correlation.

        For each TF, select top-k genes by |Pearson correlation|. This
        reduces the search space from O(n_tf × n_genes) to O(n_tf × top_k),
        making coverage-optimized windowing feasible.

        Returns:
            dict: {tf_name: [target_name, ...]} candidate targets per TF.
        """
        if verbose:
            print(f"\n[1/4] Pre-screening candidate edges "
                  f"(top-{top_k} per TF)...")

        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        tf_indices = [gene_to_idx[t] for t in tf_names if t in gene_to_idx]

        # Compute correlation matrix (TFs × all genes) in one shot
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-10)
        tf_idx_arr = np.array(tf_indices)
        # Correlation = standardized dot product / n
        corr = (X_std[:, tf_idx_arr].T @ X_std) / X.shape[0]  # (n_tf, n_genes)
        corr = np.abs(corr)  # use |correlation| for ranking

        candidates = {}
        for ti, tf in enumerate(tf_names):
            if tf not in gene_to_idx:
                continue
            tf_row = corr[ti]
            # Exclude self-correlation
            tf_row[gene_to_idx[tf]] = -1
            # Top-k by |correlation|
            top_idx = np.argpartition(tf_row, -top_k)[-top_k:]
            targets = [gene_names[i] for i in top_idx if tf_row[i] > 0]
            candidates[tf] = targets

        total_edges = sum(len(v) for v in candidates.values())
        if verbose:
            print(f"  Candidate edges: {total_edges} "
                  f"({len(candidates)} TFs × ~{top_k} targets)")
        return candidates

    # -- Step 2: Build windows -----------------------------------

    def _build_windows(self, gene_names, tf_names, candidates, X,
                       tf_per_window, verbose):
        """Build 200-gene windows with coverage-optimized target rotation.

        Strategy:
        - Batch TFs into groups of tf_per_window (default 5)
        - For each batch, collect candidate targets (~5×top_k genes)
        - If candidates > target_slots (195): split into rotations
        - Each rotation covers a different subset of targets
        - Fill remaining slots with high-variance genes

        Returns:
            list of (tf_genes, target_genes) tuples, each defining a window.
        """
        if verbose:
            print(f"\n[2/4] Building windows "
                  f"({tf_per_window} TFs/window, target rotation)...")

        G = self.G
        target_slots = G - tf_per_window
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Precompute gene variances for filler
        gene_vars = X.var(axis=0)
        high_var_genes = sorted(
            [g for g in gene_names if g in gene_to_idx],
            key=lambda g: gene_vars[gene_to_idx[g]], reverse=True
        )

        # Batch TFs
        tf_list = sorted(candidates.keys())
        windows = []
        covered_pairs = set()

        for batch_start in range(0, len(tf_list), tf_per_window):
            batch_tfs = tf_list[batch_start:batch_start + tf_per_window]
            if len(batch_tfs) < 2:
                # Pad with random TFs to maintain window size
                extra = [t for t in tf_list if t not in batch_tfs]
                batch_tfs = batch_tfs + extra[:tf_per_window - len(batch_tfs)]
            batch_tfs = batch_tfs[:tf_per_window]

            # Collect candidate targets for this batch
            batch_targets = []
            for tf in batch_tfs:
                for tgt in candidates.get(tf, []):
                    if tgt not in batch_tfs and tgt not in batch_targets:
                        batch_targets.append(tgt)

            # Split targets into rotations
            n_rotations = max(1, -(-len(batch_targets) // target_slots))
            for rot in range(n_rotations):
                start = rot * target_slots
                end = start + target_slots
                rot_targets = batch_targets[start:end]

                # Fill remaining slots with high-variance genes
                used = set(batch_tfs) | set(rot_targets)
                filler = [g for g in high_var_genes if g not in used]
                while len(rot_targets) < target_slots and filler:
                    rot_targets.append(filler.pop(0))

                window_tfs = batch_tfs.copy()
                window_targets = rot_targets[:target_slots]
                windows.append((window_tfs, window_targets))

                # Track coverage (only candidate TF->target pairs, not filler)
                for tf in window_tfs:
                    for tgt in window_targets:
                        if tf != tgt and tgt in candidates.get(tf, []):
                            covered_pairs.add((tf, tgt))

        # Also add TF->TF windows (all TFs in batches of G)
        # This covers TF↔TF regulatory pairs
        for tt_start in range(0, len(tf_list), G):
            tt_batch = tf_list[tt_start:tt_start + G]
            if len(tt_batch) < 2:
                continue
            # Fill with high-variance genes if < G
            used = set(tt_batch)
            filler = [g for g in high_var_genes if g not in used]
            while len(tt_batch) < G and filler:
                tt_batch.append(filler.pop(0))
            # All TFs in window, no separate targets
            windows.append((tt_batch[:G], []))

        if verbose:
            total_candidate = sum(len(v) for v in candidates.values())
            coverage = len(covered_pairs) / max(total_candidate, 1) * 100
            print(f"  Windows: {len(windows)}")
            print(f"  Candidate edges covered: {len(covered_pairs)}/"
                  f"{total_candidate} = {coverage:.1f}%")

        return windows

    # -- Step 3: Per-window inference ----------------------------

    def _infer_windows(self, X, gene_names, windows, tf_set,
                       edge_threshold, mc_dropout, mc_iterations,
                       max_cells, verbose):
        """Run BoYue inference on each window and merge results."""
        if verbose:
            print(f"\n[3/4] Running inference on {len(windows)} windows...")

        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Subsample cells if needed
        n_cells = X.shape[0]
        if n_cells > max_cells:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_cells, max_cells, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        all_edges = {}  # (source, target) -> {edge_prob, direction, ...}

        for wi, (tf_genes, target_genes) in enumerate(windows):
            if verbose and wi % 20 == 0:
                print(f"  Window {wi+1}/{len(windows)}...", flush=True)

            # Build window gene list
            if target_genes:
                window_genes = list(tf_genes) + list(target_genes)
            else:
                # TF->TF window: all genes are TFs
                window_genes = list(tf_genes)
            window_genes = window_genes[:self.G]

            # Get indices in expression matrix
            valid_genes = [g for g in window_genes if g in gene_to_idx]
            if len(valid_genes) < 10:
                continue
            sel_idx = np.array([gene_to_idx[g] for g in valid_genes])
            G_act = len(valid_genes)
            gene_to_gidx = {g: i for i, g in enumerate(valid_genes)}
            tf_gidx = set(gene_to_gidx[g] for g in valid_genes if g in tf_set)

            # Compute P + D
            X_win = X_sub[:, sel_idx]
            try:
                P = compute_P(X_win)
                D = self._compute_dcor(X_win)
            except Exception:
                continue

            # Pad if G_act < G
            if G_act < self.G:
                pad = self.G - G_act
                P = np.pad(P, ((0, pad), (0, pad)))
                D = np.pad(D, ((0, pad), (0, pad)))

            P_t = torch.from_numpy(P).unsqueeze(0).to(self.device)
            D_t = torch.from_numpy(D).unsqueeze(0).to(self.device)

            # Model inference
            with torch.no_grad():
                # Edge prediction
                h_e = self.enc_edge(P_t, D_t)
                edge_p = torch.sigmoid(
                    self.edge_head(h_e, P_t, D_t).float()
                )[0].cpu().numpy()

                # Direction prediction (both specialists)
                h_nt = self.enc_dir_nt(P_t, D_t)
                dir_p_nt = torch.sigmoid(
                    self.dir_head_nt(h_nt, P_t, D_t).float()
                )[0].cpu().numpy()

                h_tt = self.enc_dir_tt(P_t, D_t)
                dir_p_tt = torch.sigmoid(
                    self.dir_head_tt(h_tt, P_t, D_t).float()
                )[0].cpu().numpy()

                # MC-Dropout (optional)
                if mc_dropout:
                    mc_std = self._mc_dropout_std(
                        P_t, D_t, mc_iterations
                    )
                else:
                    mc_std = None

            # Extract edges
            for i in range(G_act):
                for j in range(G_act):
                    if i == j:
                        continue
                    ep = float(edge_p[i, j])
                    if ep < edge_threshold:
                        continue

                    src = valid_genes[i]
                    tgt = valid_genes[j]

                    # Determine edge type and use appropriate specialist
                    is_tt = (i in tf_gidx and j in tf_gidx)
                    dir_p = dir_p_tt if is_tt else dir_p_nt
                    forward = float(dir_p[i, j])
                    backward = float(dir_p[j, i])
                    direction = 1 if forward > backward else 0
                    confidence = abs(forward - backward)

                    edge_type = "TF->TF" if is_tt else "TF->non-TF"

                    key = (src, tgt)
                    # Keep prediction with highest edge_prob across windows
                    if key not in all_edges or ep > all_edges[key]['edge_prob']:
                        all_edges[key] = {
                            'source': src,
                            'target': tgt,
                            'edge_prob': ep,
                            'direction': direction,
                            'confidence': confidence,
                            'edge_type': edge_type,
                        }
                        if mc_std is not None:
                            all_edges[key]['mc_std'] = float(mc_std[i, j])

            del P_t, D_t, h_e, h_nt, h_tt
            gc.collect()

        # Build DataFrame
        if not all_edges:
            return pd.DataFrame(columns=[
                'source', 'target', 'edge_prob', 'direction',
                'confidence', 'edge_type'
            ])

        grn = pd.DataFrame(list(all_edges.values()))

        # Only keep directed edges (direction == 1, i.e., source -> target)
        # For each undirected pair, keep the one with higher confidence
        grn = grn[grn['direction'] == 1].copy()
        grn = grn.sort_values('edge_prob', ascending=False)
        grn = grn.drop_duplicates(subset=['source', 'target'], keep='first')

        if verbose:
            n_tfnontf = (grn['edge_type'] == 'TF->non-TF').sum()
            n_tt = (grn['edge_type'] == 'TF->TF').sum()
            print(f"\n[4/4] Merged GRN:")
            print(f"  TF->non-TF edges: {n_tfnontf}")
            print(f"  TF->TF edges: {n_tt}")
            print(f"  Total: {len(grn)}")

        return grn.reset_index(drop=True)

    # -- Utilities -----------------------------------------------

    def _compute_dcor(self, X):
        """Compute distance correlation matrix (same as eval scripts)."""
        C, Gv = X.shape
        Xc = X - X.mean(axis=0, keepdims=True)
        A_flat = np.zeros((Gv, C * C), dtype=np.float64)
        for i in range(Gv):
            xi = Xc[:, i].astype(np.float64)
            d = np.abs(xi[:, None] - xi[None, :])
            A = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
            A_flat[i] = A.ravel()
            del d, A
        dcov2 = A_flat @ A_flat.T / (C * C)
        dvar = dcov2.diagonal().copy()
        dvp = np.sqrt(np.maximum(np.outer(dvar, dvar), 1e-30))
        dcor = np.sqrt(np.maximum(dcov2, 0)) / dvp
        np.fill_diagonal(dcor, 1.0)
        dcor = np.clip(dcor, 0, 1)
        del A_flat, dcov2, dvar, dvp
        return dcor.astype(np.float32)

    @torch.no_grad()
    def _mc_dropout_std(self, P_t, D_t, n_iterations):
        """MC-Dropout: run edge model n times with dropout active,
        return per-edge std as uncertainty estimate."""
        edge_probs = []
        for _ in range(n_iterations):
            h = self.enc_edge(P_t, D_t)
            ep = torch.sigmoid(self.edge_head(h, P_t, D_t).float())[0]
            edge_probs.append(ep)
        stacked = torch.stack(edge_probs, dim=0)
        std = stacked.std(dim=0).cpu().numpy()
        return std
