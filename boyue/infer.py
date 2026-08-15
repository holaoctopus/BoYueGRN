"""High-level inference API.

Usage:
    from boyue import load_ensemble
    infer = load_ensemble(
        edge_ckpts=["checkpoints/main/edge_v3_seed0.pt", ...],
        dir_ckpts=["checkpoints/main/dir_specialist_tf_non_tf_seed0.pt", ...],
    )
    grn = infer.predict_from_expression(X)  # X: (n_cells, n_genes)
    # grn: dict with 'edge_prob', 'direction', 'confidence', 'tf_power'

Or from precomputed P+D:
    grn = infer.predict_from_PD(P, D)
"""
import torch
import numpy as np
from pathlib import Path
from .model import GraphTransformerEncoderV3, EdgeHeadV3, AsymmetricDirHead, DEFAULT_CONFIG
from .stats import compute_P, compute_D


def _build_encoder(config):
    return GraphTransformerEncoderV3(
        G=config["G"], d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config["dropout"],
        sd_prob=config["sd_prob"])


def _load_edge_checkpoint(path, config, device):
    """Load edge_v3 checkpoint into encoder + edge_head."""
    encoder = _build_encoder(config).to(device)
    edge_head = EdgeHeadV3(d_model=config["d_model"], d_k=config["d_k"]).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(state['encoder'])
    edge_head.load_state_dict(state['edge_head'])
    encoder.eval(); edge_head.eval()
    return encoder, edge_head


def _load_dir_checkpoint(path, config, device):
    """Load dir_specialist checkpoint into encoder + dir_head."""
    encoder = _build_encoder(config).to(device)
    dir_head = AsymmetricDirHead(d_model=config["d_model"]).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(state['encoder'])
    dir_head.load_state_dict(state['dir_head'])
    encoder.eval(); dir_head.eval()
    return encoder, dir_head


def load_ensemble(edge_ckpts, dir_ckpts, config=None, device=None):
    """Load an ensemble of BoYue checkpoints.

    Args:
        edge_ckpts: list of paths to edge_v3 checkpoints (4 seeds recommended).
        dir_ckpts: list of paths to dir_specialist_tf_non_tf checkpoints.
        config: model config dict (default DEFAULT_CONFIG, matches paper).
        device: torch device (auto-detect if None).

    Returns:
        BoYueInferencer instance.
    """
    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_models = [_load_edge_checkpoint(p, config, device) for p in edge_ckpts]
    dir_models = [_load_dir_checkpoint(p, config, device) for p in dir_ckpts]

    return BoYueInferencer(edge_models, dir_models, config, device)


class BoYueInferencer:
    """High-level BoYue inference interface.

    Wraps an ensemble of edge + direction models. Provides:
    - predict_from_expression(X): compute P+D from raw expression, then infer GRN
    - predict_from_PD(P, D): infer GRN from precomputed statistics
    - tf_power(edge_prob, direction): compute TF regulatory power
    """

    def __init__(self, edge_models, dir_models, config, device):
        self.edge_models = edge_models  # list of (encoder, edge_head)
        self.dir_models = dir_models    # list of (encoder, dir_head)
        self.config = config
        self.device = device
        self.G = config["G"]

    @torch.no_grad()
    def predict_from_PD(self, P, D):
        """Infer directed GRN from precomputed P and D matrices.

        Args:
            P: (G, G) precision matrix (numpy or torch).
            D: (G, G) distance correlation matrix.

        Returns:
            dict with:
                'edge_prob': (G, G) edge existence probability (sigmoid).
                'direction': (G, G) +1 if i->j predicted, -1 if j->i, 0 if no edge.
                'confidence': (G, G) direction confidence = sigmoid(|logit_i_j - logit_j_i|).
                'tf_power': (G,) regulatory power per TF = sum of confidence on outgoing edges.
        """
        Pt = torch.tensor(P, dtype=torch.float32, device=self.device).unsqueeze(0)
        Dt = torch.tensor(D, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Edge existence: ensemble average
        edge_probs = []
        for enc, ehead in self.edge_models:
            h = enc(Pt, Dt)
            logits = ehead(h, Pt, Dt)
            edge_probs.append(torch.sigmoid(logits.float()))
        edge_prob = torch.stack(edge_probs, dim=0).mean(dim=0)[0]  # (G, G)

        # Direction: ensemble average
        dir_logits = []
        for enc, dhead in self.dir_models:
            h = enc(Pt, Dt)
            logits = dhead(h, Pt, Dt)
            dir_logits.append(logits.float())
        dir_logit = torch.stack(dir_logits, dim=0).mean(dim=0)[0]  # (G, G)

        # Direction: i->j if logit[i,j] > logit[j,i]
        direction = torch.zeros_like(dir_logit)
        mask = dir_logit > dir_logit.t()
        direction[mask] = 1.0       # i -> j
        direction[mask.t()] = -1.0  # j -> i (reverse)
        # where i==j direction stays 0

        # Confidence: how strongly the model prefers i->j over j->i
        confidence = torch.sigmoid(torch.abs(dir_logit - dir_logit.t()))

        # TF power: per-TF sum of direction confidence on outgoing edges
        # Only count edges where edge_prob exceeds threshold and direction is i->j
        edge_mask = (edge_prob > 0.5).float()
        outgoing = (direction > 0).float() * edge_mask
        tf_power = (outgoing * confidence).sum(dim=1)  # (G,)

        return {
            'edge_prob': edge_prob.cpu().numpy(),
            'direction': direction.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'tf_power': tf_power.cpu().numpy(),
        }

    def predict_from_expression(self, X, device=None, max_n=1000):
        """Compute P+D from expression matrix, then infer directed GRN.

        Args:
            X: (n_cells, n_genes) expression matrix (log1p normalized).
            device: device for P+D computation (defaults to inferencer device).
            max_n: cell subsample limit for D computation.

        Returns:
            dict from predict_from_PD.
        """
        if device is None:
            device = self.device
        P = compute_P(X)
        D = compute_D(X, device=device, max_n=max_n)
        return self.predict_from_PD(P, D)

    @staticmethod
    def directed_edges(grn, gene_names=None, edge_threshold=0.5, top_k=None):
        """Extract directed edge list from GRN prediction.

        Args:
            grn: dict from predict_from_PD/predict_from_expression.
            gene_names: list of gene names (length G). If None, use indices.
            edge_threshold: minimum edge_prob to include.
            top_k: if set, return only top-k edges by edge_prob.

        Returns:
            list of (source, target, edge_prob, direction, confidence) tuples.
        """
        ep = grn['edge_prob']
        dr = grn['direction']
        cf = grn['confidence']
        G = ep.shape[0]

        if gene_names is None:
            gene_names = [str(i) for i in range(G)]

        edges = []
        for i in range(G):
            for j in range(G):
                if i == j:
                    continue
                if ep[i, j] < edge_threshold:
                    continue
                if dr[i, j] > 0:  # i -> j
                    edges.append((gene_names[i], gene_names[j],
                                  float(ep[i, j]), 1, float(cf[i, j])))

        if top_k is not None:
            edges.sort(key=lambda e: -e[2])
            edges = edges[:top_k]

        return edges
