"""BoYue model definitions.

Three components matching the paper:
1. GraphTransformerEncoderV3: 8-layer Transformer, P+D dual-channel input
2. EdgeHeadV3: edge existence prediction
3. AsymmetricDirHead: directed regulation prediction (i->j vs j->i)

These classes must match the training scripts exactly so that checkpoints
saved by train_gt_g200_edge_v3.py / train_gt_g200_dir_specialist.py can be
loaded without remapping.

Reference: see scripts/train/train_gt_g200_edge_v3.py for the canonical
training code that produced the released checkpoints.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Default architecture config (matches training) -------------
DEFAULT_CONFIG = {
    "G": 200,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8,
    "dropout": 0.1,
    "sd_prob": 0.1,        # stochastic depth probability
    "d_k": 128,            # EdgeHead projection dim
}


class GraphTransformerEncoderV3(nn.Module):
    """Graph Transformer encoder with stochastic depth.

    Input: P (B, G, G) precision matrix, D (B, G, G) distance correlation matrix.
    Output: h (B, G, d_model) node embeddings.

    v3 design: P+D dual-channel input (N/ANM feature removed from input,
    only used as direction loss target during training).
    """

    def __init__(self, G=200, d_model=512, n_heads=8, n_layers=8,
                 dropout=0.1, sd_prob=0.1):
        super().__init__()
        self.G = G
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.sd_prob = sd_prob

        # Input projection: 2*G -> d_model (P and D concatenated per node)
        self.feat_proj = nn.Linear(2 * G, d_model)

        # Edge bias: 2 input features (P, D) -> n_heads bias terms
        self.edge_bias_proj = nn.Linear(2, n_heads)

        # LayerNorms (pre-norm: q, k, v each; output)
        self.ln_qs = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_kvs = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_vs = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_outs = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Attention projections
        self.w_qs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.w_ks = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.w_vs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.w_os = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # FFN
        self.ffn1s = nn.ModuleList([nn.Linear(d_model, 4 * d_model) for _ in range(n_layers)])
        self.ffn2s = nn.ModuleList([nn.Linear(4 * d_model, d_model) for _ in range(n_layers)])

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, P, D_mat):
        """P, D_mat: (B, G, G). Returns h: (B, G, d_model)."""
        B, Gv = P.shape[0], P.shape[1]
        h = self.feat_proj(torch.cat([P, D_mat], dim=-1))  # (B, Gv, d_model)

        # Edge features: (P_ij, D_ij) -> (B, Gv, Gv, 2)
        edge_feat = torch.stack([P.unsqueeze(-1), D_mat.unsqueeze(-1)], dim=-1)
        edge_feat = edge_feat.view(B, Gv, Gv, 2)
        edge_bias = self.edge_bias_proj(edge_feat)  # (B, Gv, Gv, n_heads)

        for i in range(self.n_layers):
            # Stochastic depth: skip layer with probability sd_prob during training
            if self.training and torch.rand(1).item() < self.sd_prob:
                continue

            q = self.ln_qs[i](h)
            k = self.ln_kvs[i](h)
            v = self.ln_vs[i](h)

            dh = self.d_model // self.n_heads
            q = self.w_qs[i](q).view(B, Gv, self.n_heads, dh).transpose(1, 2)
            k = self.w_ks[i](k).view(B, Gv, self.n_heads, dh).transpose(1, 2)
            v = self.w_vs[i](v).view(B, Gv, self.n_heads, dh).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
            attn = attn + edge_bias.permute(0, 3, 1, 2)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout_attn(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, Gv, self.d_model)
            out = self.dropout_ffn(self.w_os[i](out))
            h = h + out

            # FFN residual
            residual = h
            h_norm = self.ln_outs[i](h)
            h = residual + self.dropout_ffn(
                self.ffn2s[i](F.gelu(self.ffn1s[i](h_norm))))

        return h


class EdgeHeadV3(nn.Module):
    """Edge existence head.

    Combines dot-product attention score (Q@K^T) with an MLP over (P_ij, D_ij)
    edge features to produce edge existence logits.
    """

    def __init__(self, d_model=512, d_k=128):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.feat_score = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h, P, D_mat):
        """h: (B, G, d_model), P/D: (B, G, G). Returns logits (B, G, G)."""
        Q = self.W_q(h)
        K = self.W_k(h)
        score = Q @ K.transpose(1, 2) / math.sqrt(self.d_k)
        edge_feat = torch.stack([P, D_mat], dim=-1)
        bias = self.feat_score(edge_feat).squeeze(-1)
        return score + bias


class AsymmetricDirHead(nn.Module):
    """Asymmetric direction head.

    Predicts direction logits for each ordered pair (i, j). The direction of
    edge i->j vs j->i is determined by comparing logits[i,j] vs logits[j,i].
    Direction confidence = sigmoid(|logits[i,j] - logits[j,i]|).

    Architecture: separate src/tgt projections of node embeddings, combined
    with an MLP over (P_ij, D_ij) edge features.
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.tgt_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.edge_net = nn.Sequential(
            nn.Linear(2, 128), nn.GELU(),
            nn.Linear(128, 1))

    def forward(self, h, P, D_mat):
        """h: (B, G, d_model), P/D: (B, G, G). Returns direction logits (B, G, G)."""
        h_src = self.src_proj(h)
        h_tgt = self.tgt_proj(h)
        score = torch.bmm(h_src, h_tgt.transpose(1, 2))
        edge_feat = torch.cat([P.unsqueeze(-1), D_mat.unsqueeze(-1)], dim=-1)
        edge_bias = self.edge_net(edge_feat).squeeze(-1)
        return score + edge_bias
