#!/usr/bin/env python
"""Direction Specialist V2: Row-level asymmetric edge features.

Key change from V1: edge_bias was computed from symmetric [P[i,j], D[i,j]],
which cancels out in direction comparison. V2 adds row-level features:
  - Project P and D rows into "source" and "target" role spaces
  - row_score[i,j] = row_src[i] @ row_tgt[j]  (genuinely asymmetric)
  - Combine with node_score for final logit

Usage:
  python scripts/train/train_gt_g200_dir_specialist_v2.py --mode tf_non_tf --seed 0
  python scripts/train/train_gt_g200_dir_specialist_v2.py --mode tf_tf --seed 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score
import time, math, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gt_g200_edge_v3 import (
    PDNDataset, GraphTransformerEncoderV3,
    G, d_model, n_heads, n_layers, dropout, sd_prob, batch_size, device
)

# -- Config ----------------------------------------------
lr = 1e-4
warmup_steps = 1000
total_steps = 20000
log_every = 200
val_every = 2000
grad_clip = 1.0
label_smooth = 0.1
n_seeds = 4
d_row = 64  # dimension for row-level projections

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))
edge_ckpt_root = PROJECT_ROOT / "checkpoints" / "gt_g200_edge_v3"


# -- AsymmetricDirHead V2 --------------------------------
class AsymmetricDirHeadV2(nn.Module):
    """Direction head with row-level asymmetric edge features.
    
    Architecture:
      logit[i,j] = node_score[i,j] + edge_bias[i,j]
      
    node_score[i,j] = src_proj(h[i]) @ tgt_proj(h[j])  (node-level, captures TF identity)
    
    edge_bias[i,j]  = f(P[i,j], D[i,j], row_score[i,j], row_score[j,i])
    row_score[i,j]  = row_src(P_row_i, D_row_i) @ row_tgt(P_row_j, D_row_j)
    
    row_score is genuinely asymmetric because row_src ≠ row_tgt.
    """
    def __init__(self, d_model=512, d_row=64):
        super().__init__()
        # -- Node score (same as V1) --
        self.src_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.tgt_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        
        # -- Row-level projections (NEW) --
        # Compress P+D rows -> shared representation -> split into src/tgt roles
        self.row_compress = nn.Sequential(
            nn.Linear(2 * G, 256), nn.GELU(),
            nn.Linear(256, 2 * d_row))  # outputs [row_src_feat, row_tgt_feat]
        
        # -- Edge network: combine P/D and row scores --
        # Input: [P[i,j], D[i,j], row_score[i,j], row_score[j,i]]
        self.edge_net = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(),
            nn.Linear(128, 1))
    
    def forward(self, h, P, D):
        """h: (B, G, d_model), P: (B, G, G), D: (B, G, G)"""
        B_act, G_act = h.shape[0], h.shape[1]
        
        # -- Node score --
        h_src = self.src_proj(h)  # (B, G, d_model)
        h_tgt = self.tgt_proj(h)  # (B, G, d_model)
        node_score = torch.bmm(h_src, h_tgt.transpose(1, 2))  # (B, G, G)
        
        # -- Row-level asymmetric score --
        P_mat = P[:, :G_act, :G_act]  # (B, G, G)
        D_mat = D[:, :G_act, :G_act]  # (B, G, G)
        
        # Concatenate P and D rows per node -> (B, G, 2G)
        row_cat = torch.cat([P_mat, D_mat], dim=-1)
        
        # Compress -> split into src/tgt
        row_feat = self.row_compress(row_cat)  # (B, G, 2*d_row)
        r_src = row_feat[:, :, :d_row]         # (B, G, d_row)
        r_tgt = row_feat[:, :, d_row:]          # (B, G, d_row)
        
        # Asymmetric row score via dot product
        row_score = torch.bmm(r_src, r_tgt.transpose(1, 2))  # (B, G, G)
        
        # -- Edge network: combine all features --
        edge_feat = torch.stack([
            P_mat,                              # (B,G,G) - symmetric P[i,j]
            D_mat,                              # (B,G,G) - symmetric D[i,j]
            row_score,                          # (B,G,G) - asymmetric row_src[i]·row_tgt[j]
            row_score.transpose(1, 2),          # (B,G,G) - row_src[j]·row_tgt[i]
        ], dim=-1)  # (B, G, G, 4)
        
        edge_bias = self.edge_net(edge_feat).squeeze(-1)  # (B, G, G)
        
        return node_score + edge_bias


# -- Helper functions ------------------------------------
def smooth_bce_loss(logits, targets, smoothing=0.1):
    targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, targets_smooth)


def compute_tf_mask(A_mat):
    """Compute TF mask per node from adjacency matrix.
    TF = node with out-degree >= median non-zero out-degree.
    Returns (B, G) boolean tensor.
    """
    batch_tf_masks = []
    for b in range(A_mat.shape[0]):
        A = A_mat[b]
        out_deg = A.sum(dim=1)
        nonzero_deg = out_deg[out_deg > 0]
        if len(nonzero_deg) == 0:
            thresh = 1.0
        else:
            thresh = nonzero_deg.median().item()
        tf_mask = out_deg >= max(thresh, 1)
        batch_tf_masks.append(tf_mask)
    return torch.stack(batch_tf_masks, dim=0)


def get_specialist_mask(A_mat, tf_mask, mode):
    """Compute edge mask for specialist training."""
    B_act, G_act = A_mat.shape[0], A_mat.shape[1]
    edge_mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0))
    
    if mode == 'all':
        return edge_mask
    
    tf_src = tf_mask.unsqueeze(2).expand(B_act, G_act, G_act)
    tf_tgt = tf_mask.unsqueeze(1).expand(B_act, G_act, G_act)
    
    if mode == 'tf_non_tf':
        role_mask = tf_src & (~tf_tgt)
    elif mode == 'tf_tf':
        role_mask = tf_src & tf_tgt
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return edge_mask & role_mask


@torch.no_grad()
def evaluate(encoder, dir_head, loader, mode='all'):
    """Evaluate DirAcc on specified edge type only."""
    encoder.eval(); dir_head.eval()
    total_correct, total_pairs = 0, 0
    for P, D_mat, N_mat, A_mat in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        h = encoder(P, D_mat)
        logits = dir_head(h, P, D_mat)
        label = A_mat.to(device).float()
        
        tf_mask = compute_tf_mask(A_mat)
        specialist_mask = get_specialist_mask(A_mat, tf_mask, mode)
        
        pred = (logits > 0).float()
        total_correct += (pred[specialist_mask] == label[specialist_mask]).sum().item()
        total_pairs += specialist_mask.sum().item()
    return total_correct / max(total_pairs, 1)


# -- Main training --------------------------------------
def train_seed(seed, mode):
    checkpoint_root = PROJECT_ROOT / "checkpoints" / f"gt_g200_dir_specialist_v2_{mode}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    edge_ckpt = edge_ckpt_root / f"seed_{seed}" / "best.pt"
    
    print(f"\n{'#'*60}")
    print(f"# DIR SPECIALIST V2 SEED {seed}: mode={mode}")
    print(f"# Row-level asymmetric edge features (d_row={d_row})")
    print(f"# Edge checkpoint: {edge_ckpt}")
    print(f"{'#'*60}")
    
    # -- Data --
    full_ds = PDNDataset(cache_dir)
    n_total = len(full_ds)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    indices = np.random.RandomState(42 + seed).permutation(n_total)
    train_ds = Subset(full_ds, indices[:n_train])
    val_ds = Subset(full_ds, indices[n_train:n_train + n_val])
    test_ds = Subset(full_ds, indices[n_train + n_val:])
    
    def collate(b):
        return (torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                torch.stack([x[2] for x in b]),
                torch.stack([x[3] for x in b]))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate, num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=4, pin_memory=True)
    
    # -- Model --
    encoder = GraphTransformerEncoderV3(
        G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        sd_prob=sd_prob).to(device)
    state = torch.load(edge_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(state['encoder'])
    dir_head = AsymmetricDirHeadV2(d_model=d_model, d_row=d_row).to(device)
    
    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dir = sum(p.numel() for p in dir_head.parameters())
    # Count parameters in new components
    n_params_row = sum(p.numel() for p in dir_head.row_compress.parameters())
    n_params_edge = sum(p.numel() for p in dir_head.edge_net.parameters())
    print(f"Encoder: {n_params_enc:,}")
    print(f"DirHeadV2: {n_params_dir:,} (row_compress: {n_params_row:,}, edge_net: {n_params_edge:,})")
    
    # -- Train --
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(dir_head.parameters()),
        lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    step, best_val = 0, 0.0
    loader_iter = iter(train_loader)
    t_start = time.time()
    
    while step < total_steps:
        try:
            P, D_mat, N_mat, A_mat = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            P, D_mat, N_mat, A_mat = next(loader_iter)
        
        P, D_mat, A_mat = P.to(device), D_mat.to(device), A_mat.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            h = encoder(P, D_mat)
            logits = dir_head(h, P, D_mat)
            
            label = A_mat.float()
            
            tf_mask = compute_tf_mask(A_mat)
            specialist_mask = get_specialist_mask(A_mat, tf_mask, mode)
            
            if specialist_mask.sum() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = smooth_bce_loss(
                    logits[specialist_mask], label[specialist_mask], label_smooth)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(dir_head.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(step)
        step += 1
        
        if step % log_every == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1)
            eta = (total_steps - step) / max(sps, 0.001)
            with torch.no_grad():
                pred = (logits.float() > 0).float()
                train_acc = (pred[specialist_mask] == label[specialist_mask]).float().mean().item()
            n_specialist = specialist_mask.sum().item()
            print(f"[{mode}_v2][seed{seed}][{step:5d}/{total_steps}] loss={loss.item():.4f} "
                  f"train_DirAcc={train_acc:.4f} n_edges={n_specialist} "
                  f"lr={get_lr(step):.2e} {sps:.1f}s/s ETA={eta/60:.0f}min")
        
        if step % val_every == 0:
            val_diracc = evaluate(encoder, dir_head, val_loader, mode=mode)
            best = "*" if val_diracc > best_val else " "
            if val_diracc > best_val:
                best_val = val_diracc
                torch.save({'encoder': encoder.state_dict(),
                            'dir_head': dir_head.state_dict()},
                           checkpoint_dir / "best.pt")
            elapsed = time.time() - t_start
            print(f"  >>> {mode}_v2 SEED{seed} STEP {step}: val_DirAcc={val_diracc:.4f}  "
                  f"best={best_val:.4f} {best}  [{elapsed/60:.1f}min]")
    
    # -- Final --
    elapsed = time.time() - t_start
    print(f"\n{mode}_v2 SEED {seed} DONE in {elapsed/60:.1f}min")
    
    torch.save({'encoder': encoder.state_dict(),
                'dir_head': dir_head.state_dict()},
               checkpoint_dir / "final.pt")
    
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder']); dir_head.load_state_dict(ckpt['dir_head'])
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    test_diracc = evaluate(encoder, dir_head, test_loader, mode=mode)
    print(f"  val DirAcc={best_val:.4f}  test DirAcc={test_diracc:.4f}")
    
    return test_diracc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['tf_non_tf', 'tf_tf'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    train_seed(args.seed, args.mode)


if __name__ == "__main__":
    main()
