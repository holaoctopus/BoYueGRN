#!/usr/bin/env python
"""Direction Specialist training: separate models for TF->non-TF and TF->TF edges.

Strategy: Train two direction heads from the same edge_v3 encoder, each
specializing in one edge type. At inference, ensemble all three models
(dir_v4 + tf_non_tf + tf_tf) for final direction prediction.

TF definition: node with out-degree >= tf_threshold in the A matrix.
The threshold is computed from training data as the median non-zero out-degree.

Usage:
  python scripts/train/train_gt_g200_dir_specialist.py --mode tf_non_tf --seed 0
  python scripts/train/train_gt_g200_dir_specialist.py --mode tf_tf --seed 0
  python scripts/train/train_gt_g200_dir_specialist.py --mode tf_non_tf --ensemble
  python scripts/train/train_gt_g200_dir_specialist.py --mode tf_tf --ensemble
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
from boyue.model import AsymmetricDirHead

# -- Config ----------------------------------------------
lr = 1e-4
warmup_steps = 1000
total_steps = 20000
log_every = 200
val_every = 2000
grad_clip = 1.0
label_smooth = 0.1
n_seeds = 4
tf_threshold = None  # None = auto-compute from training data (median non-zero out-degree)

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))
edge_ckpt_root = PROJECT_ROOT / "checkpoints" / "gt_g200_edge_v3"


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
        out_deg = A.sum(dim=1)  # (G,)
        nonzero_deg = out_deg[out_deg > 0]
        if len(nonzero_deg) == 0:
            thresh = 1.0
        else:
            thresh = nonzero_deg.median().item()
        tf_mask = out_deg >= max(thresh, 1)
        batch_tf_masks.append(tf_mask)
    return torch.stack(batch_tf_masks, dim=0)  # (B, G)


def get_specialist_mask(A_mat, tf_mask, mode):
    """Compute edge mask for specialist training.

    mode='tf_non_tf': edges where source is TF AND target is NOT TF
    mode='tf_tf':     edges where source is TF AND target is TF
    mode='all':       all edge pairs (baseline)

    Returns boolean mask (B, G, G) for edge pairs only.
    """
    B, G = A_mat.shape[0], A_mat.shape[1]

    # Base edge mask: A[i,j]=1 or A[j,i]=1
    edge_mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0))

    if mode == 'all':
        return edge_mask

    # Expand TF mask to (B, G, G): tf_src[i,j] = True if i is TF
    tf_src = tf_mask.unsqueeze(2).expand(B, G, G)
    tf_tgt = tf_mask.unsqueeze(1).expand(B, G, G)

    if mode == 'tf_non_tf':
        # Source is TF, target is NOT TF
        role_mask = tf_src & (~tf_tgt)
    elif mode == 'tf_tf':
        # Source is TF, target is TF
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


@torch.no_grad()
def ensemble_evaluate(seeds_info, loader):
    all_probs = []
    all_labels = None

    for _, _, enc, dhead in seeds_info:
        enc.eval(); dhead.eval()
        seed_probs = []
        seed_labels = []
        for P, D_mat, N_mat, A_mat in loader:
            P, D_mat = P.to(device), D_mat.to(device)
            h = enc(P, D_mat)
            logits = dhead(h, P, D_mat)
            probs = torch.sigmoid(logits)

            label = A_mat.to(device).float()
            edge_mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0)).to(device)
            seed_probs.append(probs[edge_mask].cpu())
            seed_labels.append(label[edge_mask].cpu())

        all_probs.append(torch.cat(seed_probs, dim=0))
        if all_labels is None:
            all_labels = torch.cat(seed_labels, dim=0)

    prob_stack = torch.stack(all_probs, dim=0).mean(dim=0).numpy()
    label_stack = all_labels.numpy()
    auroc = roc_auc_score(label_stack, prob_stack)
    pred = (prob_stack > 0.5).astype(float)
    diracc = (pred == label_stack).mean()
    return auroc, diracc


def train_seed(seed, mode):
    checkpoint_root = PROJECT_ROOT / "checkpoints" / f"gt_g200_dir_specialist_{mode}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    edge_ckpt = edge_ckpt_root / f"seed_{seed}" / "best.pt"

    print(f"\n{'#'*60}")
    print(f"# DIR SPECIALIST SEED {seed}: mode={mode}")
    print(f"# P+D encoder, A label, loss only on {mode} edges")
    print(f"# Edge checkpoint: {edge_ckpt}")
    print(f"{'#'*60}")

    # -- Data ------------------------------------------
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

    # -- Model -----------------------------------------
    encoder = GraphTransformerEncoderV3(
        G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        sd_prob=sd_prob).to(device)
    state = torch.load(edge_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(state['encoder'])
    dir_head = AsymmetricDirHead(d_model=d_model).to(device)

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"DirHead: {sum(p.numel() for p in dir_head.parameters()):,}")

    # -- Train ----------------------------------------
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

            # Compute TF mask and specialist edge mask
            tf_mask = compute_tf_mask(A_mat)
            specialist_mask = get_specialist_mask(A_mat, tf_mask, mode)

            # Loss only on specialist edges
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
            print(f"[{mode}][seed{seed}][{step:5d}/{total_steps}] loss={loss.item():.4f} "
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
            print(f"  >>> {mode} SEED{seed} STEP {step}: val_DirAcc={val_diracc:.4f}  "
                  f"best={best_val:.4f} {best}  [{elapsed/60:.1f}min]")

    # -- Final ----------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{mode} SEED {seed} DONE in {elapsed/60:.1f}min")

    torch.save({'encoder': encoder.state_dict(),
                'dir_head': dir_head.state_dict()},
               checkpoint_dir / "final.pt")

    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder']); dir_head.load_state_dict(ckpt['dir_head'])

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    test_diracc = evaluate(encoder, dir_head, test_loader, mode=mode)
    print(f"  val DirAcc={best_val:.4f}  test DirAcc={test_diracc:.4f}")

    return test_diracc, 0.0, encoder, dir_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['tf_non_tf', 'tf_tf'])
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()

    checkpoint_root = PROJECT_ROOT / "checkpoints" / f"gt_g200_dir_specialist_{args.mode}"

    if args.ensemble:
        seed_dirs = sorted(checkpoint_root.glob("seed_*"))
        if not seed_dirs:
            print(f"No seeds found for {args.mode}. Run --mode {args.mode} --seed 0/1/2/3 first.")
            return

        print(f"\nLoading {len(seed_dirs)} seeds for {args.mode} ensemble...")
        full_ds = PDNDataset(cache_dir)
        n_total = len(full_ds)
        n_train, n_val = int(n_total * 0.8), int(n_total * 0.1)
        indices = np.random.RandomState(42).permutation(n_total)
        test_ds = Subset(full_ds, indices[n_train + n_val:])

        def collate(b):
            return (torch.stack([x[0] for x in b]),
                    torch.stack([x[1] for x in b]),
                    torch.stack([x[2] for x in b]),
                    torch.stack([x[3] for x in b]))

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=4, pin_memory=True)

        seeds_info = []
        for sd in seed_dirs:
            seed = int(sd.name.split('_')[1])
            ckpt = torch.load(sd / "best.pt", map_location=device, weights_only=True)
            encoder = GraphTransformerEncoderV3(
                G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                sd_prob=sd_prob).to(device)
            dir_head = AsymmetricDirHead(d_model=d_model).to(device)
            encoder.load_state_dict(ckpt['encoder'])
            dir_head.load_state_dict(ckpt['dir_head'])
            seeds_info.append((seed, sd, encoder, dir_head))
            print(f"  Loaded seed {seed}")

        ens_auroc, ens_diracc = ensemble_evaluate(seeds_info, test_loader)
        print(f"\n{'='*60}")
        print(f"ENSEMBLE {args.mode} ({len(seed_dirs)} seeds): "
              f"DirAcc={ens_diracc:.4f}, AUROC={ens_auroc:.4f}")
        print(f"{'='*60}")

        for seed, sd, enc, dhead in seeds_info:
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate, num_workers=4, pin_memory=True)
            diracc = evaluate(enc, dhead, test_loader, mode=args.mode)
            print(f"  seed {seed}: DirAcc={diracc:.4f}")
        return

    if args.seed >= 0:
        train_seed(args.seed, args.mode)
    else:
        print("Usage: --mode {tf_non_tf|tf_tf} --seed N  or  --mode {tf_non_tf|tf_tf} --ensemble")
        print(f"Checkpoints: {checkpoint_root}")


if __name__ == "__main__":
    main()
