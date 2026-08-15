#!/usr/bin/env python
"""Synthetic SCM cross-specialist evaluation.

Evaluates BOTH direction specialists on BOTH edge types:
  - NT specialist on TF->non-TF (matched)
  - NT specialist on TF->TF (cross)
  - TT specialist on TF->non-TF (cross)
  - TT specialist on TF->TF (matched)

Also computes each specialist on ALL edges (no edge-type filter).

Usage:
  python scripts/eval/synthetic_cross_specialist.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import CKPT_ROOT, RESULT_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from train_gt_g200_edge_v3 import (
    PDNDataset, GraphTransformerEncoderV3, G, d_model, n_heads, n_layers,
    dropout, sd_prob, batch_size, device
)
from train_gt_g200_dir_specialist import (
    GraphTransformerEncoderV3 as DirEncoder,
    AsymmetricDirHead,
    compute_tf_mask,
    get_specialist_mask,
)

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))

# Load test split (same as training: 80/10/10 split with seed 42)
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


@torch.no_grad()
def evaluate_cross(encoder, dir_head, loader, eval_mode):
    """Evaluate DirAcc on specified edge type.

    eval_mode: 'tf_non_tf', 'tf_tf', or 'all'
    """
    encoder.eval(); dir_head.eval()
    total_correct, total_pairs = 0, 0
    for P, D_mat, N_mat, A_mat in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        h = encoder(P, D_mat)
        logits = dir_head(h, P, D_mat)
        label = A_mat.to(device).float()

        if eval_mode == 'all':
            # All edge pairs (both directions)
            mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0))
        else:
            tf_mask = compute_tf_mask(A_mat)
            mask = get_specialist_mask(A_mat, tf_mask, eval_mode)

        pred = (logits > 0).float()
        total_correct += (pred[mask] == label[mask]).sum().item()
        total_pairs += mask.sum().item()
    return total_correct / max(total_pairs, 1)


def load_specialist(mode, seed=0):
    """Load a direction specialist checkpoint."""
    # Try release checkpoints first (flat naming)
    ckpt_path = CKPT_ROOT / "main" / f"dir_specialist_{mode}_seed{seed}.pt"
    print(f"  Trying: {ckpt_path} -> exists={ckpt_path.exists()}")
    if not ckpt_path.exists():
        # Try nested checkpoints (as produced by scripts/train/)
        ckpt_path = CKPT_ROOT / f"gt_g200_dir_specialist_{mode}" / f"seed_{seed}" / "best.pt"
        print(f"  Trying: {ckpt_path} -> exists={ckpt_path.exists()}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {mode} seed {seed}")

    enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                     dropout=dropout, sd_prob=sd_prob).to(device)
    head = AsymmetricDirHead(d_model=d_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    enc.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['dir_head'])
    return enc, head


print("Loading NT specialist (seed 0)...")
nt_enc, nt_head = load_specialist('tf_non_tf', seed=0)

print("Loading TT specialist (seed 0)...")
tt_enc, tt_head = load_specialist('tf_tf', seed=0)

print("\n" + "=" * 60)
print("SYNTHETIC SCM CROSS-SPECIALIST EVALUATION")
print("=" * 60)

results = {}
for name, enc, head in [("NT", nt_enc, nt_head), ("TT", tt_enc, tt_head)]:
    for eval_mode, label in [("tf_non_tf", "TF->non-TF"), ("tf_tf", "TF->TF"), ("all", "ALL")]:
        acc = evaluate_cross(enc, head, test_loader, eval_mode)
        results[f"{name}_on_{eval_mode}"] = acc
        print(f"  {name} specialist on {label:12s}: DirAcc = {acc:.4f}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Specialist':<12} {'TF->non-TF':<12} {'TF->TF':<12} {'ALL':<12}")
print("-" * 48)
for name in ["NT", "TT"]:
    row = f"{name:<12} "
    for mode in ["tf_non_tf", "tf_tf", "all"]:
        row += f"{results[f'{name}_on_{mode}']:.4f}       "
    print(row)

# Save results
import json
out_path = RESULT_ROOT / "1_synthetic" / "synthetic_cross_specialist.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
