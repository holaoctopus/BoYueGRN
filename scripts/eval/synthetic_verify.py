#!/usr/bin/env python
"""Authoritative re-evaluation of synthetic SCM specialists on submission checkpoints.

Resolves the discrepancy between:
  - synthetic_scm_validation.csv (NT test_DirAcc=0.908, parsed from old training stdout)
  - synthetic_cross_specialist.json (NT_on_tf_non_tf=0.978, cross-eval on current ckpt)

Uses the EXACT same logic as synthetic_cross_specialist.py:
  - Fixed test split: RandomState(42), indices[9000:10000] of 10000 datasets
  - evaluate_cross() with compute_tf_mask + get_specialist_mask
  - pred = (logits > 0), DirAcc = mean(pred[mask] == label[mask])

Evaluates ALL available seeds for both NT (tf_non_tf) and TT (tf_tf) specialists,
on their matched edge type, cross edge type, and ALL edges.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys, os, json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

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

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import CKPT_ROOT, RESULT_ROOT

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))
ckpt_main = CKPT_ROOT / "main"

# Fixed test split (matches cross-eval: RandomState(42))
full_ds = PDNDataset(cache_dir)
n_total = len(full_ds)
n_train, n_val = int(n_total * 0.8), int(n_total * 0.1)
indices = np.random.RandomState(42).permutation(n_total)
test_ds = Subset(full_ds, indices[n_train + n_val:])
print(f"Dataset: {n_total} total, test={len(test_ds)} (split seed 42)")


def collate(b):
    return (torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]),
            torch.stack([x[3] for x in b]))

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate_cross(encoder, dir_head, loader, eval_mode):
    """DirAcc on specified edge type. Identical to synthetic_cross_specialist.py."""
    encoder.eval(); dir_head.eval()
    total_correct, total_pairs = 0, 0
    for P, D_mat, N_mat, A_mat in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        h = encoder(P, D_mat)
        logits = dir_head(h, P, D_mat)
        label = A_mat.to(device).float()
        if eval_mode == 'all':
            mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0))
        else:
            tf_mask = compute_tf_mask(A_mat)
            mask = get_specialist_mask(A_mat, tf_mask, eval_mode)
        pred = (logits > 0).float()
        total_correct += (pred[mask] == label[mask]).sum().item()
        total_pairs += mask.sum().item()
    return total_correct / max(total_pairs, 1)


def load_specialist(mode, seed):
    ckpt_path = ckpt_main / f"dir_specialist_{mode}_seed{seed}.pt"
    if not ckpt_path.exists():
        return None
    enc = DirEncoder(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                     dropout=dropout, sd_prob=sd_prob).to(device)
    head = AsymmetricDirHead(d_model=d_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    enc.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['dir_head'])
    return enc, head


print("\n" + "=" * 70)
print("AUTHORITATIVE SYNTHETIC SCM EVALUATION (submission checkpoints)")
print("=" * 70)

results = {}
# NT specialist: 4 seeds
for seed in range(4):
    loaded = load_specialist('tf_non_tf', seed)
    if loaded is None:
        print(f"NT seed {seed}: checkpoint missing, skip")
        continue
    enc, head = loaded
    acc_matched = evaluate_cross(enc, head, test_loader, 'tf_non_tf')
    acc_cross = evaluate_cross(enc, head, test_loader, 'tf_tf')
    acc_all = evaluate_cross(enc, head, test_loader, 'all')
    results[f"NT_seed{seed}"] = {
        'tf_non_tf': acc_matched, 'tf_tf': acc_cross, 'all': acc_all
    }
    print(f"NT seed{seed}: matched(TF->non-TF)={acc_matched:.4f}  "
          f"cross(TF->TF)={acc_cross:.4f}  ALL={acc_all:.4f}")

# TT specialist: seed 0 (only one available)
loaded = load_specialist('tf_tf', 0)
if loaded is not None:
    enc, head = loaded
    acc_matched = evaluate_cross(enc, head, test_loader, 'tf_tf')
    acc_cross = evaluate_cross(enc, head, test_loader, 'tf_non_tf')
    acc_all = evaluate_cross(enc, head, test_loader, 'all')
    results["TT_seed0"] = {
        'tf_non_tf': acc_cross, 'tf_tf': acc_matched, 'all': acc_all
    }
    print(f"TT seed0: matched(TF->TF)={acc_matched:.4f}  "
          f"cross(TF->non-TF)={acc_cross:.4f}  ALL={acc_all:.4f}")

# NT ensemble (4-seed mean of probabilities)
print("\n--- NT 4-seed ensemble ---")
all_probs = []
all_labels = None
for seed in range(4):
    loaded = load_specialist('tf_non_tf', seed)
    if loaded is None:
        continue
    enc, head = loaded
    enc.eval(); head.eval()
    seed_probs = []
    seed_labels = []
    with torch.no_grad():
        for P, D_mat, N_mat, A_mat in test_loader:
            P, D_mat = P.to(device), D_mat.to(device)
            h = enc(P, D_mat)
            logits = head(h, P, D_mat)
            probs = torch.sigmoid(logits)
            label = A_mat.to(device).float()
            edge_mask = ((A_mat > 0) | (A_mat.transpose(1, 2) > 0)).to(device)
            seed_probs.append(probs[edge_mask].cpu())
            seed_labels.append(label[edge_mask].cpu())
    all_probs.append(torch.cat(seed_probs, dim=0))
    if all_labels is None:
        all_labels = torch.cat(seed_labels, dim=0)

if all_probs:
    prob_stack = torch.stack(all_probs, dim=0).mean(dim=0).numpy()
    label_stack = all_labels.numpy()
    pred = (prob_stack > 0.5).astype(float)
    ens_diracc = (pred == label_stack).mean()
    from sklearn.metrics import roc_auc_score
    ens_auroc = roc_auc_score(label_stack, prob_stack)
    results["NT_ensemble4"] = {
        'DirAcc_all': float(ens_diracc), 'AUROC_all': float(ens_auroc)
    }
    print(f"NT ensemble (4 seed, ALL edges): DirAcc={ens_diracc:.4f}, AUROC={ens_auroc:.4f}")

# Save
out_path = RESULT_ROOT / "1_synthetic" / "synthetic_authoritative_eval.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
