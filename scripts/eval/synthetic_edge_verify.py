#!/usr/bin/env python
"""Verify edge prediction AUROC on current submission checkpoints."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys, os, json
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

from train_gt_g200_edge_v3 import (
    PDNDataset, GraphTransformerEncoderV3, EdgeHeadV3,
    G, d_model, n_heads, n_layers, dropout, sd_prob, batch_size, device
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _config import CKPT_ROOT, RESULT_ROOT

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))
ckpt_main = CKPT_ROOT / "main"

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
def eval_edge_auroc(encoder, edge_head, loader):
    encoder.eval(); edge_head.eval()
    all_probs, all_labels = [], []
    mask = ~np.eye(G, dtype=bool)
    for P, D_mat, N_mat, A_mat in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        h = encoder(P, D_mat)
        logits = edge_head(h, P, D_mat)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        labels = A_mat.cpu().numpy()
        for b in range(len(probs)):
            all_probs.append(probs[b][mask])
            all_labels.append(labels[b][mask])
    p = np.concatenate(all_probs)
    l = np.concatenate(all_labels)
    return roc_auc_score(l, p)

def load_edge_model(seed):
    ckpt_path = ckpt_main / f"edge_v3_seed{seed}.pt"
    if not ckpt_path.exists():
        return None, None
    enc = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                    dropout=dropout, sd_prob=sd_prob).to(device)
    head = EdgeHeadV3(d_model=d_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    enc.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['edge_head'])
    return enc, head

results = {}
# Per-seed
for seed in range(4):
    enc, head = load_edge_model(seed)
    if enc is None:
        print(f"edge seed {seed}: missing")
        continue
    auroc = eval_edge_auroc(enc, head, test_loader)
    results[f"edge_seed{seed}"] = auroc
    print(f"edge_v3 seed{seed}: test AUROC = {auroc:.4f}")

# Ensemble
print("\n--- edge 4-seed ensemble ---")
all_probs_seeds = []
all_labels = None
for seed in range(4):
    enc, head = load_edge_model(seed)
    if enc is None:
        continue
    enc.eval(); head.eval()
    seed_probs = []
    seed_labels = []
    mask = ~np.eye(G, dtype=bool)
    with torch.no_grad():
        for P, D_mat, N_mat, A_mat in test_loader:
            P, D_mat = P.to(device), D_mat.to(device)
            h = enc(P, D_mat)
            logits = head(h, P, D_mat)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            labels = A_mat.cpu().numpy()
            for b in range(len(probs)):
                seed_probs.append(probs[b][mask])
                seed_labels.append(labels[b][mask])
    all_probs_seeds.append(np.concatenate(seed_probs))
    if all_labels is None:
        all_labels = np.concatenate(seed_labels)

ens_probs = np.mean(all_probs_seeds, axis=0)
ens_auroc = roc_auc_score(all_labels, ens_probs)
results["edge_ensemble4"] = ens_auroc
print(f"edge ensemble (4 seed): test AUROC = {ens_auroc:.4f}")

out_path = RESULT_ROOT / "1_synthetic" / "synthetic_edge_authoritative.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
